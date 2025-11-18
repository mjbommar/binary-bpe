//! Model types and helpers for working with trained BPE tokenizers.

use std::path::Path;

use ahash::AHashMap;
use tokenizers::decoders::{fuse::Fuse, DecoderWrapper};
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::{
    split::{Split as SplitPreTokenizer, SplitPattern},
    PreTokenizerWrapper,
};
use tokenizers::tokenizer::{AddedToken, SplitDelimiterBehavior};
use tokenizers::Tokenizer;

use crate::bytes::{bytes_to_latin1, latin1_to_bytes};
use crate::config::{PreprocessorConfig, PreprocessorKind, TrainerConfig};
use crate::error::{BbpeError, Result};
use crate::serialization::{save_huggingface_tokenizer, tokenizer_json};

/// Token identifier used throughout the crate.
pub type TokenId = u32;
/// Merge pair encoded as `(left, right)` token identifiers.
pub type Pair = (TokenId, TokenId);

/// Trained BPE model containing the learned vocabulary and merge table.
#[must_use]
#[derive(Debug, Clone)]
pub struct BpeModel {
    token_bytes: Vec<Vec<u8>>,
    merges: Vec<Pair>,
    special_tokens: Vec<String>,
    config: TrainerConfig,
}

/// Thin wrapper around `tokenizers::Tokenizer` that maintains binary vocab bytes.
#[must_use]
#[derive(Debug, Clone)]
pub struct BinaryTokenizer {
    inner: Tokenizer,
}

impl BpeModel {
    /// Constructs a new model from the supplied tokens, merges, and configuration.
    pub fn new(token_bytes: Vec<Vec<u8>>, merges: Vec<Pair>, config: TrainerConfig) -> Self {
        Self {
            token_bytes,
            merges,
            special_tokens: config.special_tokens.clone(),
            config,
        }
    }

    /// Returns the raw bytes backing each token in the base vocabulary.
    #[must_use]
    pub fn token_bytes(&self) -> &[Vec<u8>] {
        &self.token_bytes
    }

    /// Returns the merge table encoded as `(left, right)` token identifiers.
    #[must_use]
    pub fn merges(&self) -> &[Pair] {
        &self.merges
    }

    /// Returns the configured special tokens appended to the vocabulary.
    #[must_use]
    pub fn special_tokens(&self) -> &[String] {
        &self.special_tokens
    }

    /// Returns the [`TrainerConfig`] used to produce the model.
    #[must_use]
    pub fn trainer_config(&self) -> &TrainerConfig {
        &self.config
    }

    /// Returns the total vocabulary size including special tokens.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.token_bytes.len() + self.special_tokens.len()
    }

    /// Builds a Hugging Face [`Tokenizer`] representing the trained model.
    pub fn build_tokenizer(&self) -> Result<Tokenizer> {
        let vocab: AHashMap<String, TokenId> = self
            .token_bytes
            .iter()
            .enumerate()
            .map(|(idx, token)| (bytes_to_latin1(token), idx as TokenId))
            .collect();

        let merges: Vec<(String, String)> = self
            .merges
            .iter()
            .map(|&(left, right)| {
                let left = bytes_to_latin1(&self.token_bytes[left as usize]);
                let right = bytes_to_latin1(&self.token_bytes[right as usize]);
                (left, right)
            })
            .collect();

        let builder = BPE::builder()
            .vocab_and_merges(vocab, merges)
            .byte_fallback(true)
            .build()
            .map_err(|err| BbpeError::Tokenizers(err.to_string()))?;
        let mut tokenizer = Tokenizer::new(builder);

        if let Some(pre_tokenizer) = build_pre_tokenizer(&self.config.preprocessor)? {
            tokenizer.with_pre_tokenizer(Some(pre_tokenizer));
        } else {
            tokenizer.with_pre_tokenizer(None::<PreTokenizerWrapper>);
        }
        tokenizer.with_decoder(Some(DecoderWrapper::Fuse(Fuse::new())));

        if !self.special_tokens.is_empty() {
            let added = self
                .special_tokens
                .iter()
                .map(|token| AddedToken::from(token.clone(), true))
                .collect::<Vec<_>>();
            tokenizer.add_special_tokens(&added);
        }

        Ok(tokenizer)
    }

    /// Serialises the tokenizer to disk in Hugging Face JSON format.
    pub fn save_huggingface<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        save_huggingface_tokenizer(self, path, false)
    }

    /// Serialises the tokenizer to a JSON string in Hugging Face format.
    pub fn to_huggingface_json(&self, pretty: bool) -> Result<String> {
        tokenizer_json(self, pretty)
    }

    /// Builds a [`BinaryTokenizer`] helper wrapping the Hugging Face tokenizer.
    pub fn binary_tokenizer(&self) -> Result<BinaryTokenizer> {
        BinaryTokenizer::from_model(self)
    }

    /// Creates a derived model trimmed to the requested vocabulary size while preserving merge order.
    pub fn derive_with_vocab(&self, target_vocab_size: usize) -> Result<Self> {
        let special_count = self.special_tokens.len();
        let min_vocab = 256 + special_count;
        if target_vocab_size < min_vocab {
            return Err(BbpeError::InvalidConfig(format!(
                "requested family vocab {target_vocab_size} is smaller than required minimum {min_vocab}"
            )));
        }
        let current_total = self.vocab_size();
        if target_vocab_size > current_total {
            return Err(BbpeError::InvalidConfig(format!(
                "requested family vocab {target_vocab_size} exceeds trained size {current_total}"
            )));
        }

        let target_base = target_vocab_size - special_count;
        if target_base > self.token_bytes.len() {
            return Err(BbpeError::Internal(
                "computed base vocabulary exceeds stored tokens".into(),
            ));
        }
        let target_merges = target_base.saturating_sub(256);
        let token_bytes = self.token_bytes[..target_base].to_vec();
        let merges = self.merges[..target_merges].to_vec();

        let mut config = self.config.clone();
        config.target_vocab_size = target_vocab_size;
        if let Some(limit) = config.max_merge_iterations {
            config.max_merge_iterations = Some(limit.min(target_merges));
        }

        Ok(BpeModel::new(token_bytes, merges, config))
    }
}

impl BinaryTokenizer {
    /// Wraps an existing Hugging Face [`Tokenizer`], extracting binary vocab bytes.
    pub fn from_tokenizer(tokenizer: Tokenizer) -> Result<Self> {
        Ok(Self { inner: tokenizer })
    }

    /// Builds a [`BinaryTokenizer`] from a trained [`BpeModel`].
    pub fn from_model(model: &BpeModel) -> Result<Self> {
        Self::from_tokenizer(model.build_tokenizer()?)
    }

    /// Provides immutable access to the underlying tokenizer.
    #[must_use]
    pub fn inner(&self) -> &Tokenizer {
        &self.inner
    }

    /// Encodes raw bytes into token identifiers.
    pub fn encode_bytes(&self, data: &[u8], add_special_tokens: bool) -> Result<Vec<TokenId>> {
        let text = bytes_to_latin1(data);
        let encoding = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|err| BbpeError::Tokenizers(err.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decodes token identifiers back into raw bytes.
    pub fn decode_to_bytes(
        &self,
        tokens: &[TokenId],
        skip_special_tokens: bool,
    ) -> Result<Vec<u8>> {
        let ids: Vec<u32> = tokens.to_vec();
        let text = self
            .inner
            .decode(&ids, skip_special_tokens)
            .map_err(|err| BbpeError::Tokenizers(err.to_string()))?;
        Ok(latin1_to_bytes(&text))
    }
}

const ASCII_WHITESPACE_REGEX: &str = r"[ \t\n\r\x0B\x0C]+";
const UNICODE_WHITESPACE_REGEX: &str = r"[\x{0009}\x{000A}\x{000B}\x{000C}\x{000D}\x{0020}\x{0085}\x{00A0}\x{1680}\x{2000}\x{2001}\x{2002}\x{2003}\x{2004}\x{2005}\x{2006}\x{2007}\x{2008}\x{2009}\x{200A}\x{2028}\x{2029}\x{202F}\x{205F}\x{3000}\x{FEFF}]+";
const NULL_DELIMITED_REGEX: &str = r"\x00+";

fn build_pre_tokenizer(config: &PreprocessorConfig) -> Result<Option<PreTokenizerWrapper>> {
    if config.split_probability < 1.0 {
        return Ok(None);
    }
    let behavior = SplitDelimiterBehavior::Isolated;
    let make_split = |pattern: &str| -> Result<PreTokenizerWrapper> {
        let split = SplitPreTokenizer::new(SplitPattern::Regex(pattern.into()), behavior, false)
            .map_err(|err| BbpeError::Tokenizers(err.to_string()))?;
        Ok(PreTokenizerWrapper::Split(split))
    };

    let wrapper = match config.kind {
        PreprocessorKind::None => return Ok(None),
        PreprocessorKind::AsciiWhitespace => make_split(ASCII_WHITESPACE_REGEX)?,
        PreprocessorKind::UnicodeWhitespace => make_split(UNICODE_WHITESPACE_REGEX)?,
        PreprocessorKind::NullDelimited => make_split(NULL_DELIMITED_REGEX)?,
    };

    Ok(Some(wrapper))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn sample_model(vocab: usize, special_tokens: Vec<String>) -> BpeModel {
        let mut token_bytes = Vec::new();
        for idx in 0..vocab {
            token_bytes.push(vec![(idx % 256) as u8]);
        }
        let merges = (0..(vocab.saturating_sub(256)))
            .map(|i| (i as TokenId, (i + 1) as TokenId))
            .collect::<Vec<_>>();
        let cfg = TrainerConfig::builder()
            .target_vocab_size(vocab + special_tokens.len())
            .special_tokens(special_tokens.clone())
            .show_progress(false)
            .build()
            .expect("valid config");
        BpeModel::new(token_bytes, merges, cfg)
    }

    fn tiny_model() -> BpeModel {
        let mut token_bytes: Vec<Vec<u8>> = (0u8..=255).map(|b| vec![b]).collect();
        token_bytes.push(b"hi".to_vec());
        let merges: Vec<Pair> = vec![(b'h' as TokenId, b'i' as TokenId)];
        let config = TrainerConfig {
            special_tokens: vec!["<|pad|>".into()],
            show_progress: false,
            ..TrainerConfig::default()
        };
        BpeModel::new(token_bytes, merges, config)
    }

    #[test]
    fn derive_with_vocab_trims_merges_and_tokens() {
        let model = sample_model(300, Vec::new());
        let derived = model.derive_with_vocab(280).expect("derive");
        assert_eq!(derived.token_bytes().len(), 280);
        assert_eq!(derived.merges().len(), 24);
    }

    #[test]
    fn derive_with_vocab_preserves_special_tokens() {
        let model = sample_model(280, vec!["<|start|>".into(), "<|end|>".into()]);
        let derived = model.derive_with_vocab(262).expect("derive with specials");
        assert_eq!(derived.special_tokens(), model.special_tokens());
        assert_eq!(derived.vocab_size(), 262);
    }

    #[test]
    fn binary_tokenizer_round_trips_ascii() {
        let model = tiny_model();
        let tokenizer = model.binary_tokenizer().expect("tokenizer");
        let encoded = tokenizer
            .encode_bytes(b"hi", false)
            .expect("encode should work");
        assert_eq!(encoded, vec![256]);
        let decoded = tokenizer
            .decode_to_bytes(&encoded, false)
            .expect("decode should work");
        assert_eq!(decoded, b"hi");
    }

    #[test]
    fn huggingface_json_is_well_formed() {
        let model = tiny_model();
        let json = model
            .to_huggingface_json(true)
            .expect("serialization should work");
        let value: Value = serde_json::from_str(&json).expect("valid json");
        assert_eq!(value["model"]["type"], "BPE");
    }
}
