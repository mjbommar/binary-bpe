//! Model types and helpers for working with trained BPE tokenizers.

use std::path::Path;

use ahash::AHashMap;
use rustc_hash::FxHashSet;
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
use crate::special_tokens;

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
    vocab_bytes: Vec<Vec<u8>>,
    special_ids: FxHashSet<TokenId>,
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
        self.token_bytes.len()
    }

    /// Builds a Hugging Face [`Tokenizer`] representing the trained model.
    pub fn build_tokenizer(&self) -> Result<Tokenizer> {
        let vocab: AHashMap<String, TokenId> = self
            .token_bytes
            .iter()
            .enumerate()
            .map(|(idx, _)| (self.token_string(idx), idx as TokenId))
            .collect();

        let merges: Vec<(String, String)> = self
            .merges
            .iter()
            .map(|&(left, right)| {
                let left = self.token_string(left as usize);
                let right = self.token_string(right as usize);
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
        let leading_count = special_tokens::leading_tokens().len();
        let trailing_count = self.special_tokens.len();
        let min_vocab = leading_count + 256 + trailing_count;
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

        let initial_vocab = min_vocab;
        if target_vocab_size > self.token_bytes.len() {
            return Err(BbpeError::Internal(
                "computed vocabulary exceeds stored tokens".into(),
            ));
        }
        let target_merges = target_vocab_size.saturating_sub(initial_vocab);
        let token_bytes = self.token_bytes[..target_vocab_size].to_vec();
        let merges = self.merges[..target_merges].to_vec();

        let mut config = self.config.clone();
        config.target_vocab_size = target_vocab_size;
        if let Some(limit) = config.max_merge_iterations {
            config.max_merge_iterations = Some(limit.min(target_merges));
        }

        Ok(BpeModel::new(token_bytes, merges, config))
    }

    fn token_string(&self, idx: usize) -> String {
        let leading = special_tokens::leading_tokens();
        if idx < leading.len() {
            return leading[idx].clone();
        }
        let byte_start = leading.len();
        let trailing_start = byte_start + 256;
        if idx >= trailing_start && idx < trailing_start + self.special_tokens.len() {
            return self.special_tokens[idx - trailing_start].clone();
        }
        bytes_to_latin1(&self.token_bytes[idx])
    }
}

impl BinaryTokenizer {
    /// Wraps an existing Hugging Face [`Tokenizer`], extracting binary vocab bytes.
    pub fn from_tokenizer(tokenizer: Tokenizer) -> Result<Self> {
        let mut entries: Vec<(String, TokenId)> = tokenizer.get_vocab(true).into_iter().collect();
        entries.sort_by_key(|(_, id)| *id);
        let vocab_bytes = entries
            .into_iter()
            .map(|(token, _)| latin1_to_bytes(&token))
            .collect::<Vec<_>>();
        let mut special_ids = FxHashSet::default();
        for (id, token) in tokenizer.get_added_tokens_decoder() {
            if token.special {
                special_ids.insert(id);
            }
        }
        if special_ids.is_empty() {
            return Err(BbpeError::InvalidConfig(
                "tokenizer is missing special token metadata; regenerate with the latest bbpe"
                    .into(),
            ));
        }
        Ok(Self {
            inner: tokenizer,
            vocab_bytes,
            special_ids,
        })
    }

    /// Builds a [`BinaryTokenizer`] from a trained [`BpeModel`].
    pub fn from_model(model: &BpeModel) -> Result<Self> {
        let tokenizer = model.build_tokenizer()?;
        let mut entries: Vec<(String, TokenId)> = tokenizer.get_vocab(true).into_iter().collect();
        entries.sort_by_key(|(_, id)| *id);
        let vocab_bytes = entries
            .into_iter()
            .map(|(token, _)| latin1_to_bytes(&token))
            .collect::<Vec<_>>();
        let leading = special_tokens::leading_tokens().len();
        let trailing_start = leading + 256;
        let mut special_ids = FxHashSet::default();
        for id in 0..leading {
            special_ids.insert(id as TokenId);
        }
        for offset in 0..model.special_tokens.len() {
            special_ids.insert((trailing_start + offset) as TokenId);
        }
        Ok(Self {
            inner: tokenizer,
            vocab_bytes,
            special_ids,
        })
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
        let mut bytes = Vec::new();
        for &id in tokens {
            let idx = id as usize;
            if idx >= self.vocab_bytes.len() {
                return Err(BbpeError::Internal(format!(
                    "token id {} exceeds vocab size {}",
                    id,
                    self.vocab_bytes.len()
                )));
            }
            if skip_special_tokens && self.special_ids.contains(&id) {
                continue;
            }
            bytes.extend_from_slice(&self.vocab_bytes[idx]);
        }
        Ok(bytes)
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

    fn sample_model(total_vocab: usize, extra_special_tokens: Vec<String>) -> BpeModel {
        let leading = special_tokens::leading_tokens();
        let min_vocab = leading.len() + 256 + extra_special_tokens.len();
        assert!(
            total_vocab >= min_vocab,
            "sample_model requires total_vocab >= {min_vocab}"
        );
        let mut token_bytes = Vec::new();
        for token in leading {
            token_bytes.push(token.as_bytes().to_vec());
        }
        for byte in 0u8..=255 {
            token_bytes.push(vec![byte]);
        }
        for token in &extra_special_tokens {
            token_bytes.push(token.as_bytes().to_vec());
        }
        let mut merges = Vec::new();
        while token_bytes.len() < total_vocab {
            let left = leading.len() + (merges.len() % 256);
            let right = leading.len() + ((merges.len() + 1) % 256);
            merges.push((left as TokenId, right as TokenId));
            let mut combined = token_bytes[left].clone();
            combined.extend_from_slice(&token_bytes[right]);
            token_bytes.push(combined);
        }
        let cfg = TrainerConfig::builder()
            .target_vocab_size(total_vocab)
            .special_tokens(extra_special_tokens.clone())
            .reasoning_tokens_enabled(false)
            .show_progress(false)
            .build()
            .expect("valid config");
        BpeModel::new(token_bytes, merges, cfg)
    }

    fn tiny_model() -> BpeModel {
        sample_model(special_tokens::leading_tokens().len() + 257, Vec::new())
    }

    #[test]
    fn derive_with_vocab_trims_merges_and_tokens() {
        let model = sample_model(300, Vec::new());
        let derived = model.derive_with_vocab(280).expect("derive");
        assert_eq!(derived.token_bytes().len(), 280);
        assert_eq!(derived.merges().len(), 17);
    }

    #[test]
    fn derive_with_vocab_preserves_special_tokens() {
        let model = sample_model(285, vec!["<|bos|>".into(), "<|eos|>".into()]);
        let derived = model.derive_with_vocab(270).expect("derive with specials");
        assert_eq!(derived.special_tokens(), model.special_tokens());
        assert_eq!(derived.vocab_size(), 270);
    }

    #[test]
    fn binary_tokenizer_round_trips_ascii() {
        let model = tiny_model();
        let tokenizer = model.binary_tokenizer().expect("tokenizer");
        let encoded = tokenizer
            .encode_bytes(b"hi", false)
            .expect("encode should work");
        assert!(!encoded.is_empty());
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
