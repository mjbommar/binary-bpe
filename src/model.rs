//! Model types and helpers for working with trained BPE tokenizers.

use std::path::Path;

use ahash::AHashMap;
use rustc_hash::FxHashSet;
use tokenizers::decoders::{fuse::Fuse, DecoderWrapper};
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::{
    sequence::Sequence,
    split::{Split as SplitPreTokenizer, SplitPattern},
    PreTokenizerWrapper,
};
use tokenizers::tokenizer::{AddedToken, SplitDelimiterBehavior};
use tokenizers::Tokenizer;

use crate::bytes::{
    bytes_to_latin1, bytes_to_string, decode_legacy_with_overrides, encode_legacy_with_overrides,
    string_to_bytes, ByteEncoding, LegacyByteOverrides,
};
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
    input_encoding: ByteEncoding,
    legacy_overrides: LegacyByteOverrides,
}

/// Controls how legacy non-ByteLevel tokenizer JSONs handle byte/special-token collisions.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum LegacyByteBehavior {
    /// Detect private-use legacy byte escapes from the tokenizer vocabulary.
    #[default]
    Auto,
    /// Treat legacy tokenizer strings as direct Latin-1 bytes.
    Plain,
    /// Escape bytes that collide with actual single-codepoint Latin-1 special tokens.
    Escaped,
}

/// Options used when wrapping an existing Hugging Face tokenizer.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BinaryTokenizerOptions {
    legacy_byte_behavior: LegacyByteBehavior,
}

impl BinaryTokenizerOptions {
    /// Sets the legacy byte collision behavior.
    #[must_use]
    pub fn legacy_byte_behavior(mut self, behavior: LegacyByteBehavior) -> Self {
        self.legacy_byte_behavior = behavior;
        self
    }

    /// Returns the configured legacy byte collision behavior.
    #[must_use]
    pub fn legacy_byte_behavior_value(&self) -> LegacyByteBehavior {
        self.legacy_byte_behavior
    }
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
        let reasoning_count = if self.config.reasoning_tokens_enabled {
            special_tokens::reasoning_tokens().len()
        } else {
            0
        };
        let trailing_count = reasoning_count + self.special_tokens.len();
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
        let reasoning_count = if self.config.reasoning_tokens_enabled {
            special_tokens::reasoning_tokens().len()
        } else {
            0
        };
        let custom_start = trailing_start + reasoning_count;
        if idx >= custom_start && idx < custom_start + self.special_tokens.len() {
            return self.special_tokens[idx - custom_start].clone();
        }
        bytes_to_latin1(&self.token_bytes[idx])
    }
}

impl BinaryTokenizer {
    /// Wraps an existing Hugging Face [`Tokenizer`], extracting binary vocab bytes.
    pub fn from_tokenizer(tokenizer: Tokenizer) -> Result<Self> {
        Self::from_tokenizer_with_options(tokenizer, BinaryTokenizerOptions::default())
    }

    /// Wraps an existing Hugging Face [`Tokenizer`] with explicit loading options.
    pub fn from_tokenizer_with_options(
        tokenizer: Tokenizer,
        options: BinaryTokenizerOptions,
    ) -> Result<Self> {
        let has_byte_level_decoder =
            matches!(tokenizer.get_decoder(), Some(DecoderWrapper::ByteLevel(_)));
        let vocab = tokenizer.get_vocab(true);
        let added_tokens = tokenizer.get_added_tokens_decoder();
        let mut special_token_strings = added_tokens
            .iter()
            .filter(|(_, token)| token.special)
            .map(|(id, token)| (*id, token.content.as_str()))
            .collect::<Vec<_>>();
        for token in special_tokens::reasoning_tokens() {
            if let Some(id) = vocab.get(token) {
                special_token_strings.push((*id, token.as_str()));
            }
        }
        special_token_strings.sort_by_key(|(id, _)| *id);
        special_token_strings.dedup_by_key(|(_, token)| *token);
        let candidate_overrides = LegacyByteOverrides::from_special_tokens(
            special_token_strings.iter().map(|(_, token)| *token),
        );
        let legacy_overrides = if has_byte_level_decoder {
            LegacyByteOverrides::default()
        } else {
            match options.legacy_byte_behavior {
                LegacyByteBehavior::Auto => {
                    if tokenizer_uses_legacy_overrides(&tokenizer, &candidate_overrides) {
                        candidate_overrides
                    } else {
                        LegacyByteOverrides::default()
                    }
                }
                LegacyByteBehavior::Plain => LegacyByteOverrides::default(),
                LegacyByteBehavior::Escaped => candidate_overrides,
            }
        };
        let encoding = match tokenizer.get_decoder() {
            Some(DecoderWrapper::ByteLevel(_)) => ByteEncoding::Gpt2,
            _ if legacy_overrides.is_empty() => ByteEncoding::LegacyPlain,
            _ => ByteEncoding::Legacy,
        };
        let mut tokenizer = tokenizer;
        strip_byte_level_pre_tokenizer(&mut tokenizer);
        let mut entries: Vec<(String, TokenId)> = tokenizer.get_vocab(true).into_iter().collect();
        entries.sort_by_key(|(_, id)| *id);
        let vocab_bytes = entries
            .into_iter()
            .map(|(token, _)| decode_token_string(&token, encoding, &legacy_overrides))
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
            input_encoding: encoding,
            legacy_overrides,
        })
    }

    /// Builds a [`BinaryTokenizer`] from a trained [`BpeModel`].
    pub fn from_model(model: &BpeModel) -> Result<Self> {
        let tokenizer = model.build_tokenizer()?;
        let mut entries: Vec<(String, TokenId)> = tokenizer.get_vocab(true).into_iter().collect();
        entries.sort_by_key(|(_, id)| *id);
        let vocab_bytes = entries
            .into_iter()
            .map(|(token, _)| string_to_bytes(&token, ByteEncoding::Gpt2))
            .collect::<Vec<_>>();
        let leading = special_tokens::leading_tokens().len();
        let trailing_start = leading + 256;
        let reasoning_count = if model.config.reasoning_tokens_enabled {
            special_tokens::reasoning_tokens().len()
        } else {
            0
        };
        let mut special_ids = FxHashSet::default();
        for id in 0..leading {
            special_ids.insert(id as TokenId);
        }
        for offset in 0..model.special_tokens.len() {
            special_ids.insert((trailing_start + reasoning_count + offset) as TokenId);
        }
        Ok(Self {
            inner: tokenizer,
            vocab_bytes,
            special_ids,
            input_encoding: ByteEncoding::Gpt2,
            legacy_overrides: LegacyByteOverrides::default(),
        })
    }

    /// Provides immutable access to the underlying tokenizer.
    #[must_use]
    pub fn inner(&self) -> &Tokenizer {
        &self.inner
    }

    /// Returns the byte-string encoding used to feed the wrapped tokenizer.
    #[must_use]
    pub fn input_encoding(&self) -> ByteEncoding {
        self.input_encoding
    }

    /// Encodes raw bytes into token identifiers.
    pub fn encode_bytes(&self, data: &[u8], add_special_tokens: bool) -> Result<Vec<TokenId>> {
        let text = encode_input_bytes(data, self.input_encoding, &self.legacy_overrides);
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

fn encode_input_bytes(
    data: &[u8],
    encoding: ByteEncoding,
    legacy_overrides: &LegacyByteOverrides,
) -> String {
    match encoding {
        ByteEncoding::Legacy => encode_legacy_with_overrides(data, legacy_overrides),
        other => bytes_to_string(data, other),
    }
}

fn decode_token_string(
    token: &str,
    encoding: ByteEncoding,
    legacy_overrides: &LegacyByteOverrides,
) -> Vec<u8> {
    match encoding {
        ByteEncoding::Legacy => decode_legacy_with_overrides(token, legacy_overrides),
        other => string_to_bytes(token, other),
    }
}

fn tokenizer_uses_legacy_overrides(tokenizer: &Tokenizer, overrides: &LegacyByteOverrides) -> bool {
    if overrides.is_empty() {
        return false;
    }
    let vocab = tokenizer.get_vocab(true);
    overrides
        .replacement_chars()
        .any(|ch| vocab.contains_key(&ch.to_string()))
}

fn strip_byte_level_pre_tokenizer(tokenizer: &mut Tokenizer) {
    let replacement = tokenizer
        .get_pre_tokenizer()
        .cloned()
        .and_then(remove_byte_level_wrapper);
    match replacement {
        Some(wrapper) => {
            tokenizer.with_pre_tokenizer(Some(wrapper));
        }
        None => {
            tokenizer.with_pre_tokenizer(None::<PreTokenizerWrapper>);
        }
    }
}

fn remove_byte_level_wrapper(wrapper: PreTokenizerWrapper) -> Option<PreTokenizerWrapper> {
    match wrapper {
        PreTokenizerWrapper::ByteLevel(_) => None,
        PreTokenizerWrapper::Sequence(seq) => {
            let filtered: Vec<PreTokenizerWrapper> = seq
                .into_iter()
                .filter_map(remove_byte_level_wrapper)
                .collect();
            match filtered.len() {
                0 => None,
                1 => filtered.into_iter().next(),
                _ => Some(PreTokenizerWrapper::Sequence(Sequence::new(filtered))),
            }
        }
        other => Some(other),
    }
}

const ASCII_WHITESPACE_REGEX: &str = r"[ \t\n\r\x0B\x0C]+";
const UNICODE_WHITESPACE_REGEX: &str = r"[\x{0009}\x{000A}\x{000B}\x{000C}\x{000D}\x{0020}\x{0085}\x{00A0}\x{1680}\x{2000}\x{2001}\x{2002}\x{2003}\x{2004}\x{2005}\x{2006}\x{2007}\x{2008}\x{2009}\x{200A}\x{2028}\x{2029}\x{202F}\x{205F}\x{3000}\x{FEFF}]+";
const NULL_DELIMITED_REGEX: &str = r"\x00+";

pub(crate) fn build_pre_tokenizer(
    config: &PreprocessorConfig,
) -> Result<Option<PreTokenizerWrapper>> {
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

    fn legacy_latin1_tokenizer(escaped: bool) -> Tokenizer {
        let mut vocab = AHashMap::new();
        for byte in 0u8..=255 {
            let token = match (escaped, byte) {
                (true, 0xAB) => "\u{E000}".to_string(),
                (true, 0xBB) => "\u{E001}".to_string(),
                _ => (byte as char).to_string(),
            };
            vocab.insert(token, byte as TokenId);
        }
        if escaped {
            vocab.insert("«".to_string(), 256);
            vocab.insert("»".to_string(), 257);
        }
        let model = BPE::builder()
            .vocab_and_merges(vocab, Vec::<(String, String)>::new())
            .byte_fallback(true)
            .build()
            .expect("valid bpe model");
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_decoder(Some(DecoderWrapper::Fuse(Fuse::new())));
        tokenizer.add_special_tokens(&[AddedToken::from("<|sep|>".to_string(), true)]);
        tokenizer
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
    fn legacy_plain_tokenizer_keeps_ab_bb_bytes_lossless() {
        let tokenizer =
            BinaryTokenizer::from_tokenizer(legacy_latin1_tokenizer(false)).expect("tokenizer");
        assert_eq!(tokenizer.input_encoding(), ByteEncoding::LegacyPlain);
        let bytes = [0x00, 0xAB, 0xBB, 0xAD, 0xFF];
        let encoded = tokenizer.encode_bytes(&bytes, false).expect("encode");
        let decoded = tokenizer.decode_to_bytes(&encoded, false).expect("decode");
        assert_eq!(decoded, bytes);
    }

    #[test]
    fn legacy_escaped_tokenizer_auto_detects_private_use_bytes() {
        let tokenizer =
            BinaryTokenizer::from_tokenizer(legacy_latin1_tokenizer(true)).expect("tokenizer");
        assert_eq!(tokenizer.input_encoding(), ByteEncoding::Legacy);
        let bytes = [0xAB, 0xBB, 0xAB, 0xBB];
        let encoded = tokenizer.encode_bytes(&bytes, false).expect("encode");
        let decoded = tokenizer.decode_to_bytes(&encoded, false).expect("decode");
        assert_eq!(decoded, bytes);
    }

    #[test]
    fn legacy_byte_behavior_can_force_plain_mode() {
        let options =
            BinaryTokenizerOptions::default().legacy_byte_behavior(LegacyByteBehavior::Plain);
        let tokenizer =
            BinaryTokenizer::from_tokenizer_with_options(legacy_latin1_tokenizer(true), options)
                .expect("tokenizer");
        assert_eq!(tokenizer.input_encoding(), ByteEncoding::LegacyPlain);
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
