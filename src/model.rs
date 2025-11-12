//! Model types and helpers for working with trained BPE tokenizers.

use std::path::Path;

use ahash::AHashMap;
use tokenizers::decoders::{fuse::Fuse, DecoderWrapper};
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::tokenizer::AddedToken;
use tokenizers::Tokenizer;

use crate::bytes::{bytes_to_latin1, latin1_to_bytes};
use crate::config::TrainerConfig;
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
    vocab_bytes: Vec<Vec<u8>>,
    base_vocab_size: usize,
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

        tokenizer.with_pre_tokenizer(None::<PreTokenizerWrapper>);
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
}

impl BinaryTokenizer {
    /// Wraps an existing Hugging Face [`Tokenizer`], extracting binary vocab bytes.
    pub fn from_tokenizer(tokenizer: Tokenizer) -> Result<Self> {
        let base_vocab_size = tokenizer.get_vocab(false).len();
        let mut entries: Vec<(String, TokenId)> = tokenizer.get_vocab(true).into_iter().collect();
        entries.sort_by_key(|(_, id)| *id);
        let vocab_bytes = entries
            .into_iter()
            .map(|(token, _)| latin1_to_bytes(&token))
            .collect::<Vec<_>>();
        Ok(Self {
            inner: tokenizer,
            vocab_bytes,
            base_vocab_size,
        })
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
            if skip_special_tokens && idx >= self.base_vocab_size {
                continue;
            }
            bytes.extend_from_slice(&self.vocab_bytes[idx]);
        }
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

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
