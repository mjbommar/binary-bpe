//! Configuration builders controlling training and corpus ingestion.

use std::convert::TryFrom;

use crate::error::{BbpeError, Result};
use serde::{Deserialize, Serialize};

/// Configuration for binary BPE training.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrainerConfig {
    /// Target vocabulary size including the 256 base byte tokens and special tokens.
    pub target_vocab_size: usize,
    /// Minimum number of pair occurrences required before a merge is considered.
    pub min_frequency: usize,
    /// Inclusive list of allowed token lengths (in bytes) produced by merges.
    pub allowed_token_lengths: Vec<usize>,
    /// Enables per-iteration logging through the `log` facade.
    pub show_progress: bool,
    /// Additional tokens appended to the vocabulary after training.
    pub special_tokens: Vec<String>,
    /// Frequency threshold below which merges are considered part of a plateau.
    pub plateau_frequency_floor: usize,
    /// Number of consecutive plateau iterations before considering early stopping.
    pub plateau_patience: usize,
    /// Ratio between the initial and current pair frequency that also signals a plateau.
    pub plateau_frequency_divisor: usize,
    /// Hard cap on merge iterations; `None` uses the target vocabulary size.
    pub max_merge_iterations: Option<usize>,
    /// Enables plateau-based early stopping with `plateau_patience`.
    pub plateau_stop_enabled: bool,
}

impl TrainerConfig {
    /// Returns a builder initialised with [`TrainerConfig::default`].
    #[must_use]
    pub fn builder() -> TrainerBuilder {
        TrainerBuilder::default()
    }

    /// Validates the invariants required for training.
    pub fn validate(&self) -> Result<()> {
        if self.target_vocab_size < 256 + self.special_tokens.len() {
            return Err(BbpeError::InvalidConfig(format!(
                "target_vocab_size ({}) must be at least 256 + special tokens ({}).",
                self.target_vocab_size,
                self.special_tokens.len()
            )));
        }
        if self.min_frequency == 0 {
            return Err(BbpeError::InvalidConfig(
                "min_frequency must be greater than zero".into(),
            ));
        }
        let max_vocab = usize::try_from(u32::MAX).unwrap_or(usize::MAX);
        if self.target_vocab_size > max_vocab {
            return Err(BbpeError::InvalidConfig(format!(
                "target_vocab_size ({}) exceeds {max_vocab}, the maximum representable TokenId",
                self.target_vocab_size
            )));
        }
        if !self.allowed_token_lengths.contains(&1) {
            return Err(BbpeError::InvalidConfig(
                "allowed_token_lengths must include the base length of 1".into(),
            ));
        }
        if self.plateau_frequency_divisor == 0 {
            return Err(BbpeError::InvalidConfig(
                "plateau_frequency_divisor must be greater than zero".into(),
            ));
        }
        if self.plateau_stop_enabled && self.plateau_patience == 0 {
            return Err(BbpeError::InvalidConfig(
                "plateau_patience must be > 0 when plateau_stop_enabled is true".into(),
            ));
        }
        if self.allowed_token_lengths.is_empty() {
            return Err(BbpeError::InvalidConfig(
                "allowed_token_lengths must not be empty".into(),
            ));
        }
        Ok(())
    }
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            target_vocab_size: 32_768,
            min_frequency: 4,
            allowed_token_lengths: (1..=32).collect(),
            show_progress: true,
            special_tokens: vec![
                "<|start|>".into(),
                "<|end|>".into(),
                "<|pad|>".into(),
                "<|unk|>".into(),
                "<|cls|>".into(),
                "<|sep|>".into(),
                "<|mask|>".into(),
            ],
            plateau_frequency_floor: 128,
            plateau_patience: 32,
            plateau_frequency_divisor: 512,
            max_merge_iterations: None,
            plateau_stop_enabled: false,
        }
    }
}

/// Builder for [`TrainerConfig`].
#[derive(Debug, Default, Clone)]
pub struct TrainerBuilder {
    cfg: TrainerConfig,
}

impl TrainerBuilder {
    /// Creates a builder with [`TrainerConfig::default`] settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the desired vocabulary size (including base byte tokens).
    #[must_use]
    pub fn target_vocab_size(mut self, value: usize) -> Self {
        self.cfg.target_vocab_size = value;
        self
    }

    /// Sets the minimum merge frequency.
    #[must_use]
    pub fn min_frequency(mut self, value: usize) -> Self {
        self.cfg.min_frequency = value;
        self
    }

    /// Overrides the allowed token lengths.
    #[must_use]
    pub fn allowed_token_lengths<I>(mut self, lengths: I) -> Self
    where
        I: IntoIterator<Item = usize>,
    {
        self.cfg.allowed_token_lengths = lengths.into_iter().collect();
        self
    }

    /// Enables or disables per-iteration logging.
    #[must_use]
    pub fn show_progress(mut self, enabled: bool) -> Self {
        self.cfg.show_progress = enabled;
        self
    }

    /// Overrides the set of special tokens appended to the vocabulary.
    #[must_use]
    pub fn special_tokens<I, S>(mut self, tokens: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.cfg.special_tokens = tokens.into_iter().map(|s| s.into()).collect();
        self
    }

    /// Configures plateau-based early stopping thresholds.
    #[must_use]
    pub fn plateau_thresholds(mut self, floor: usize, patience: usize, divisor: usize) -> Self {
        self.cfg.plateau_frequency_floor = floor;
        self.cfg.plateau_patience = patience;
        self.cfg.plateau_frequency_divisor = divisor;
        self
    }

    /// Enables plateau-based early stopping.
    #[must_use]
    pub fn plateau_stop_enabled(mut self, enabled: bool) -> Self {
        self.cfg.plateau_stop_enabled = enabled;
        self
    }

    /// Sets a hard merge iteration limit.
    #[must_use]
    pub fn max_merge_iterations(mut self, value: Option<usize>) -> Self {
        self.cfg.max_merge_iterations = value;
        self
    }

    /// Finalises the builder, returning a validated [`TrainerConfig`].
    pub fn build(mut self) -> Result<TrainerConfig> {
        self.cfg.allowed_token_lengths.sort_unstable();
        self.cfg.allowed_token_lengths.dedup();
        self.cfg.validate()?;
        Ok(self.cfg)
    }
}

/// Configuration controlling how binary corpora are read from disk.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IngestConfig {
    /// Size of chunks to read from each input file; `0` reads entire files.
    pub chunk_size: usize,
    /// Enables recursive directory traversal.
    pub recursive: bool,
    /// Follows symlinks encountered during traversal.
    pub follow_symlinks: bool,
}

impl Default for IngestConfig {
    fn default() -> Self {
        Self {
            chunk_size: 8192,
            recursive: true,
            follow_symlinks: false,
        }
    }
}

impl IngestConfig {
    /// Returns a builder initialised with [`IngestConfig::default`].
    #[must_use]
    pub fn builder() -> IngestBuilder {
        IngestBuilder::default()
    }
}

/// Builder for [`IngestConfig`].
#[derive(Debug, Default, Clone)]
pub struct IngestBuilder {
    cfg: IngestConfig,
}

impl IngestBuilder {
    /// Creates a new builder with [`IngestConfig::default`] settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the chunk size in bytes (0 = read entire file at once).
    #[must_use]
    pub fn chunk_size(mut self, size: usize) -> Self {
        self.cfg.chunk_size = size;
        self
    }

    /// Enables or disables recursive directory traversal.
    #[must_use]
    pub fn recursive(mut self, enabled: bool) -> Self {
        self.cfg.recursive = enabled;
        self
    }

    /// Enables or disables following of symlinks when traversing directories.
    #[must_use]
    pub fn follow_symlinks(mut self, enabled: bool) -> Self {
        self.cfg.follow_symlinks = enabled;
        self
    }

    /// Finalises the builder, returning the [`IngestConfig`].
    pub fn build(self) -> IngestConfig {
        self.cfg
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_deduplicates_allowed_lengths() {
        let cfg = TrainerConfig::builder()
            .allowed_token_lengths([4, 2, 2, 1])
            .show_progress(false)
            .build()
            .expect("config should be valid");
        assert_eq!(&cfg.allowed_token_lengths, &[1, 2, 4]);
    }

    #[test]
    fn validate_rejects_missing_base_length() {
        let cfg = TrainerConfig {
            allowed_token_lengths: vec![2, 3],
            ..TrainerConfig::default()
        };
        let err = cfg.validate().expect_err("validation should fail");
        assert!(matches!(
            err,
            BbpeError::InvalidConfig(message) if message.contains("allowed_token_lengths must include")
        ));
    }

    #[test]
    fn ingest_builder_overrides_defaults() {
        let cfg = IngestConfig::builder()
            .chunk_size(1024)
            .recursive(false)
            .follow_symlinks(true)
            .build();
        assert_eq!(cfg.chunk_size, 1024);
        assert!(!cfg.recursive);
        assert!(cfg.follow_symlinks);
    }
}
