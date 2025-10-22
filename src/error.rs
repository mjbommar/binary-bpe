//! Error handling utilities shared across the crate.

use std::path::PathBuf;

use thiserror::Error;

/// Convenient result type used throughout the crate.
pub type Result<T, E = BbpeError> = std::result::Result<T, E>;

/// Domain-specific error describing failures during configuration, IO, or tokenizer operations.
#[derive(Debug, Error)]
pub enum BbpeError {
    /// Training configuration failed validation.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
    /// Filesystem IO error with optional context path.
    #[error("io error while processing {path:?}: {source}")]
    Io {
        /// Underlying IO error returned by the standard library.
        source: std::io::Error,
        /// Target path associated with the IO failure if available.
        path: Option<PathBuf>,
    },
    /// Error bubbled up from the `tokenizers` crate.
    #[error("huggingface tokenizers error: {0}")]
    Tokenizers(String),
    /// Serialization or deserialization failure.
    #[error("serialization error: {0}")]
    Serialization(String),
    /// Catch-all variant for invariants that should not occur.
    #[error("internal error: {0}")]
    Internal(String),
}

impl From<tokenizers::Error> for BbpeError {
    fn from(err: tokenizers::Error) -> Self {
        Self::Tokenizers(err.to_string())
    }
}

impl From<serde_json::Error> for BbpeError {
    fn from(err: serde_json::Error) -> Self {
        Self::Serialization(err.to_string())
    }
}

impl BbpeError {
    /// Helper constructor that attaches an optional path when wrapping IO errors.
    pub fn io(source: std::io::Error, path: Option<PathBuf>) -> Self {
        Self::Io { source, path }
    }
}
