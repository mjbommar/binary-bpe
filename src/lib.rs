//! Binary byte pair encoding (BPE) training library and CLI.
//!
//! The crate exposes both a library API and a `bbpe` command line interface for
//! training binary-aware BPE tokenizers that interoperate with the Hugging Face
//! `tokenizers` ecosystem.  Typical usage loads a binary corpus, trains a
//! `BpeModel`, and then persists the resulting `tokenizer.json`.
//!
//! ```no_run
//! use bbpe::{IngestConfig, Trainer, TrainerConfig};
//!
//! # fn main() -> bbpe::Result<()> {
//! let trainer_cfg = TrainerConfig::builder()
//!     .target_vocab_size(4096)
//!     .min_frequency(2)
//!     .show_progress(false)
//!     .build()?;
//! let trainer = Trainer::new(trainer_cfg);
//! let ingest_cfg = IngestConfig::default();
//! let artifacts = trainer.train_from_paths(&["/path/to/binaries"], &ingest_cfg)?;
//! artifacts.model.save_huggingface("tokenizer.json")?;
//! # Ok(())
//! # }
//! ```
//!
//! The CLI is enabled by default through the `cli` feature.  Users targeting the
//! library portion only can disable default features to avoid the CLI
//! dependencies: `bbpe = { version = "...", default-features = false }`.

#![forbid(unsafe_code)]
#![warn(
    missing_docs,
    clippy::all,
    rust_2018_idioms,
    future_incompatible,
    unused_lifetimes,
    unreachable_pub
)]
#![allow(
    clippy::module_name_repetitions,
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    clippy::doc_markdown,
    clippy::multiple_crate_versions
)]

pub mod bytes;
pub mod config;
pub mod corpus;
pub mod error;
pub mod metrics;
pub mod model;
pub mod serialization;
pub mod trainer;

pub use config::{IngestConfig, TrainerBuilder, TrainerConfig};
pub use error::{BbpeError, Result};
pub use metrics::{IterationMetrics, TrainingMetrics};
pub use model::{BinaryTokenizer, BpeModel, TokenId};
pub use trainer::{Trainer, TrainerArtifacts};
