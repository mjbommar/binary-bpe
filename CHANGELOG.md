# Changelog

## Unreleased
- _Nothing yet._

## 0.5.0 - 2025-11-14
- Added configurable ASCII, Unicode, and null-delimited preprocessing to both the library (`TrainerConfig::preprocessor`) and CLI (`--preprocessor`), wiring the behaviour into exported Hugging Face tokenizers.
- Introduced probabilistic preprocessing with optional RNG seeding (`--preprocessor-probability`, `--preprocessor-seed`, and the new builder helpers), allowing occasional cross-delimiter merges while keeping inference aligned with the raw byte stream whenever randomness is enabled.
- `bbpe train` (and `Trainer::train_from_jsonl`) can now ingest newline-delimited JSON by specifying `--jsonl path.jsonl:field.path`, including transparent support for `.jsonl.gz` inputs, enabling text-heavy corpora without pre-extracting files.
- Added hierarchical tokenizer families via repeated `--family-size` flags (and the new `BpeModel::derive_with_vocab` helper) plus optional `--family-template` output routing.
- Documented the new workflows (preprocessors, JSONL ingestion, hierarchical families) and expanded the CLI/integration test suites, including round-trip coverage via the Hugging Face `tokenizers` crate.

## 0.4.1 - 2025-11-12
- Added `--min-entropy` and `--max-entropy` flags to `bbpe train` command for filtering chunks during regular training.
- Enhanced test coverage with 5 new integration tests for entropy filtering in train command.
- Fixed clippy warning in corpus.rs test (cloned_ref_to_slice_refs).

## 0.4.0 - 2025-11-12
- Added parallel chunk training using rayon for significant performance improvements on multi-core systems.
- Introduced entropy-based filtering with `--min-entropy` and `--max-entropy` flags to skip low-information and compressed chunks.
- Implemented memory-efficient corpus streaming via `stream_binary_corpus()` to avoid loading entire datasets into memory.
- Added `--no-report` flag to reduce memory usage by skipping JSON report generation.
- Optimized memory usage with Arc-based zero-copy sharing of models and merge records across duplicate chunks.
- Extended default token length limit from 16 to 32 bytes for learning longer sequences.
- Improved HuggingFace tokenizer compatibility by explicitly setting pre_tokenizer to None and decoder to Fuse type.

## 0.3.2 - 2025-10-29
- Added content-based chunk caching to `bbpe chunk-train`, eliminating redundant retraining for duplicated binaries.
- Exposed a `--duplicates` flag to control whether duplicate chunks are counted or collapsed during combination, with detailed metrics in the chunk report and CLI output.
- Extended chunk-train reports and README documentation to capture the new deduplication workflow.

## 0.3.1 - 2025-10-27
- Added an `indicatif`-powered progress bar to `bbpe chunk-train` plus a `--no-progress` escape hatch for quiet runs.
- Switched the default chunk combiner to the support-based strategy for stronger out-of-the-box results and documented the change in the README.

## 0.2.0 - 2025-10-22
- Added crate-level documentation with stricter lint gates (`#![forbid(unsafe_code)]`, `missing_docs`).
- Documented configuration/corpus modules and added unit tests across bytes/config/corpus/model.
- Hardened trainer implementation (safe token ID conversions, accurate pair-count updates).
- Refreshed README with CLI reference, installation notes, MSRV policy, and Python compatibility example.
- Added Apache 2.0 license file and publishing metadata (`homepage`, `documentation`, keywords/categories).
- Introduced `.gitignore` and Cargo `exclude` rules to keep artifacts out of git/crate packages.
- Upgraded dependencies (`rustc-hash 2.1`, `thiserror 2.0`, `indicatif 0.18`, `tokenizers 0.22.1`, `criterion 0.7`, etc.).
- Enhanced Criterion benchmark to report throughput with `Throughput::Bytes` and `SamplingMode::Flat`.
