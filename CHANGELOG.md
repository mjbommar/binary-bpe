# Changelog

## Unreleased
- _Nothing yet._

## 0.6.4 - 2025-11-24
- Enhanced `bbpe info` so it inspects the reasoning/argumentation block inside the vocabulary, decodes the canonical glyphs, and reports whether the Category 4 range is intact (plus the full glyph list) without forcing you to reverse the ByteLevel serialization by hand.
- Added a CLI regression test that trains both reasoning-enabled and reasoning-disabled models and asserts the new inspector flags the correct state in `--json` mode.

## 0.6.3 - 2025-11-23
- Hardened the streaming trainer so it tracks every byte sequence already present in the vocabulary, skipping merges that would recreate leading specials, byte tokens, or reasoning glyphs; this eliminates the OrderedVocab “holes” warning when corpora contain those symbols and keeps Hugging Face exports contiguous.
- Updated the whitespace padding helper and added a regression test that repeatedly feeds the “●” reasoning glyph through `Trainer::train_from_sequences`, ensuring no duplicate token bytes can ever enter the vocabulary again.

## 0.6.2 - 2025-11-21
- Ensured every Hugging Face export wraps the trained model with a ByteLevel pre-tokenizer/decoder, preserving non-ASCII bytes (CJK, emoji, reasoning glyphs, etc.) even when downstream tokenizers are run outside of `bbpe`.
- Reworked serialization to use the GPT-2 byte alphabet end-to-end, teaching `BinaryTokenizer` to detect existing ByteLevel decoders and strip them when round-tripping so legacy Latin-1 artefacts keep working.
- Moved the Category 4 reasoning/argumentation tokens into the fixed base vocabulary (immediately after the 256-byte alphabet) instead of marking them as specials, preventing Hugging Face from stripping them during decode while keeping the CLI toggles intact.
- Updated probabilistic preprocessor exports to keep the mandatory ByteLevel stage but drop only the optional whitespace/null splitter, aligning inference byte streams with what the trainer actually saw and refreshing the CLI tests/docs to describe the behavior.

## 0.6.1 - 2025-11-21
- Added an incremental preprocessing pipeline (`PreprocessorRunner`) plus the new `SequenceStream`/`Trainer::train_from_stream` APIs so callers can normalize, tokenize, and aggregate corpora chunk-by-chunk without staging every sequence in memory.
- Introduced `stream_jsonl_corpus` alongside the existing binary chunk iterator, enabling both binary and JSONL sources to feed the streaming trainer directly (with optional length hints for progress reporting).
- Implemented a lightweight pair-cache cap that prunes the low-frequency frontier after each merge, keeping `pair_counts`, `pair_positions`, and the candidate heap bounded even on massive datasets.
- Refreshed the README with streaming examples, updated installation instructions (`cargo add bbpe@0.6.1`), and documented the `uv run --with tokenizers ...` sanity check; ran the command during release prep to confirm Hugging Face round-trips still pass.

## 0.6.0 - 2025-11-19
- Locked the Category 3 specials (`<|start|>` … `<|mask|>`) to token IDs 0–6 across every model via the new `references/special_tokens.json`, guaranteeing deterministic layouts for both the CLI and the library.
- Inserted the full set of Category 4 reasoning tokens immediately after the 256-byte alphabet, enabled by default but controllable through `--disable-reasoning-tokens` / `TrainerBuilder::reasoning_tokens_enabled(false)`; extra custom specials are now appended without disturbing the reserved slots.
- Reworked Hugging Face serialization and `BinaryTokenizer` metadata so exported `tokenizer.json` files preserve the fixed IDs, `bbpe info` reports base/special/total counts (embedding width), and round-trip encode/decode matches exactly when checked via `uv run --with tokenizers python -c ...`.
- Refined probabilistic whitespace preprocessing to keep whitespace-only spans clean while still enabling multi-word merges when `--preprocessor-probability < 1`, and expanded documentation/tests covering the reserved-token workflow.

## 0.5.2 - 2025-11-18
- Documented special token numbering behavior: special tokens are always assigned the lowest contiguous ID range `[0, N)` where N is the number of special tokens, ensuring consistency across both `train` and `chunk-train` algorithms and compatibility with Hugging Face tokenizers.

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
