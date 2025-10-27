# Changelog

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
