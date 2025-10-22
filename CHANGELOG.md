# Changelog

## 0.2.0 - 2025-10-22
- Added crate-level documentation with stricter lint gates (`#![forbid(unsafe_code)]`, `missing_docs`).
- Documented configuration/corpus modules and added unit tests across bytes/config/corpus/model.
- Hardened trainer implementation (safe token ID conversions, accurate pair-count updates).
- Refreshed README with CLI reference, installation notes, MSRV policy, and Python compatibility example.
- Added Apache 2.0 license file and publishing metadata (`homepage`, `documentation`, keywords/categories).
- Introduced `.gitignore` and Cargo `exclude` rules to keep artifacts out of git/crate packages.
- Upgraded dependencies (`rustc-hash 2.1`, `thiserror 2.0`, `indicatif 0.18`, `tokenizers 0.22.1`, `criterion 0.7`, etc.).
- Enhanced Criterion benchmark to report throughput with `Throughput::Bytes` and `SamplingMode::Flat`.
