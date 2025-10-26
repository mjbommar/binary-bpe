# bbpe

Binary byte pair encoding (BPE) tokenizer for Rust. Generates production-ready `tokenizer.json` files compatible with [Hugging Face tokenizers](https://github.com/huggingface/tokenizers/).

## Installation

```bash
# Install CLI and library
cargo install bbpe

# Library only
cargo add bbpe
```

For library-only usage without CLI:
```toml
bbpe = { version = "0.2", default-features = false }
```

## Quick Start

Train a tokenizer on binary files and use it:

```bash
# Train tokenizer (270MB ISO takes ~3 minutes, achieves 1.46 MiB/s)
bbpe train /tmp/alpine-standard-3.22.2-x86_64.iso --vocab-size 8192 -o tokenizer.json

# Low-memory alternative (32 MiB streaming chunks, writes report + tokenizer)
bbpe chunk-train /tmp/alpine-standard-3.22.2-x86_64.iso \
  --output chunk_report.json \
  --final-tokenizer chunked-tokenizer.json

# Inspect aggregate report
jq '.global_merges[0:5]' chunk_report.json

# Encode binary to tokens
bbpe encode -m tokenizer.json binary.file --json
{"path":"binary.file","tokens":[6299,144,144,144,6299,6335,213,4238]}

# Decode tokens back to bytes
bbpe decode -m tokenizer.json --output decoded.bin 6299 144 144 144

# Inspect tokenizer
bbpe info -m tokenizer.json
Model type   : BPE
Vocab size   : 8185
Merges       : 7929
Byte fallback: true
Special tokens: <|start|>, <|end|>, <|pad|>, <|unk|>, <|cls|>, <|sep|>, <|mask|>
```

## Library Usage

```rust
use bbpe::{Trainer, TrainerConfig, IngestConfig};

// Configure and train
let cfg = TrainerConfig::builder()
    .target_vocab_size(4096)
    .min_frequency(2)
    .build()?;

let trainer = Trainer::new(cfg);
let artifacts = trainer.train_from_paths(
    &["/path/to/binaries"],
    &IngestConfig::default()
)?;

// Save tokenizer
artifacts.model.save_huggingface("tokenizer.json")?;

// Use for encoding/decoding
let tokenizer = artifacts.model.binary_tokenizer()?;
let tokens = tokenizer.encode_bytes(b"\x7fELF\x02\x01", false)?;
let decoded = tokenizer.decode_to_bytes(&tokens, true)?;
assert_eq!(decoded, b"\x7fELF\x02\x01");
```

## CLI Commands

### train
Build a tokenizer from binary inputs.

```bash
bbpe train <INPUT_PATH> [OPTIONS]

Options:
  --vocab-size <SIZE>       Target vocabulary size [default: 32768]
  --min-frequency <FREQ>    Minimum pair frequency [default: 4]
  --chunk-size <BYTES>      File chunk size [default: 8192]
  -o, --output <PATH>       Output tokenizer path [default: tokenizer.json]
  --no-progress             Disable progress bar
  --threads <NUM>           Thread pool size
```

Example with 1GB corpus:
```bash
bbpe train ./corpus --vocab-size 16384 --min-frequency 128 -o large.json
```

### chunk-train
Stream large corpora in fixed-size chunks, aggregate merge statistics, and optionally synthesise a tokenizer from the aggregated merges.

```bash
bbpe chunk-train <INPUT_PATH>... [OPTIONS]

Key options:
  --chunk-size <BYTES>         Chunk size in bytes (default: 33554432 / 32 MiB)
  --chunk-merges <COUNT>       Per-chunk merge budget (default: 512)
  --global-merges <COUNT>      Maximum merges to keep after aggregation (default: 4096)
  --min-frequency <COUNT>      Minimum pair frequency per chunk (default: 4)
  --rank-mode <MODE>           Aggregation ranking: weight | support | balanced (default: weight)
  --support-weight <FACTOR>    Extra support multiplier when rank-mode=balanced (default: 1)
  --min-chunk-support <COUNT>  Drop merges seen in fewer than COUNT chunks
  --ensemble-mode <MODE>       Combine chunks via baseline | boost | sampled (default: baseline)
  --boost-rounds <COUNT>       Boosting passes when ensemble-mode=boost (default: 1)
  --boost-sample-size <COUNT>  Chunks per boosting round (defaults to all chunks)
  --boost-learning-rate <RATE> Learning rate for boost weight updates (default: 0.5)
  --sample-rounds <COUNT>      Monte Carlo rounds when ensemble-mode=sampled (default: 8)
  --sample-chunks <COUNT>      Chunks per sampling round (defaults to √N)
  --ensemble-seed <SEED>       Deterministic seed for boost/sample sampling
  --final-tokenizer <PATH>     Emit a synthesised tokenizer from the aggregated merges
  --final-merges <COUNT>       Cap merges realised in the synthesised tokenizer
  --validation-fraction <RATIO> Hold out this fraction (0.0–0.5) of every chunk for validation (default: 0.08)
  --validation-seed <SEED>     RNG seed used when sampling validation slices
  --verbose-chunks             Log per-chunk stats as they are processed
```

The command writes a human-readable progress summary to stdout, a JSON report (default `chunked_merges.json`) describing every chunk plus the aggregated ranking, and, when `--final-tokenizer` is supplied, a Hugging Face-compatible `tokenizer.json`.

When `--final-tokenizer` is requested, `chunk-train` now reserves `--validation-fraction` of every chunk (default 8%) as a held-out byte pool (pass `--validation-fraction 0` to disable). The final tokenizer is built by greedily replaying aggregated merge candidates, selecting only those that actually reduce token counts on the validation bytes. This outcome-driven pass keeps the RAM footprint bounded by the chunk size while ensuring the synthesised tokenizer matches the canonical trainer whenever the chunk size exceeds the corpus.

`--ensemble-mode` unlocks two higher-variance reducers designed for large, heterogeneous corpora:

- `boost` runs several weighted rounds, re-sampling chunks that were under-represented in earlier passes. The report’s `boost_details` array captures per-round coverage and the weight ranges that emerged.
- `sampled` averages merge statistics across repeated random subsets, similar in spirit to bagging/extra-trees. The `sample_details` section records coverage statistics for every Monte Carlo draw.

Both variants keep the memory footprint within the chunk budget while trading determinism for better coverage of rare patterns.

Example (32 MiB chunks, default ranking):

```bash
bbpe chunk-train /tmp/alpine-standard-3.22.2-x86_64.iso \
  --output /tmp/chunk_report.json \
  --final-tokenizer /tmp/chunked-tokenizer.json

# Quick look at the aggregated merges
jq '.global_merges[0:3]' /tmp/chunk_report.json
```

The JSON report captures the full aggregation context, including `rank_mode`, `min_chunk_support`, the number of candidates considered, and per-chunk previews so you can audit what survives the reducer.

To favour merges that appear across many chunks, switch to balanced ranking and increase the support weight:

```bash
bbpe chunk-train ./corpus \
  --rank-mode balanced \
  --support-weight 4 \
  --min-chunk-support 3 \
  --final-tokenizer chunked-balanced.json
```

### encode
Convert binary files to token sequences.

```bash
bbpe encode -m <TOKENIZER> <FILES>... [OPTIONS]

Options:
  --json                    Output JSON format
  --output <PATH>           Save tokens to file
  --skip-special-tokens     Skip special tokens
```

Examples:
```bash
# JSON output
bbpe encode -m tokenizer.json binary.exe --json

# Save to file
bbpe encode -m tokenizer.json data.bin --output data.tokens
```

### decode
Reconstruct bytes from token IDs.

```bash
bbpe decode -m <TOKENIZER> [IDS]... [OPTIONS]

Options:
  --input <PATH>            Read tokens from file
  --output <PATH>           Save decoded bytes (default: stdout)
  --skip-special-tokens     Skip special tokens
```

Examples:
```bash
# Decode from arguments
bbpe decode -m tokenizer.json 6299 144 144 --output restored.bin

# Decode from file
bbpe decode -m tokenizer.json --input tokens.txt --output restored.bin
```

### info
Display tokenizer metadata.

```bash
bbpe info -m <TOKENIZER> [--json]
```

## Configuration

### TrainerConfig

```rust
TrainerConfig::builder()
    .target_vocab_size(8192)        // Target vocabulary size
    .min_frequency(2)                // Minimum pair frequency for merging
    .allowed_token_lengths(1..=16)   // Token length range
    .special_tokens(vec![...])      // Special tokens to add
    .show_progress(true)             // Display progress bar
    .build()
```

### IngestConfig

```rust
IngestConfig::builder()
    .chunk_size(8192)       // File reading chunk size
    .recursive(true)        // Traverse directories recursively
    .follow_symlinks(false) // Follow symbolic links
    .build()
```

## Python Interoperability

Generated tokenizers work directly with Hugging Face:

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer.json")
encoded = tokenizer.encode(binary_data)
print(encoded.ids)  # [6299, 144, 144, ...]
```

## Performance

| Run                                      | Chunk Size | Wall Time | Peak RSS | Canon merges matched | Top-100 match | Bottom-100 match | Avg token len | Tokens ≥8 B | Realised merges | Dups/Unresolved | Notes |
|------------------------------------------|------------|-----------|----------|----------------------|---------------|------------------|---------------|-------------|------------------|-----------------|-------|
| `bbpe train` (full corpus)               | —          | 2m 49s    | 1.76 GB  | 4 089 (100 %)        | 100 %         | 100 %            | 2.03          | 0.7 %      | 4 089            | —               | Canonical reference tokenizer |
| `chunk-train` boost                      | 1 MiB      | 1m 21s    | 0.29 GB  | 1 078 (26.4 %)       | 58 %          | 0 %              | 3.26          | 9.3 %      | 4 033            | 60 / 3          | 270 chunks; weights up to ×6 after 5 rounds |
| `chunk-train` sampled                    | 1 MiB      | 1m 22s    | 0.29 GB  | 1 067 (26.1 %)       | 63 %          | 0 %              | 3.87          | 14.0 %     | 3 979            | 115 / 2         | 16 rounds ×64-chunk draws, heavier long-tail tokens |
| `chunk-train` boost                      | 2 MiB      | 1m 20s    | 0.30 GB  | 1 189 (29.1 %)       | 69 %          | 0 %              | 2.67          | 5.0 %      | 4 058            | 34 / 4          | 135 chunks; coverage still ~18 % per round |
| `chunk-train` sampled                    | 2 MiB      | 1m 20s    | 0.30 GB  | 1 243 (30.4 %)       | 65 %          | 0 %              | 3.08          | 8.0 %      | 4 035            | 59 / 2          | 16 rounds ×48 chunks improves overlap slightly |
| `chunk-train` boost                      | 4 MiB      | 1m 21s    | 0.31 GB  | 1 209 (29.6 %)       | 71 %          | 0 %              | 2.66          | 5.2 %      | 4 057            | 37 / 2          | 68 chunks; fewer duplicates |
| `chunk-train` sampled                    | 4 MiB      | 1m 21s    | 0.31 GB  | 1 263 (30.9 %)       | 73 %          | 0 %              | 2.66          | 5.1 %      | 4 060            | 33 / 3          | Highest overlap among tested chunk sizes |
| `chunk-train` boost                      | 16 MiB     | 1m 27s    | 0.37 GB  | 1 196 (29.2 %)       | 81 %          | 0 %              | 2.31          | 2.6 %      | 4 083            | 13 / 0          | 17 chunks; coverage maxes at 100 % for sampled chunks |
| `chunk-train` sampled                    | 16 MiB     | 1m 26s    | 0.37 GB  | 1 210 (29.6 %)       | 81 %          | 1 %              | 2.30          | 2.5 %      | 4 085            | 11 / 0          | Tail still diverges; overlap stabilises ~30 % |

Observations:

- Even with more sophisticated reducers, only ~26–31 % of canonical merges survive; variance lives almost entirely in the long tail (bottom‑100 overlap ≈ 0 %).
- Top‑100 agreement improves with larger chunks (58 % ➝ 81 %), suggesting the ensembles capture high-frequency structure but continue to miss canonical late-stage merges.
- Boosting concentrates probability mass on under-served chunks (round coverage means ≈ 0.17) but adds modest runtime overhead and still leaves ≈ 10 % of candidates as duplicates or unresolved dependencies.
- Sampling hits slightly higher global overlap at 2–4 MiB, but amplifies chunk-local long literals (avg token length and ≥8 B share rise, especially at 1 MiB) and produces more duplicate merges.
- Peak RSS scales with chunk size rather than ensemble mode; the chunk budget remains the dominant lever, not the reducer choice.

## Development

```bash
cargo fmt
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

## License

Apache-2.0
