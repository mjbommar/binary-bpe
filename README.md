# bbpe

Binary-aware byte pair encoding (BPE) toolkit for Rust. `bbpe` trains Hugging Face–compatible `tokenizer.json` assets directly from raw binaries or newline-delimited JSON (JSONL) text, making it easy to build tokenizers for firmware, malware, structured data dumps, or hybrid corpora.

- **Binary-first**: Streams arbitrary bytes, honors null-delimited boundaries, and always exports true byte fallback Hugging Face models.
- **Configurable preprocessing**: Deterministic ASCII/Unicode/null splitters or probabilistic boundary sampling to encourage multi-word merges.
- **JSONL ingestion**: Point at `file.jsonl[:field.path]` (gzip optional) to train on textual corpora without pre-extracting files.
- **Hierarchical vocabularies**: Train once at a large vocab, derive smaller siblings whose token IDs remain aligned.
- **Chunked training**: Experimental `chunk-train` command allows ensemble-style training on arbitrarily large corpora.

## Installation

```bash
# CLI + library
cargo install bbpe

# Add as a dependency (library only)
cargo add bbpe@0.5.2
```

For library-only usage without the CLI feature:

```toml
bbpe = { version = "0.5.2", default-features = false }
```

## Quick Start

### Train on binaries

```bash
bbpe train firmware.bin --vocab-size 4096 --min-frequency 4 \
  --preprocessor null-delimited -o tokenizer.json
```

### Train on JSONL text (gzip supported)

```bash
bbpe train \
  --jsonl corpus.jsonl:text \
  --jsonl docs.jsonl.gz:data.body \
  --vocab-size 8192 --min-frequency 2 \
  --preprocessor ascii-whitespace \
  --preprocessor-probability 0.75 \
  --preprocessor-seed 42 \
  --family-size 4096 --family-size 2048 \
  --family-template out/tokenizer-{size}.json \
  -o tokenizer-8192.json
```

### Encode / Decode

```bash
bbpe encode -m tokenizer.json binary.bin --json > tokens.json
bbpe decode -m tokenizer.json --input tokens.txt --output restored.bin
```

## CLI Reference

Every command accepts `-q/--quiet` and `-v/--verbose` for logging control. Paths can be repeated.

### `train`

Build a tokenizer from binary files, directories, and/or JSONL specs.

```
bbpe train [FILES]... [--jsonl PATH:FIELD]... [OPTIONS]
```

Key options:

| Flag | Description |
| --- | --- |
| `--jsonl PATH:FIELD` | Train from newline-delimited JSON. `FIELD` uses `dot.separated.keys`. Files ending in `.gz` are transparently decompressed. Repeatable. |
| `-o, --output PATH` | Output `tokenizer.json` (default `tokenizer.json`). |
| `--vocab-size SIZE` | Target vocabulary (default 32768). Must be ≥ 256 + special tokens. |
| `--min-frequency COUNT` | Minimum pair frequency for merges (default 4). |
| `--chunk-size BYTES` | Read binary inputs in chunks (default 8192, `0` = whole file). |
| `--special-token TOKEN` | Append custom special tokens (repeatable). |
| `--preprocessor MODE` | `none`, `ascii-whitespace`, `unicode-whitespace`, or `null-delimited`. |
| `--preprocessor-probability P` | Probability `[0,1]` that each detected boundary is kept (default `1.0`). Set `< 1` to occasionally merge across whitespace/null runs. |
| `--preprocessor-seed SEED` | Seed RNG for probabilistic preprocessing. |
| `--family-size SIZE` | Emit derived vocabularies after training (repeat flag). |
| `--family-template PATTERN` | Output template for derived models (supports `{size}`). |
| `--max-merge-iterations COUNT` | Hard merge ceiling (defaults to vocab budget). |
| `--allowed-length LEN` | Restrict merge lengths (repeat). |
| `--threads N` | Override Rayon worker count. |
| `--no-progress` | Disable spinner output (logging still shows start/end). |
| `--min-entropy BITS`, `--max-entropy BITS` | Filter sequences outside entropy window before training. |

### `chunk-train` (experimental)

Train chunk-wise vocabularies and combine them. Useful when full-corpus training would exceed memory.

```
bbpe chunk-train [FILES]... [OPTIONS]
```

Notable flags:

- `--chunk-size BYTES` (required, default 32 MiB)
- `--combine-mode first|frequency|support|entropy`
- `--duplicates count|unique`
- `--output PATH`, `--report PATH`, `--no-report`
- Same preprocessing / probability knobs as `train`. (Currently, JSONL ingestion is only available on the main `train` command.)

### `encode`

```
bbpe encode -m tokenizer.json <FILES>... [--json] [--output PATH] [--skip-special-tokens]
```

### `decode`

```
bbpe decode -m tokenizer.json [IDS]... [--input PATH] [--output PATH] [--skip-special-tokens]
```

### `info`

```
bbpe info -m tokenizer.json [--json]
```

## Preprocessing Modes

`bbpe` optionally splits inputs before merge counting:

- **ascii-whitespace** – Splits on ASCII whitespace (`space`, `\t`, `\r`, `\n`, `VT`, `FF`).
- **unicode-whitespace** – Uses `char::is_whitespace`, preserving Unicode separators (implemented with `bstr`).
- **null-delimited** – Splits on contiguous `0x00` runs (great for binaries with C strings or padded records).
- **none** – Raw byte stream.

Set `--preprocessor-probability <P>` (or `TrainerConfig::builder().preprocessor_split_probability(P)`) to randomly *keep* only a subset of boundaries. Example: `P=0.8` keeps 80 % of whitespace splits while letting 20 % of tokens cross word boundaries, encouraging the model to learn multi-word merges. Provide `--preprocessor-seed` for reproducibility. Hugging Face exports automatically drop the pre-tokenizer section when `P < 1.0`, so downstream inference sees the exact byte stream used during training.

## JSONL Ingestion

Use `--jsonl path.jsonl:field.path` to read newline-delimited JSON alongside binary files. Examples:

- `--jsonl data.jsonl:text`
- `--jsonl downloads/articles.jsonl.gz:payload.body`

Field paths use dot notation; numeric array indices are allowed (`choices.0.text`). Empty lines are skipped, non-string fields error out. The reader accepts gzip-compressed inputs (`.jsonl.gz`).

Library equivalent:

```rust
use bbpe::{Trainer, TrainerConfig, TrainerArtifacts};
use bbpe::corpus::JsonlSpec;

let cfg = TrainerConfig::builder()
    .target_vocab_size(8192)
    .min_frequency(2)
    .preprocessor_split_probability(0.7)
    .preprocessor_seed(Some(1337))
    .build()?;
let trainer = Trainer::new(cfg);
let specs = ["corpus.jsonl:text".parse::<JsonlSpec>()?];
let TrainerArtifacts { model, .. } = trainer.train_from_jsonl(&specs)?;
model.save_huggingface("tokenizer.json")?;
```

## Hierarchical Tokenizer Families

After training the largest vocab, derive smaller siblings whose token IDs stay aligned:

```bash
bbpe train corpus.bin --vocab-size 32768 \
  --family-size 8192 --family-size 4096 \
  --family-template artifacts/model-{size}.json \
  -o artifacts/model-32768.json
```

`BpeModel::derive_with_vocab(size)` mirrors the CLI for library users. Special tokens remain clustered at the end so IDs stay consistent across the family.

## Special Token Numbering

Special tokens (added via `--special-token` or `TrainerConfig::special_tokens`) are always assigned the **lowest contiguous ID range `[0, N)`** where N is the number of special tokens. This normalization happens during serialization to Hugging Face format and applies to both `train` and `chunk-train` algorithms.

For example, with 2 special tokens `<|start|>` and `<|end|>`:
- Special token IDs: `[0, 1]`
- Base byte vocabulary: `[2, 257]` (256 bytes)
- Learned merges: `[258, ...]`

This ensures:
- Consistent token IDs across derived vocabulary families
- Compatibility with Hugging Face tokenizers library expectations
- Deterministic special token numbering regardless of training algorithm

The base 256-byte vocabulary and learned merges never participate in this normalization; only special tokens are renumbered.

## Chunk Training (experimental)

`bbpe chunk-train` slices the corpus into fixed-size windows, trains independent vocabularies, and merges them via user-selectable strategies (`support`, `frequency`, etc.). Duplicate chunk caching, entropy filtering, and reporting are built in so you can prototype large corpora without huge memory spikes.

## Library Primer

```rust
use bbpe::{Trainer, TrainerConfig, IngestConfig, PreprocessorConfig, PreprocessorKind};

let trainer_cfg = TrainerConfig::builder()
    .target_vocab_size(4096)
    .min_frequency(2)
    .preprocessor(PreprocessorConfig {
        kind: PreprocessorKind::UnicodeWhitespace,
        split_probability: 0.9,
        seed: Some(42),
    })
    .build()?;
let trainer = Trainer::new(trainer_cfg);
let ingest_cfg = IngestConfig::builder().chunk_size(0).build();
let artifacts = trainer.train_from_paths(&["firmware.bin"], &ingest_cfg)?;
artifacts.model.save_huggingface("tokenizer.json")?;

// Hierarchy
let small = artifacts.model.derive_with_vocab(2048)?;
small.save_huggingface("tokenizer-2048.json")?;
```

`Trainer::train_from_jsonl(&[JsonlSpec])` mirrors the CLI `--jsonl` behavior.

## Python Interoperability

All tokenizers are Hugging Face JSON artifacts:

```python
from tokenizers import Tokenizer

model = Tokenizer.from_file("tokenizer.json")
encoding = model.encode("\x7fELF", add_special_tokens=False)
print(encoding.ids)
```

## Development & Testing

```bash
cargo fmt
cargo clippy --all-targets --all-features -- -D warnings
cargo test
# Optional sanity check before publishing
cargo package
```

## License

Apache-2.0
