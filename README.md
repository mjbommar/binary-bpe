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
bbpe = { version = "0.1", default-features = false }
```

## Quick Start

Train a tokenizer on binary files and use it:

```bash
# Train tokenizer (270MB ISO takes ~3 minutes, achieves 1.46 MiB/s)
bbpe train /tmp/alpine-standard-3.22.2-x86_64.iso --vocab-size 8192 -o tokenizer.json

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

- Training: >1 MiB/s on release builds
- Parallel processing with Rayon
- Incremental pair counting for efficient merging
- Tested on 270MB files with sub-4-minute training time

## Development

```bash
cargo fmt
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

## License

Apache-2.0