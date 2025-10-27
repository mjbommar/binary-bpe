#!/usr/bin/env python3
"""
Compare tokenizations produced by multiple tokenizer.json files.

Usage:
    python scripts/compare_tokenizations.py --text TEXT_PATH --model name:path [...]

Example:
    python scripts/compare_tokenizations.py \
        --text ~/sample-001.txt \
        --model full:/tmp/sample001-full.json \
        --model support-4m:/tmp/sample001-support-4m.json
"""

import argparse
import json
import textwrap
from pathlib import Path

from tokenizers import Tokenizer


def load_sample(path: Path, max_bytes: int) -> str:
    data = path.read_bytes()
    sample = data[:max_bytes]
    try:
        return sample.decode("utf-8")
    except UnicodeDecodeError:
        return sample.decode("latin-1")


def format_tokens(tokens, max_per_line=8):
    lines = []
    line = []
    for tok in tokens:
        line.append(repr(tok))
        if len(line) >= max_per_line:
            lines.append(", ".join(line))
            line = []
    if line:
        lines.append(", ".join(line))
    return "\n    ".join(lines)


def summarize_mode(name: str, tokenizer_path: Path, sample_text: str):
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    encoding = tokenizer.encode(sample_text)
    token_count = len(encoding.ids)
    tokens = encoding.tokens
    summary = {
        "mode": name,
        "token_count": token_count,
        "tokens": tokens,
    }
    return summary


def print_summary(summary, wrap_width=80):
    mode = summary["mode"]
    token_count = summary["token_count"]
    tokens = summary["tokens"]
    print(f"=== {mode} ===")
    print(f"tokens: {token_count}")
    print("sequence:")
    formatted = format_tokens(tokens)
    for line in formatted.splitlines():
        for wrapped in textwrap.wrap(line, width=wrap_width):
            print(f"    {wrapped}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Compare tokenizer outputs for the same text.")
    parser.add_argument("--text", required=True, type=Path, help="Path to text file to tokenize.")
    parser.add_argument(
        "--bytes",
        type=int,
        default=256,
        help="Number of bytes from the text to inspect (default: 256).",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        required=True,
        help="Model spec as name:path_to_tokenizer.json (repeatable).",
    )
    args = parser.parse_args()

    sample_text = load_sample(args.text, args.bytes)
    print("=== Sample excerpt ===")
    print(sample_text)
    print()

    summaries = []
    for spec in args.models:
        if ":" not in spec:
            raise ValueError(f"Invalid model spec '{spec}'. Expected name:path.")
        name, path_str = spec.split(":", 1)
        tokenizer_path = Path(path_str)
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer path not found: {tokenizer_path}")
        summary = summarize_mode(name, tokenizer_path, sample_text)
        summaries.append(summary)

    for summary in summaries:
        print_summary(summary)


if __name__ == "__main__":
    main()
