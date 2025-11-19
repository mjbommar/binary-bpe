#!/usr/bin/env python3
"""Compute lexical/orthographic metrics for bbpe tokenizer artifacts."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import List, Dict, Any

ASCII_WHITESPACE = {9, 10, 11, 12, 13, 32}
PRINTABLE_WHITESPACE = {9, 10, 13, 32}
DIGITS = set(range(ord("0"), ord("9") + 1))
UPPER = set(range(ord("A"), ord("Z") + 1))
LOWER = set(range(ord("a"), ord("z") + 1))
ALPHA = UPPER | LOWER
PUNCT = {ord(ch) for ch in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"}


def percentile(values: List[int], q: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return float(values[0])
    data = sorted(values)
    pos = (len(data) - 1) * q
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return float(data[int(pos)])
    frac = pos - low
    return float(data[low] * (1 - frac) + data[high] * frac)


def token_bytes(token: str) -> bytes:
    return token.encode("latin1")


def describe_bytes(data: bytes, limit: int = 32) -> str:
    snippet = data[:limit].decode("latin1").encode("unicode_escape").decode("ascii")
    if len(data) > limit:
        snippet += "…"
    return snippet


def analyze_tokenizer(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text())
    vocab = data["model"]["vocab"]
    tokens = [token_bytes(tok) for tok in vocab.keys()]
    total_tokens = len(tokens)
    if total_tokens == 0:
        raise ValueError(f"{path} has empty vocab")

    lengths = [len(tok) for tok in tokens]
    whitespace_only_lengths = []
    counts = {
        "ws_only": 0,
        "leading_ws": 0,
        "trailing_ws": 0,
        "internal_ws": 0,
        "newline": 0,
        "tab": 0,
        "digit": 0,
        "upper": 0,
        "alpha": 0,
        "non_ascii": 0,
        "non_printable": 0,
        "punct": 0,
        "len_ge_16": 0,
        "len_ge_32": 0,
        "len_eq_1": 0,
    }
    total_whitespace_bytes = 0
    total_bytes = sum(lengths)
    leading_ws_nonpure = 0
    trailing_ws_nonpure = 0

    for tok in tokens:
        if not tok:
            continue
        is_ws_only = all(b in ASCII_WHITESPACE for b in tok)
        has_ws = any(b in ASCII_WHITESPACE for b in tok)
        starts_ws = tok[0] in ASCII_WHITESPACE
        ends_ws = tok[-1] in ASCII_WHITESPACE
        has_non_ascii = any(b >= 128 for b in tok)
        has_non_printable = any((b < 32 and b not in PRINTABLE_WHITESPACE) or b == 127 for b in tok)
        has_digit = any(b in DIGITS for b in tok)
        has_upper = any(b in UPPER for b in tok)
        has_alpha = any(b in ALPHA for b in tok)
        has_punct = any(b in PUNCT for b in tok)
        has_newline = any(b == 10 for b in tok)
        has_tab = any(b == 9 for b in tok)

        if is_ws_only:
            counts["ws_only"] += 1
            whitespace_only_lengths.append(len(tok))
            total_whitespace_bytes += len(tok)
        elif has_ws:
            counts["internal_ws"] += 1
            total_whitespace_bytes += sum(1 for b in tok if b in ASCII_WHITESPACE)
        if starts_ws:
            counts["leading_ws"] += 1
            if not is_ws_only:
                leading_ws_nonpure += 1
        if ends_ws:
            counts["trailing_ws"] += 1
            if not is_ws_only:
                trailing_ws_nonpure += 1
        if has_newline:
            counts["newline"] += 1
        if has_tab:
            counts["tab"] += 1
        if has_digit:
            counts["digit"] += 1
        if has_upper:
            counts["upper"] += 1
        if has_alpha:
            counts["alpha"] += 1
        if has_non_ascii:
            counts["non_ascii"] += 1
        if has_non_printable:
            counts["non_printable"] += 1
        if has_punct:
            counts["punct"] += 1
        if len(tok) >= 16:
            counts["len_ge_16"] += 1
        if len(tok) >= 32:
            counts["len_ge_32"] += 1
        if len(tok) == 1:
            counts["len_eq_1"] += 1

    whitespace_ratio = total_whitespace_bytes / total_bytes if total_bytes else 0.0

    metrics = {
        "path": str(path),
        "token_count": total_tokens,
        "avg_len": sum(lengths) / total_tokens,
        "median_len": statistics.median(lengths),
        "p95_len": percentile(lengths, 0.95),
        "max_len": max(lengths),
        "share_len_ge16": counts["len_ge_16"] / total_tokens,
        "share_len_ge32": counts["len_ge_32"] / total_tokens,
        "share_len_eq1": counts["len_eq_1"] / total_tokens,
        "share_whitespace_only": counts["ws_only"] / total_tokens,
        "share_internal_whitespace": counts["internal_ws"] / total_tokens,
        "share_leading_whitespace": counts["leading_ws"] / total_tokens,
        "share_trailing_whitespace": counts["trailing_ws"] / total_tokens,
        "share_leading_whitespace_nonpure": leading_ws_nonpure / total_tokens,
        "share_trailing_whitespace_nonpure": trailing_ws_nonpure / total_tokens,
        "share_newline": counts["newline"] / total_tokens,
        "share_tab": counts["tab"] / total_tokens,
        "share_digit": counts["digit"] / total_tokens,
        "share_upper": counts["upper"] / total_tokens,
        "share_alpha": counts["alpha"] / total_tokens,
        "share_non_ascii": counts["non_ascii"] / total_tokens,
        "share_non_printable": counts["non_printable"] / total_tokens,
        "share_punct": counts["punct"] / total_tokens,
        "whitespace_byte_ratio": whitespace_ratio,
        "whitespace_only_avg_len": (sum(whitespace_only_lengths) / len(whitespace_only_lengths))
        if whitespace_only_lengths
        else 0.0,
        "whitespace_only_examples": [
            describe_bytes(tokens[idx])
            for idx, tok in enumerate(tokens)
            if all(b in ASCII_WHITESPACE for b in tok)
        ][:5],
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="tokenizer.json paths to inspect")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text table")
    args = parser.parse_args()

    reports = [analyze_tokenizer(path) for path in args.paths]
    if args.json:
        print(json.dumps(reports, indent=2))
        return

    header = (
        "path",
        "avg_len",
        "median",
        "p95",
        ">=16",
        "internal_ws",
        "lead_ws",
        "trail_ws",
        "lead_ws_nonpure",
        "trail_ws_nonpure",
        "whitespace_only",
        "non_ascii",
        "non_printable",
        "digits",
        "upper",
    )
    print("\t".join(header))
    for report in reports:
        row = [
            report["path"],
            f"{report['avg_len']:.2f}",
            f"{report['median_len']:.2f}",
            f"{report['p95_len']:.2f}",
            f"{report['share_len_ge16']*100:.2f}%",
            f"{report['share_internal_whitespace']*100:.2f}%",
            f"{report['share_leading_whitespace']*100:.2f}%",
            f"{report['share_trailing_whitespace']*100:.2f}%",
            f"{report['share_leading_whitespace_nonpure']*100:.2f}%",
            f"{report['share_trailing_whitespace_nonpure']*100:.2f}%",
            f"{report['share_whitespace_only']*100:.2f}%",
            f"{report['share_non_ascii']*100:.2f}%",
            f"{report['share_non_printable']*100:.2f}%",
            f"{report['share_digit']*100:.2f}%",
            f"{report['share_upper']*100:.2f}%",
        ]
        print("\t".join(row))


if __name__ == "__main__":
    main()
