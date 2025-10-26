#!/usr/bin/env bash
set -euo pipefail

# Runs bbpe train / chunk-train sweeps against a large sample corpus.
# Usage: scripts/run_sample_001.sh [PATH_TO_CORPUS]

DATA="${1:-$HOME/sample-001.txt}"
if [[ ! -f "$DATA" ]]; then
  echo "error: corpus '$DATA' not found" >&2
  exit 1
fi

TIMESTAMP="$(date +%s)"
OUTROOT="${OUTDIR:-/tmp/bbpe-sample-001-$TIMESTAMP}"
mkdir -p "$OUTROOT"

echo "➜ results will be stored under $OUTROOT"

if [[ ! -x target/release/bbpe ]]; then
  echo "➜ building release bbpe binary"
  cargo build --release
fi

MANIFEST="$OUTROOT/run_index.csv"
printf "label,kind,path\n" >"$MANIFEST"

TRAIN_LABEL="${TRAIN_LABEL:-train-baseline}"
TRAIN_VOCAB="${TRAIN_VOCAB:-4096}"
TRAIN_MIN_FREQ="${TRAIN_MIN_FREQ:-4}"
TRAIN_CHUNK_SIZE="${TRAIN_CHUNK_SIZE:-16777216}" # 16 MiB
TRAIN_THREADS="${TRAIN_THREADS:-}"
TRAIN_ENV_SEED="${TRAIN_SEED:-1337}"

run_train() {
  local label=$1
  local out="$OUTROOT/train-$label"
  shift

  mkdir -p "$out"
  echo "➜ [train] $label"
  local cmd=(target/release/bbpe train "$DATA" --vocab-size "$TRAIN_VOCAB" --min-frequency "$TRAIN_MIN_FREQ" --output "$out/tokenizer.json" --chunk-size "$TRAIN_CHUNK_SIZE" --no-progress)
  if [[ -n "$TRAIN_THREADS" ]]; then
    cmd+=(--threads "$TRAIN_THREADS")
  fi

  /usr/bin/time -v "${cmd[@]}" >"$out/stdout.log" 2>"$out/time.log"
  printf "%s,train,%s\n" "$label" "$out" >>"$MANIFEST"
}

CHUNK_GLOBAL_MERGES="${CHUNK_GLOBAL_MERGES:-4096}"
VALIDATION_SEED="${VALIDATION_SEED:-1337}"
FINAL_MERGES="${FINAL_MERGES:-$CHUNK_GLOBAL_MERGES}"

run_chunk_train() {
  local label=$1
  local chunk_size=$2
  local chunk_merges=$3
  local validation_fraction=$4
  local ensemble_mode=$5
  shift 5
  local extra_flags=("$@")

  local out="$OUTROOT/chunk-$label"
  mkdir -p "$out"

  echo "➜ [chunk-train] $label (chunk_size=$(numfmt --to=iec "$chunk_size"), merges=$chunk_merges, vf=$validation_fraction, mode=$ensemble_mode)"

  local cmd=(target/release/bbpe chunk-train "$DATA"
    --chunk-size "$chunk_size"
    --chunk-merges "$chunk_merges"
    --global-merges "$CHUNK_GLOBAL_MERGES"
    --output "$out/report.json"
    --final-tokenizer "$out/tokenizer.json"
    --final-merges "$FINAL_MERGES"
    --validation-fraction "$validation_fraction"
    --validation-seed "$VALIDATION_SEED"
    --ensemble-mode "$ensemble_mode"
    --verbose-chunks)

  if [[ -n "${CHUNK_THREADS:-}" ]]; then
    cmd+=(--threads "$CHUNK_THREADS")
  fi

  cmd+=("${extra_flags[@]}")

  /usr/bin/time -v "${cmd[@]}" >"$out/stdout.log" 2>"$out/time.log"
  printf "%s,chunk,%s\n" "$label" "$out" >>"$MANIFEST"
}

# Baseline train run.
run_train "$TRAIN_LABEL"

# Chunked scenarios (keep runtime <~3h).
run_chunk_train "32MiB-m1024-v08-baseline" $((32 * 1024 * 1024)) 1024 0.08 baseline
run_chunk_train "32MiB-m1024-v00-baseline" $((32 * 1024 * 1024)) 1024 0 baseline
run_chunk_train "64MiB-m2048-v05-baseline" $((64 * 1024 * 1024)) 2048 0.05 baseline
run_chunk_train "64MiB-m2048-v08-boost" $((64 * 1024 * 1024)) 2048 0.08 boost --boost-rounds 3 --boost-learning-rate 0.6
run_chunk_train "128MiB-m4096-v08-baseline" $((128 * 1024 * 1024)) 4096 0.08 baseline

echo "➜ generating summary"

python3 - <<'PY' "$MANIFEST" "$OUTROOT"
import json
import math
import re
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
outroot = Path(sys.argv[2])

entries = []
with manifest.open() as fh:
    next(fh)  # skip header
    for line in fh:
        label, kind, path = line.strip().split(",")
        entries.append((label, kind, Path(path)))

canonical_path = Path("/tmp/bbpe-alpine-full-compare-recheck-1761429586/train_tokenizer.json")
canonical_pairs = []
canonical_vocab = {}
if canonical_path.exists():
    with canonical_path.open() as f:
        data = json.load(f)
    canonical_pairs = [
        tuple(s.encode("latin-1") for s in pair) for pair in data["model"]["merges"]
    ]
    canonical_vocab = data["model"]["vocab"]
canonical_set = set(canonical_pairs)
canonical_top = set(canonical_pairs[:100])
canonical_bot = set(canonical_pairs[-100:]) if len(canonical_pairs) >= 100 else set()

def parse_time(log_path: Path):
    elapsed_re = re.compile(r"Elapsed \\(wall clock\\) time.*: (.+)$")
    rss_re = re.compile(r"Maximum resident set size \\(kbytes\\): (\\d+)")
    elapsed = None
    rss = None
    with log_path.open() as fh:
        for line in fh:
            if elapsed is None:
                m = elapsed_re.search(line)
                if m:
                    elapsed = m.group(1)
            if rss is None:
                m = rss_re.search(line)
                if m:
                    rss = int(m.group(1))
    secs = None
    if elapsed:
        parts = elapsed.split(":")
        if len(parts) == 3:
            h, m, s = parts
            secs = int(h) * 3600 + int(m) * 60 + float(s)
        elif len(parts) == 2:
            m, s = parts
            secs = int(m) * 60 + float(s)
        else:
            secs = float(parts[0])
    return secs, rss / 1024 if rss else None

rows = []
for label, kind, path in entries:
    time_log = path / "time.log"
    stdout_log = path / "stdout.log"
    wall, rss = parse_time(time_log) if time_log.exists() else (None, None)

    tokenizer_path = path / "tokenizer.json"
    merges = []
    tokens = []
    vocab_size = None
    if tokenizer_path.exists():
        with tokenizer_path.open() as f:
            data = json.load(f)
        vocab_size = data["model"]["vocab_size"] if "model" in data and "vocab_size" in data["model"] else None
        merges = [tuple(s.encode("latin-1") for s in pair) for pair in data["model"]["merges"]]
        vocab = data["model"]["vocab"]
        tokens = [None] * len(vocab)
        for tok, idx in vocab.items():
            if idx < len(tokens):
                tokens[idx] = tok
        tokens = [tok for tok in tokens if tok is not None]

    overlap = len(set(merges) & canonical_set) if canonical_set else None
    top_overlap = len(set(merges[:100]) & canonical_top) if canonical_top else None
    bottom_overlap = len(set(merges[-100:]) & canonical_bot) if canonical_bot else None

    lengths = [len(tok.encode("latin-1")) for tok in tokens] if tokens else []
    avg_len = sum(lengths) / len(lengths) if lengths else None
    ge8 = sum(1 for L in lengths if L >= 8) / len(lengths) if lengths else None

    rows.append({
        "label": label,
        "kind": kind,
        "wall": wall,
        "rss": rss,
        "overlap": overlap,
        "top_overlap": top_overlap,
        "bottom_overlap": bottom_overlap,
        "avg_len": avg_len,
        "ge8": ge8,
        "merges": len(merges),
        "vocab_size": vocab_size,
        "path": str(path),
    })

summary_path = outroot / "summary.csv"
with summary_path.open("w") as fh:
    fh.write("label,kind,wall_seconds,max_rss_mb,merges,vocab_size,overlap,top100_overlap,bottom100_overlap,avg_token_len,share_ge8,path\n")
    for row in rows:
        fh.write(",".join([
            row["label"],
            row["kind"],
            f"{row['wall']:.2f}" if row["wall"] else "",
            f"{row['rss']:.2f}" if row["rss"] else "",
            str(row["merges"]),
            str(row["vocab_size"]) if row["vocab_size"] else "",
            str(row["overlap"]) if row["overlap"] is not None else "",
            str(row["top_overlap"]) if row["top_overlap"] is not None else "",
            str(row["bottom_overlap"]) if row["bottom_overlap"] is not None else "",
            f"{row['avg_len']:.4f}" if row["avg_len"] else "",
            f"{row['ge8']:.4f}" if row["ge8"] else "",
            row["path"],
        ]) + "\n")

print("➜ summary written to", summary_path)

PY

echo "✓ done"
