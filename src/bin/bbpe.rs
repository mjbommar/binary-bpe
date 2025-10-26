use std::cmp::Ordering;
use std::collections::hash_map::Entry;
use std::collections::{BinaryHeap, VecDeque};
use std::convert::TryFrom;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use std::{env, fmt};

use anyhow::{anyhow, Context, Result};
use bbpe::bytes::{bytes_to_latin1, is_allowed_length};
use bbpe::config::{IngestConfig, TrainerConfig};
use bbpe::corpus::load_binary_corpus;
use bbpe::model::{BpeModel, Pair, TokenId};
use bbpe::serialization;
use bbpe::{BinaryTokenizer, Trainer, TrainerArtifacts};
use clap::{ArgAction, Args, Parser, Subcommand, ValueEnum};
use env_logger::Env;
use indicatif::{ProgressBar, ProgressStyle};
use log::{info, warn};
use rand::distributions::{Distribution, WeightedIndex};
use rand::rngs::StdRng;
use rand::seq::index::sample;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use serde_json::{self, json};

const DEFAULT_OUTPUT: &str = "tokenizer.json";
const DEFAULT_VALIDATION_SEED: u64 = 0x5641_4c49_4441_5445; // "VALIDATE"

#[derive(Parser, Debug)]
#[command(author, version, about = "Binary BPE toolkit", long_about = None)]
struct Cli {
    /// Increase verbosity (-v, -vv)
    #[arg(short = 'v', long, global = true, action = ArgAction::Count)]
    verbose: u8,

    /// Decrease verbosity (-q, -qq)
    #[arg(short = 'q', long, global = true, action = ArgAction::Count)]
    quiet: u8,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Train a new tokenizer from binary inputs
    Train(TrainArgs),
    /// Experimental chunked training prototype
    ChunkTrain(ChunkTrainArgs),
    /// Encode binary files with a trained tokenizer
    Encode(EncodeArgs),
    /// Decode token ids back into bytes
    Decode(DecodeArgs),
    /// Inspect tokenizer metadata
    Info(InfoArgs),
}

#[derive(Args, Debug)]
struct TrainArgs {
    /// Files or directories to ingest
    #[arg(required = true)]
    inputs: Vec<PathBuf>,

    /// Output path for tokenizer.json
    #[arg(short, long, value_name = "PATH", default_value = DEFAULT_OUTPUT)]
    output: PathBuf,

    /// Target vocabulary size
    #[arg(long, value_name = "SIZE")]
    vocab_size: Option<usize>,

    /// Minimum frequency for merges
    #[arg(long, value_name = "COUNT")]
    min_frequency: Option<usize>,

    /// Allowed token lengths in bytes (repeat flag)
    #[arg(long = "allowed-length", value_name = "LEN")]
    allowed_lengths: Vec<usize>,

    /// Override chunk size (0 = whole file)
    #[arg(long, value_name = "BYTES")]
    chunk_size: Option<usize>,

    /// Disable per-iteration logging/progress
    #[arg(long)]
    no_progress: bool,

    /// Append additional special tokens
    #[arg(long = "special-token", value_name = "TOKEN")]
    special_tokens: Vec<String>,

    /// Maximum merge iterations
    #[arg(long, value_name = "COUNT")]
    max_merge_iterations: Option<usize>,

    /// Plateau frequency floor
    #[arg(long, value_name = "FREQ")]
    plateau_floor: Option<usize>,

    /// Plateau patience
    #[arg(long, value_name = "COUNT")]
    plateau_patience: Option<usize>,

    /// Plateau frequency divisor
    #[arg(long, value_name = "DIV")]
    plateau_divisor: Option<usize>,

    /// Enable plateau-based early stopping
    #[arg(long)]
    plateau_stop: bool,

    /// Emit pretty JSON
    #[arg(long)]
    pretty: bool,

    /// Limit Rayon worker threads
    #[arg(long, value_name = "N")]
    threads: Option<usize>,

    /// Disable recursive directory traversal
    #[arg(long)]
    no_recursive: bool,

    /// Follow symlinks during traversal
    #[arg(long)]
    follow_symlinks: bool,
}

#[derive(Args, Debug)]
struct EncodeArgs {
    /// Tokenizer JSON to load
    #[arg(short = 'm', long, value_name = "PATH")]
    tokenizer: PathBuf,

    /// Binary inputs to encode
    #[arg(required = true)]
    inputs: Vec<PathBuf>,

    /// Emit JSON lines instead of human-readable output
    #[arg(long)]
    json: bool,

    /// Include special tokens during encoding
    #[arg(long)]
    add_special_tokens: bool,

    /// Optional directory to write .tokens files
    #[arg(long, value_name = "DIR")]
    output_dir: Option<PathBuf>,
}

#[derive(Args, Debug)]
struct ChunkTrainArgs {
    /// Files or directories to ingest
    #[arg(required = true)]
    inputs: Vec<PathBuf>,

    /// Output path for aggregate merge report (JSON)
    #[arg(long, value_name = "PATH", default_value = "chunked_merges.json")]
    output: PathBuf,

    /// Chunk size in bytes
    #[arg(
        long = "chunk-size",
        value_name = "BYTES",
        default_value_t = 33_554_432
    )]
    chunk_size: usize,

    /// Merge budget per chunk (excluding the 256 byte alphabet)
    #[arg(long = "chunk-merges", value_name = "COUNT", default_value_t = 512)]
    chunk_merges: usize,

    /// Maximum merges to keep after aggregation
    #[arg(long = "global-merges", value_name = "COUNT", default_value_t = 4096)]
    global_merges: usize,

    /// Minimum pair frequency threshold
    #[arg(long = "min-frequency", value_name = "COUNT", default_value_t = 4)]
    min_frequency: usize,

    /// Limit Rayon worker threads
    #[arg(long, value_name = "N")]
    threads: Option<usize>,

    /// Disable recursive directory traversal
    #[arg(long)]
    no_recursive: bool,

    /// Follow symlinks during traversal
    #[arg(long)]
    follow_symlinks: bool,

    /// Emit per-chunk progress logs
    #[arg(long)]
    verbose_chunks: bool,

    /// Optional path to emit a synthesized tokenizer using aggregated merges
    #[arg(long = "final-tokenizer", value_name = "PATH")]
    final_tokenizer: Option<PathBuf>,

    /// Maximum number of aggregated merges to realise in the synthesized tokenizer
    #[arg(long = "final-merges", value_name = "COUNT")]
    final_merges: Option<usize>,

    /// Minimum chunk support required for a merge to be retained
    #[arg(long = "min-chunk-support", value_name = "COUNT")]
    min_chunk_support: Option<u32>,

    /// Ranking strategy used when aggregating merges across chunks
    #[arg(long = "rank-mode", value_enum, default_value_t = ChunkRankMode::Weight)]
    rank_mode: ChunkRankMode,

    /// Additional weight multiplier applied to chunk support when rank-mode is balanced
    #[arg(long = "support-weight", value_name = "FACTOR", default_value_t = 1)]
    support_weight: u32,

    /// Ensemble aggregation strategy to use when combining chunk tokenizers
    #[arg(
        long = "ensemble-mode",
        value_enum,
        default_value_t = ChunkEnsembleMode::Baseline
    )]
    ensemble_mode: ChunkEnsembleMode,

    /// Number of boosting rounds to run (ensemble-mode=boost)
    #[arg(long = "boost-rounds", value_name = "COUNT", default_value_t = 1)]
    boost_rounds: usize,

    /// Number of chunks sampled per boosting round (defaults to all chunks)
    #[arg(long = "boost-sample-size", value_name = "COUNT")]
    boost_sample_size: Option<usize>,

    /// Learning rate applied to boost weights after each round
    #[arg(
        long = "boost-learning-rate",
        value_name = "RATE",
        default_value_t = 0.5
    )]
    boost_learning_rate: f64,

    /// Number of Monte Carlo sampling rounds (ensemble-mode=sampled)
    #[arg(long = "sample-rounds", value_name = "COUNT", default_value_t = 8)]
    sample_rounds: usize,

    /// Number of chunks drawn per sampling round (defaults to √N)
    #[arg(long = "sample-chunks", value_name = "COUNT")]
    sample_chunks: Option<usize>,

    /// Seed controlling ensemble randomness (boost/sample modes)
    #[arg(long = "ensemble-seed", value_name = "SEED")]
    ensemble_seed: Option<u64>,

    /// Fraction of each chunk reserved for outcome-driven validation (0.0-0.5)
    #[arg(
        long = "validation-fraction",
        value_name = "RATIO",
        default_value_t = 0.08
    )]
    validation_fraction: f64,

    /// RNG seed used when sampling validation slices
    #[arg(long = "validation-seed", value_name = "SEED")]
    validation_seed: Option<u64>,
}

/// Ensemble strategies for combining chunk-level training runs.
#[derive(Copy, Clone, Debug, ValueEnum)]
enum ChunkEnsembleMode {
    /// Aggregate every chunk exactly once (original behaviour).
    Baseline,
    /// Run weighted boosting rounds that emphasise under-covered chunks.
    Boost,
    /// Estimate merge utility via repeated random subsampling of chunks.
    Sampled,
}

impl fmt::Display for ChunkEnsembleMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            ChunkEnsembleMode::Baseline => "baseline",
            ChunkEnsembleMode::Boost => "boost",
            ChunkEnsembleMode::Sampled => "sampled",
        };
        f.write_str(label)
    }
}

/// Ranking strategies available for chunk aggregation.
#[derive(Copy, Clone, Debug, ValueEnum)]
enum ChunkRankMode {
    /// Prioritise total pair frequency across all chunks (default).
    Weight,
    /// Prioritise the number of chunks that observed the merge.
    Support,
    /// Blend frequency with chunk support to favour broadly useful merges.
    Balanced,
}

impl fmt::Display for ChunkRankMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            ChunkRankMode::Weight => "weight",
            ChunkRankMode::Support => "support",
            ChunkRankMode::Balanced => "balanced",
        };
        f.write_str(label)
    }
}

#[derive(Copy, Clone, Debug)]
struct ChunkRankOptions {
    mode: ChunkRankMode,
    support_weight: u32,
    min_chunk_support: Option<u32>,
}

impl ChunkRankOptions {
    fn accepts(&self, aggregate: &MergeAggregate) -> bool {
        match self.min_chunk_support {
            Some(min) => aggregate.chunk_support >= min,
            None => true,
        }
    }
}

#[derive(Args, Debug)]
struct DecodeArgs {
    /// Tokenizer JSON to load
    #[arg(short = 'm', long, value_name = "PATH")]
    tokenizer: PathBuf,

    /// Path to whitespace separated token ids
    #[arg(long, value_name = "PATH")]
    input: Option<PathBuf>,

    /// Token ids to decode when --input is omitted
    #[arg(value_name = "ID", required_unless_present = "input")]
    tokens: Vec<u32>,

    /// Skip special tokens while decoding
    #[arg(long)]
    skip_special_tokens: bool,

    /// Output file for decoded bytes (defaults to stdout)
    #[arg(long, value_name = "PATH")]
    output: Option<PathBuf>,
}

#[derive(Args, Debug)]
struct InfoArgs {
    /// Tokenizer JSON to inspect
    #[arg(short = 'm', long, value_name = "PATH")]
    tokenizer: PathBuf,

    /// Emit machine-readable JSON summary
    #[arg(long)]
    json: bool,
}

#[derive(Deserialize)]
struct TokenizerFile {
    model: ModelSection,
    #[serde(default)]
    added_tokens: Vec<AddedToken>,
}

#[derive(Deserialize)]
struct ModelSection {
    #[serde(rename = "type")]
    kind: String,
    vocab: serde_json::Map<String, serde_json::Value>,
    merges: Vec<serde_json::Value>,
    #[serde(default)]
    byte_fallback: bool,
}

#[derive(Deserialize)]
struct AddedToken {
    content: String,
    #[serde(default)]
    special: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    init_logging(cli.verbose, cli.quiet);

    match cli.command {
        Commands::Train(args) => run_train(args),
        Commands::ChunkTrain(args) => run_chunk_train(args),
        Commands::Encode(args) => run_encode(args),
        Commands::Decode(args) => run_decode(args),
        Commands::Info(args) => run_info(args),
    }
}

fn init_logging(verbose: u8, quiet: u8) {
    use log::LevelFilter;

    let level = if quiet > 0 {
        match quiet {
            0 => LevelFilter::Info,
            1 => LevelFilter::Warn,
            _ => LevelFilter::Error,
        }
    } else {
        match verbose {
            0 => LevelFilter::Info,
            1 => LevelFilter::Debug,
            _ => LevelFilter::Trace,
        }
    };

    let mut builder = env_logger::Builder::from_env(Env::default().default_filter_or("info"));
    builder.format_timestamp_millis();
    builder.filter_level(level);
    let _ = builder.try_init();
}

fn run_train(args: TrainArgs) -> Result<()> {
    if let Some(threads) = args.threads {
        ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .context("unable to configure Rayon thread pool")?;
    }

    let defaults = TrainerConfig::default();
    let mut cfg = TrainerConfig::builder();
    if let Some(vocab_size) = args.vocab_size {
        cfg = cfg.target_vocab_size(vocab_size);
    }
    if let Some(min_frequency) = args.min_frequency {
        cfg = cfg.min_frequency(min_frequency);
    }
    if !args.allowed_lengths.is_empty() {
        cfg = cfg.allowed_token_lengths(args.allowed_lengths.clone());
    }
    if !args.special_tokens.is_empty() {
        cfg = cfg.special_tokens(args.special_tokens.clone());
    }
    if args.plateau_floor.is_some()
        || args.plateau_patience.is_some()
        || args.plateau_divisor.is_some()
    {
        cfg = cfg.plateau_thresholds(
            args.plateau_floor
                .unwrap_or(defaults.plateau_frequency_floor),
            args.plateau_patience.unwrap_or(defaults.plateau_patience),
            args.plateau_divisor
                .unwrap_or(defaults.plateau_frequency_divisor),
        );
    }
    cfg = cfg.max_merge_iterations(args.max_merge_iterations);
    cfg = cfg.plateau_stop_enabled(args.plateau_stop);
    cfg = cfg.show_progress(!args.no_progress);
    let trainer_cfg = cfg.build()?;

    let ingest_cfg = IngestConfig {
        chunk_size: args.chunk_size.unwrap_or(defaults_chunk_size()),
        recursive: !args.no_recursive,
        follow_symlinks: args.follow_symlinks,
    };

    let sequences = load_binary_corpus(&args.inputs, &ingest_cfg)
        .with_context(|| "failed to load binary corpus")?;
    let corpus_bytes: usize = sequences.iter().map(|seq| seq.len()).sum();
    info!(
        "loaded {} sequences totalling {:.2} MiB",
        sequences.len(),
        bytes_to_mebibytes(corpus_bytes)
    );

    let spinner = if args.no_progress {
        None
    } else {
        let pb = ProgressBar::new_spinner();
        let style = ProgressStyle::with_template("{spinner} training merges... {elapsed}")
            .unwrap()
            .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏");
        pb.set_style(style);
        pb.enable_steady_tick(Duration::from_millis(80));
        Some(pb)
    };

    let trainer = Trainer::new(trainer_cfg.clone());
    let start = Instant::now();
    let artifacts = trainer.train_from_sequences(&sequences)?;
    drop(sequences);
    if let Some(pb) = spinner {
        pb.finish_with_message("training complete");
    }

    let elapsed = start.elapsed();
    let merges = artifacts.model.merges().len();
    let vocab_size = artifacts.model.vocab_size();
    let throughput = if elapsed.as_secs_f64() > 0.0 {
        bytes_to_mebibytes(corpus_bytes) / elapsed.as_secs_f64()
    } else {
        0.0
    };

    artifacts
        .model
        .save_huggingface(&args.output)
        .with_context(|| format!("failed to save tokenizer to {}", args.output.display()))?;
    if args.pretty {
        let pretty = artifacts.model.to_huggingface_json(true)?;
        fs::write(&args.output, pretty)
            .with_context(|| format!("failed to pretty print {}", args.output.display()))?;
    }

    info!(
        "training complete: merges={merges} vocab={vocab_size} duration={elapsed:.2?} throughput={throughput:.2} MiB/s"
    );
    println!(
        "✅ wrote tokenizer with vocab {} ({} merges) to {}",
        vocab_size,
        merges,
        args.output.display()
    );
    println!(
        "   corpus {:.2} MiB | duration {:.2?} | throughput {:.2} MiB/s",
        bytes_to_mebibytes(corpus_bytes),
        elapsed,
        throughput
    );

    Ok(())
}

fn run_chunk_train(args: ChunkTrainArgs) -> Result<()> {
    if args.chunk_size == 0 {
        return Err(anyhow!("chunk-size must be greater than zero"));
    }
    if args.chunk_merges == 0 {
        return Err(anyhow!("chunk-merges must be greater than zero"));
    }
    if args.global_merges == 0 {
        return Err(anyhow!("global-merges must be greater than zero"));
    }
    if args.support_weight == 0 {
        return Err(anyhow!("support-weight must be greater than zero"));
    }

    if let Some(threads) = args.threads {
        ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .context("unable to configure Rayon thread pool")?;
    }

    let default_special_tokens = TrainerConfig::default().special_tokens;
    let ingest_cfg = IngestConfig {
        chunk_size: args.chunk_size,
        recursive: !args.no_recursive,
        follow_symlinks: args.follow_symlinks,
    };

    let chunks =
        load_binary_corpus(&args.inputs, &ingest_cfg).with_context(|| "failed to load corpus")?;
    let total_chunks = chunks.len();
    if total_chunks == 0 {
        return Err(anyhow!("no data loaded for chunked training"));
    }

    let trainer_cfg = TrainerConfig::builder()
        .target_vocab_size(256 + args.chunk_merges + default_special_tokens.len())
        .max_merge_iterations(Some(args.chunk_merges))
        .min_frequency(args.min_frequency)
        .special_tokens(default_special_tokens.clone())
        .show_progress(false)
        .plateau_stop_enabled(false)
        .build()?;
    let trainer = Trainer::new(trainer_cfg);
    let rank_opts = ChunkRankOptions {
        mode: args.rank_mode,
        support_weight: args.support_weight.max(1),
        min_chunk_support: args.min_chunk_support,
    };
    let validation_fraction = args.validation_fraction.clamp(0.0, 0.5);
    let mut validation_rng = if validation_fraction > f64::EPSILON {
        Some(StdRng::seed_from_u64(
            args.validation_seed.unwrap_or(DEFAULT_VALIDATION_SEED),
        ))
    } else {
        None
    };
    let single_chunk = total_chunks == 1;
    let mut single_chunk_artifacts: Option<TrainerArtifacts> = None;

    let mut original_bytes = 0usize;
    let mut total_bytes = 0usize;
    let mut validation_bytes = 0usize;
    let mut validation_pool: Vec<Vec<u8>> = Vec::new();
    let mut chunk_data: Vec<ChunkData> = Vec::with_capacity(total_chunks);

    for (index, chunk) in chunks.into_iter().enumerate() {
        let chunk_len = chunk.len();
        original_bytes += chunk_len;
        let (training_bytes, validation_opt) = if let Some(rng) = validation_rng.as_mut() {
            split_chunk_for_validation(chunk, validation_fraction, rng)
        } else {
            (chunk, None)
        };
        if let Some(eval) = validation_opt {
            validation_bytes += eval.len();
            if !eval.is_empty() {
                validation_pool.push(eval);
            }
        }
        total_bytes += training_bytes.len();
        let sequences = vec![training_bytes];

        let artifacts = trainer
            .train_from_sequences(&sequences)
            .with_context(|| format!("failed to train chunk {index}"))?;
        if single_chunk {
            single_chunk_artifacts = Some(artifacts.clone());
        }
        let merges = artifacts.model.merges();
        let token_bytes = artifacts.model.token_bytes();
        let iteration_metrics = &artifacts.metrics.iterations;

        let mut records = Vec::with_capacity(merges.len());
        let mut preview = Vec::new();
        for (merge_idx, &(left_id, right_id)) in merges.iter().enumerate() {
            if merge_idx >= iteration_metrics.len() {
                break;
            }
            let left_bytes = token_bytes[left_id as usize].clone();
            let right_bytes = token_bytes[right_id as usize].clone();
            if preview.len() < 3 {
                preview.push(format!(
                    "{}+{}",
                    bytes_to_latin1(&left_bytes),
                    bytes_to_latin1(&right_bytes)
                ));
            }
            let key = MergeKey {
                left: left_bytes,
                right: right_bytes,
            };
            let metrics = &iteration_metrics[merge_idx];
            records.push(ChunkMergeRecord {
                key,
                best_frequency: metrics.best_frequency as u64,
                merges_applied: metrics.merges_applied as u64,
            });
        }

        let max_frequency = iteration_metrics.iter().map(|m| m.best_frequency).max();
        let summary = ChunkSummary {
            index,
            byte_len: sequences[0].len(),
            merges_generated: merges.len(),
            max_frequency,
            preview_merges: preview,
        };
        if args.verbose_chunks {
            info!(
                "chunk {:>4}: bytes={} merges={} top_freq={:?}",
                summary.index, summary.byte_len, summary.merges_generated, summary.max_frequency
            );
        }

        chunk_data.push(ChunkData { summary, records });
    }

    let chunk_summaries: Vec<ChunkSummary> =
        chunk_data.iter().map(|data| data.summary.clone()).collect();

    if validation_bytes > 0 {
        let pct = if original_bytes > 0 {
            (validation_bytes as f64 / original_bytes as f64) * 100.0
        } else {
            0.0
        };
        info!(
            "reserved {:.2} MiB ({:.2}%) for validation pool",
            bytes_to_mebibytes(validation_bytes),
            pct
        );
    }

    let mut key_to_chunks: FxHashMap<MergeKey, Vec<usize>> = FxHashMap::default();
    for (chunk_idx, data) in chunk_data.iter().enumerate() {
        for record in &data.records {
            key_to_chunks
                .entry(record.key.clone())
                .or_insert_with(Vec::new)
                .push(chunk_idx);
        }
    }

    let ensemble_seed = args.ensemble_seed.unwrap_or(0x42425045_4d454e53);
    let boost_sample_size = args
        .boost_sample_size
        .map(|value| value.clamp(1, total_chunks))
        .unwrap_or(total_chunks.max(1));
    let default_sample = ((total_chunks as f64).sqrt().ceil() as usize).max(1);
    let sample_chunk_count = args
        .sample_chunks
        .map(|value| value.clamp(1, total_chunks))
        .unwrap_or_else(|| default_sample.clamp(1, total_chunks));

    let (ranked, total_candidates, dropped_due_to_support, boost_details, sample_details) =
        if single_chunk && matches!(args.ensemble_mode, ChunkEnsembleMode::Baseline) {
            let artifacts = single_chunk_artifacts
                .clone()
                .expect("single chunk artifacts should be captured");
            let merges = artifacts.model.merges();
            let token_bytes = artifacts.model.token_bytes();
            let iteration_metrics = &artifacts.metrics.iterations;
            let mut result = Vec::with_capacity(merges.len());
            let mut dropped = 0usize;
            for (idx, &(left_id, right_id)) in merges.iter().enumerate() {
                let key = MergeKey {
                    left: token_bytes[left_id as usize].clone(),
                    right: token_bytes[right_id as usize].clone(),
                };
                let metrics = &iteration_metrics[idx];
                let aggregate = MergeAggregate {
                    total_weight: metrics.best_frequency as u64,
                    total_merges: metrics.merges_applied as u64,
                    chunk_support: 1,
                };
                if rank_opts.accepts(&aggregate) {
                    result.push((key, aggregate));
                } else {
                    dropped += 1;
                }
            }
            (result, merges.len(), dropped, None, None)
        } else {
            let mut boost_details_opt = None;
            let mut sample_details_opt = None;
            let aggregation = match args.ensemble_mode {
                ChunkEnsembleMode::Baseline => aggregate_all(&chunk_data, &rank_opts),
                ChunkEnsembleMode::Boost => {
                    let mut rng = StdRng::seed_from_u64(ensemble_seed);
                    let (result, details) = aggregate_boost(
                        &chunk_data,
                        &rank_opts,
                        args.global_merges,
                        args.boost_rounds,
                        boost_sample_size,
                        args.boost_learning_rate,
                        &key_to_chunks,
                        &mut rng,
                    );
                    boost_details_opt = Some(details);
                    result
                }
                ChunkEnsembleMode::Sampled => {
                    let mut rng = StdRng::seed_from_u64(ensemble_seed);
                    let (result, details) = aggregate_sampled(
                        &chunk_data,
                        &rank_opts,
                        args.global_merges,
                        args.sample_rounds,
                        sample_chunk_count,
                        &key_to_chunks,
                        &mut rng,
                    );
                    sample_details_opt = Some(details);
                    result
                }
            };
            (
                aggregation.ranked,
                aggregation.total_candidates,
                aggregation.dropped_below_support,
                boost_details_opt,
                sample_details_opt,
            )
        };

    if dropped_due_to_support > 0 {
        if let Some(min_support) = rank_opts.min_chunk_support {
            info!(
                "filtered out {} merges below min-chunk-support {}",
                dropped_due_to_support, min_support
            );
        }
    }

    let mut report_ranked = ranked.clone();
    if report_ranked.len() > args.global_merges {
        report_ranked.truncate(args.global_merges);
    }

    let global_merges: Vec<MergeSummary> = report_ranked
        .iter()
        .map(|(key, agg)| MergeSummary::from_key(key, agg))
        .collect();

    let final_summary = if let Some(path) = &args.final_tokenizer {
        let final_limit = args.final_merges.unwrap_or(args.global_merges);
        if final_limit == 0 {
            warn!("final-merges resolved to zero; skipping tokenizer synthesis");
            None
        } else if single_chunk && dropped_due_to_support == 0 && final_limit >= ranked.len() {
            let artifacts = single_chunk_artifacts
                .clone()
                .expect("single chunk artifacts should be captured");

            if let Some(parent) = path.parent() {
                if !parent.as_os_str().is_empty() {
                    fs::create_dir_all(parent)
                        .with_context(|| format!("failed to create {}", parent.display()))?;
                }
            }

            artifacts
                .model
                .save_huggingface(path)
                .with_context(|| format!("failed to write tokenizer {}", path.display()))?;

            let vocab_size = artifacts.model.vocab_size();
            println!(
                "🧪 chunked tokenizer: reused single-chunk model with {} merges, vocab {} -> {}",
                ranked.len(),
                vocab_size,
                path.display()
            );

            Some(FinalTokenizerSummary {
                merges_requested: ranked.len(),
                merges_realized: ranked.len(),
                duplicates_skipped: 0,
                unresolved_dependencies: 0,
                vocab_size,
                tokenizer_path: path.display().to_string(),
            })
        } else {
            let validation_threshold = if validation_pool.is_empty() { 0 } else { 1 };
            let outcome_attempt =
                outcome_finalize(&ranked, final_limit, &validation_pool, validation_threshold);
            let mut used_outcome = false;
            let assembly = match outcome_attempt {
                Ok(Some(assembly)) if assembly.stats.realised > 0 => {
                    info!(
                        "outcome-driven pass realised {} merges on validation pool",
                        assembly.stats.realised
                    );
                    used_outcome = true;
                    assembly
                }
                Ok(Some(_assembly)) => {
                    warn!(
                        "outcome-driven finaliser produced 0 merges; reverting to dependency replay"
                    );
                    assemble_final_tokenizer(&ranked, final_limit)?
                }
                Ok(None) => {
                    if !validation_pool.is_empty() {
                        warn!(
                            "validation pool was insufficient to score merges; falling back to dependency replay"
                        );
                    }
                    assemble_final_tokenizer(&ranked, final_limit)?
                }
                Err(err) => {
                    warn!(
                        "outcome-driven finaliser failed ({}); falling back to dependency replay",
                        err
                    );
                    assemble_final_tokenizer(&ranked, final_limit)?
                }
            };
            if !used_outcome {
                info!(
                    "dependency replay realised {} merges after fallback",
                    assembly.stats.realised
                );
            }
            let FinalAssembly {
                token_bytes,
                merges,
                stats,
            } = assembly;

            let max_len = token_bytes.iter().map(|tok| tok.len()).max().unwrap_or(1);
            let mut allowed_lengths = vec![1usize];
            if max_len > 1 {
                allowed_lengths.push(max_len);
            }

            let total_vocab = token_bytes.len() + default_special_tokens.len();
            let trainer_cfg = TrainerConfig::builder()
                .target_vocab_size(total_vocab)
                .min_frequency(args.min_frequency)
                .allowed_token_lengths(allowed_lengths)
                .special_tokens(default_special_tokens.clone())
                .show_progress(false)
                .plateau_stop_enabled(false)
                .max_merge_iterations(Some(stats.realised))
                .build()?;

            if let Some(parent) = path.parent() {
                if !parent.as_os_str().is_empty() {
                    fs::create_dir_all(parent)
                        .with_context(|| format!("failed to create {}", parent.display()))?;
                }
            }

            let model = BpeModel::new(token_bytes, merges, trainer_cfg);
            model
                .save_huggingface(path)
                .with_context(|| format!("failed to write tokenizer {}", path.display()))?;

            let vocab_size = model.vocab_size();
            if stats.realised == 0 {
                warn!(
                    "chunked synthesis produced no merges (requested {}); tokenizer is byte-only",
                    stats.requested
                );
            } else {
                println!(
                    "🧪 chunked tokenizer: realised {} merges (requested {}), vocab {} -> {}",
                    stats.realised,
                    stats.requested,
                    vocab_size,
                    path.display()
                );
                if used_outcome {
                    info!("final tokenizer produced via outcome-driven validation pass");
                }
            }
            if stats.duplicates > 0 {
                warn!(
                    "{} aggregated merges were duplicates of earlier tokens and were skipped",
                    stats.duplicates
                );
            }
            if stats.unresolved > 0 {
                warn!(
                    "{} aggregated merges could not be realised due to missing dependencies",
                    stats.unresolved
                );
            }

            Some(FinalTokenizerSummary {
                merges_requested: stats.requested,
                merges_realized: stats.realised,
                duplicates_skipped: stats.duplicates,
                unresolved_dependencies: stats.unresolved,
                vocab_size,
                tokenizer_path: path.display().to_string(),
            })
        }
    } else {
        None
    };

    let validation_fraction_report = if validation_bytes > 0 && original_bytes > 0 {
        Some(validation_bytes as f64 / original_bytes as f64)
    } else {
        None
    };

    let report = ChunkedReport {
        chunk_size_bytes: args.chunk_size,
        chunk_merge_target: args.chunk_merges,
        global_merge_target: args.global_merges,
        min_frequency: args.min_frequency,
        rank_mode: args.rank_mode.to_string(),
        support_weight: rank_opts.support_weight,
        ensemble_mode: args.ensemble_mode.to_string(),
        min_chunk_support: rank_opts.min_chunk_support,
        aggregated_candidates: total_candidates,
        retained_merges: ranked.len(),
        chunks_processed: chunk_summaries.len(),
        total_bytes,
        validation_bytes: (validation_bytes > 0).then_some(validation_bytes),
        validation_fraction: validation_fraction_report,
        global_merges,
        chunks: chunk_summaries,
        boost_rounds: if matches!(args.ensemble_mode, ChunkEnsembleMode::Boost) {
            Some(args.boost_rounds.max(1))
        } else {
            None
        },
        boost_learning_rate: if matches!(args.ensemble_mode, ChunkEnsembleMode::Boost) {
            Some(args.boost_learning_rate)
        } else {
            None
        },
        boost_sample_size: if matches!(args.ensemble_mode, ChunkEnsembleMode::Boost) {
            Some(boost_sample_size)
        } else {
            None
        },
        sample_rounds: if matches!(args.ensemble_mode, ChunkEnsembleMode::Sampled) {
            Some(args.sample_rounds.max(1))
        } else {
            None
        },
        sample_chunks: if matches!(args.ensemble_mode, ChunkEnsembleMode::Sampled) {
            Some(sample_chunk_count)
        } else {
            None
        },
        ensemble_seed: if matches!(args.ensemble_mode, ChunkEnsembleMode::Baseline) {
            None
        } else {
            Some(ensemble_seed)
        },
        boost_details,
        sample_details,
        final_tokenizer: final_summary.clone(),
    };

    write_chunk_report(&args.output, &report)?;

    println!(
        "✅ processed {} chunks (~{:.2} MiB); kept {} global merges -> {}",
        report.chunks_processed,
        bytes_to_mebibytes(report.total_bytes),
        report.global_merges.len(),
        args.output.display()
    );

    Ok(())
}

fn run_encode(args: EncodeArgs) -> Result<()> {
    let tokenizer = serialization::load_tokenizer(&args.tokenizer)
        .with_context(|| format!("failed to load tokenizer from {}", args.tokenizer.display()))?;
    let tokenizer = BinaryTokenizer::from_tokenizer(tokenizer)?;

    if let Some(dir) = &args.output_dir {
        if !dir.exists() {
            fs::create_dir_all(dir)
                .with_context(|| format!("failed to create output directory {}", dir.display()))?;
        }
    }

    for path in &args.inputs {
        let mut file =
            File::open(path).with_context(|| format!("failed to open input {}", path.display()))?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)
            .with_context(|| format!("failed to read {}", path.display()))?;

        let tokens = tokenizer.encode_bytes(&data, args.add_special_tokens)?;
        if let Some(dir) = &args.output_dir {
            let mut out_path = dir.clone();
            let filename = path
                .file_name()
                .map(|name| name.to_string_lossy().to_string())
                .unwrap_or_else(|| "stdin".to_string());
            out_path.push(format!("{filename}.tokens"));
            let mut output = File::create(&out_path)
                .with_context(|| format!("failed to open {}", out_path.display()))?;
            write_token_sequence(&mut output, &tokens)?;
            println!("{} => {}", path.display(), out_path.display());
        } else if args.json {
            let record = json!({
                "path": path.display().to_string(),
                "tokens": tokens
            });
            println!("{}", serde_json::to_string(&record)?);
        } else {
            print!("{}:\t", path.display());
            for (idx, token) in tokens.iter().enumerate() {
                if idx > 0 {
                    print!(" ");
                }
                print!("{token}");
            }
            println!();
        }
    }

    Ok(())
}

fn run_decode(args: DecodeArgs) -> Result<()> {
    let tokenizer = serialization::load_tokenizer(&args.tokenizer)
        .with_context(|| format!("failed to load tokenizer from {}", args.tokenizer.display()))?;
    let tokenizer = BinaryTokenizer::from_tokenizer(tokenizer)?;

    let tokens = if let Some(input_path) = &args.input {
        let contents = fs::read_to_string(input_path)
            .with_context(|| format!("failed to read {}", input_path.display()))?;
        parse_token_list(&contents)?
    } else {
        args.tokens
    };

    let bytes = tokenizer.decode_to_bytes(&tokens, args.skip_special_tokens)?;

    if let Some(path) = &args.output {
        let mut file =
            File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
        file.write_all(&bytes)
            .with_context(|| format!("failed to write {}", path.display()))?;
        println!("wrote {} bytes to {}", bytes.len(), path.display());
    } else {
        io::stdout().write_all(&bytes)?;
    }

    Ok(())
}

fn run_info(args: InfoArgs) -> Result<()> {
    let data = fs::read_to_string(&args.tokenizer)
        .with_context(|| format!("failed to read {}", args.tokenizer.display()))?;
    let parsed: TokenizerFile =
        serde_json::from_str(&data).context("failed to parse tokenizer.json")?;

    let vocab_size = parsed.model.vocab.len();
    let merges = parsed.model.merges.len();
    let special_tokens = parsed
        .added_tokens
        .iter()
        .filter(|token| token.special)
        .map(|token| token.content.clone())
        .collect::<Vec<_>>();
    let summary = json!({
        "path": args.tokenizer.display().to_string(),
        "model_type": parsed.model.kind,
        "vocab_size": vocab_size,
        "merges": merges,
        "byte_fallback": parsed.model.byte_fallback,
        "special_tokens": special_tokens,
    });

    if args.json {
        println!("{}", serde_json::to_string_pretty(&summary)?);
    } else {
        let model_type = summary["model_type"].as_str().unwrap_or("unknown");
        let byte_fallback = parsed.model.byte_fallback;
        println!("Model type   : {model_type}");
        println!("Vocab size   : {vocab_size}");
        println!("Merges       : {merges}");
        println!("Byte fallback: {byte_fallback}");
        if summary["special_tokens"].is_array()
            && !summary["special_tokens"].as_array().unwrap().is_empty()
        {
            println!(
                "Special tokens: {}",
                summary["special_tokens"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_str().unwrap_or_default())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        } else {
            println!("Special tokens: (none)");
        }
    }

    Ok(())
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct MergeKey {
    left: Vec<u8>,
    right: Vec<u8>,
}

#[derive(Default, Clone, Copy)]
struct MergeAggregate {
    total_weight: u64,
    total_merges: u64,
    chunk_support: u32,
}

#[derive(Default, Clone)]
struct MergeAggregateBuilder {
    total_weight: u64,
    total_merges: u64,
    chunk_ids: FxHashSet<u32>,
}

impl MergeAggregateBuilder {
    fn include(&mut self, chunk_idx: usize, record: &ChunkMergeRecord) {
        self.total_weight = self.total_weight.saturating_add(record.best_frequency);
        self.total_merges = self.total_merges.saturating_add(record.merges_applied);
        let id = u32::try_from(chunk_idx).unwrap_or(u32::MAX);
        self.chunk_ids.insert(id);
    }

    fn extend(&mut self, other: MergeAggregateBuilder) {
        self.total_weight = self.total_weight.saturating_add(other.total_weight);
        self.total_merges = self.total_merges.saturating_add(other.total_merges);
        self.chunk_ids.extend(other.chunk_ids);
    }

    fn as_merge_aggregate(&self) -> MergeAggregate {
        let support = u32::try_from(self.chunk_ids.len()).unwrap_or(u32::MAX);
        MergeAggregate {
            total_weight: self.total_weight,
            total_merges: self.total_merges,
            chunk_support: support,
        }
    }
}

impl MergeAggregate {
    fn compare(&self, other: &Self, opts: &ChunkRankOptions) -> std::cmp::Ordering {
        match opts.mode {
            ChunkRankMode::Weight => self.weight_key().cmp(&other.weight_key()),
            ChunkRankMode::Support => self.support_key().cmp(&other.support_key()),
            ChunkRankMode::Balanced => self
                .balanced_key(opts.support_weight)
                .cmp(&other.balanced_key(opts.support_weight)),
        }
    }

    fn weight_key(&self) -> (u64, u64, u32) {
        (self.total_weight, self.total_merges, self.chunk_support)
    }

    fn support_key(&self) -> (u32, u64, u64) {
        (self.chunk_support, self.total_weight, self.total_merges)
    }

    fn balanced_key(&self, support_weight: u32) -> (u128, u64, u32) {
        let support_factor =
            1u128 + (self.chunk_support as u128).saturating_mul(support_weight as u128);
        let combined = (self.total_weight as u128).saturating_mul(support_factor);
        (combined, self.total_merges, self.chunk_support)
    }
}

#[derive(Clone)]
struct ChunkMergeRecord {
    key: MergeKey,
    best_frequency: u64,
    merges_applied: u64,
}

#[derive(Clone)]
struct ChunkData {
    summary: ChunkSummary,
    records: Vec<ChunkMergeRecord>,
}

#[derive(Default)]
struct AggregationResult {
    ranked: Vec<(MergeKey, MergeAggregate)>,
    total_candidates: usize,
    dropped_below_support: usize,
}

#[derive(Serialize, Clone)]
struct BoostRoundSummary {
    round: usize,
    sample_size: usize,
    coverage_mean: f64,
    coverage_min: f64,
    coverage_max: f64,
    weight_min: f64,
    weight_max: f64,
}

#[derive(Serialize, Clone)]
struct SampleRoundSummary {
    round: usize,
    sample_size: usize,
    coverage_mean: f64,
    coverage_min: f64,
    coverage_max: f64,
}

#[derive(Serialize, Clone)]
struct ChunkSummary {
    index: usize,
    byte_len: usize,
    merges_generated: usize,
    max_frequency: Option<usize>,
    preview_merges: Vec<String>,
}

#[derive(Serialize)]
struct MergeSummary {
    left: String,
    right: String,
    combined: String,
    total_weight: u64,
    total_merges: u64,
    chunk_support: u32,
    left_len: usize,
    right_len: usize,
}

impl MergeSummary {
    fn from_key(key: &MergeKey, aggregate: &MergeAggregate) -> Self {
        let mut combined = Vec::with_capacity(key.left.len() + key.right.len());
        combined.extend_from_slice(&key.left);
        combined.extend_from_slice(&key.right);
        Self {
            left: bytes_to_latin1(&key.left),
            right: bytes_to_latin1(&key.right),
            combined: bytes_to_latin1(&combined),
            total_weight: aggregate.total_weight,
            total_merges: aggregate.total_merges,
            chunk_support: aggregate.chunk_support,
            left_len: key.left.len(),
            right_len: key.right.len(),
        }
    }
}

#[derive(Serialize)]
struct ChunkedReport {
    chunk_size_bytes: usize,
    chunk_merge_target: usize,
    global_merge_target: usize,
    min_frequency: usize,
    rank_mode: String,
    support_weight: u32,
    ensemble_mode: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    min_chunk_support: Option<u32>,
    aggregated_candidates: usize,
    retained_merges: usize,
    chunks_processed: usize,
    total_bytes: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    validation_bytes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    validation_fraction: Option<f64>,
    global_merges: Vec<MergeSummary>,
    chunks: Vec<ChunkSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    boost_rounds: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    boost_learning_rate: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    boost_sample_size: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sample_rounds: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sample_chunks: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ensemble_seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    boost_details: Option<Vec<BoostRoundSummary>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sample_details: Option<Vec<SampleRoundSummary>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    final_tokenizer: Option<FinalTokenizerSummary>,
}

#[derive(Serialize, Clone)]
struct FinalTokenizerSummary {
    merges_requested: usize,
    merges_realized: usize,
    duplicates_skipped: usize,
    unresolved_dependencies: usize,
    vocab_size: usize,
    tokenizer_path: String,
}

struct FinalAssembly {
    token_bytes: Vec<Vec<u8>>,
    merges: Vec<Pair>,
    stats: FinalMergeStats,
}

struct FinalMergeStats {
    requested: usize,
    realised: usize,
    duplicates: usize,
    unresolved: usize,
}

struct CandidateState {
    key: MergeKey,
    pair: Option<Pair>,
    used: bool,
}

impl CandidateState {
    fn new(key: MergeKey) -> Self {
        Self {
            key,
            pair: None,
            used: false,
        }
    }
}

fn outcome_finalize(
    ranked: &[(MergeKey, MergeAggregate)],
    limit: usize,
    validation_samples: &[Vec<u8>],
    validation_threshold: usize,
) -> Result<Option<FinalAssembly>> {
    if validation_samples.is_empty() || limit == 0 {
        return Ok(None);
    }

    let mut sequences: Vec<Vec<TokenId>> = validation_samples
        .iter()
        .filter_map(|sample| {
            if sample.is_empty() {
                None
            } else {
                Some(sample.iter().map(|&b| TokenId::from(b)).collect())
            }
        })
        .collect();
    if sequences.is_empty() {
        return Ok(None);
    }

    let mut token_bytes: Vec<Vec<u8>> = (0u8..=u8::MAX).map(|b| vec![b]).collect();
    let mut token_map: FxHashMap<Vec<u8>, TokenId> = FxHashMap::default();
    token_map.reserve(token_bytes.len().saturating_mul(2));
    for (idx, token) in token_bytes.iter().enumerate() {
        token_map.insert(token.clone(), idx as TokenId);
    }
    let mut token_lengths: Vec<usize> = vec![1; token_bytes.len()];
    let mut allowed_lengths = TrainerConfig::default().allowed_token_lengths;

    let target_merges = ranked.len().min(limit);
    if target_merges == 0 {
        let stats = FinalMergeStats {
            requested: 0,
            realised: 0,
            duplicates: 0,
            unresolved: 0,
        };
        return Ok(Some(FinalAssembly {
            token_bytes,
            merges: Vec::new(),
            stats,
        }));
    }

    let mut pair_counts = final_compute_pair_counts(&sequences, &token_lengths, &allowed_lengths);
    let mut heap = BinaryHeap::new();
    let mut candidates: Vec<CandidateState> = ranked
        .iter()
        .map(|(key, _)| CandidateState::new(key.clone()))
        .collect();
    let mut candidate_lookup: FxHashMap<Pair, usize> = FxHashMap::default();

    refresh_candidate_pairs(
        &mut candidates,
        &token_map,
        &pair_counts,
        &mut heap,
        &mut candidate_lookup,
        validation_threshold,
    );

    let mut merges: Vec<Pair> = Vec::with_capacity(target_merges);

    while merges.len() < target_merges {
        let maybe_best = loop {
            match heap.pop() {
                Some(score) => {
                    let current = pair_counts.get(&score.pair).copied().unwrap_or(0);
                    if current == 0 || current != score.frequency {
                        continue;
                    }
                    if let Some(&idx) = candidate_lookup.get(&score.pair) {
                        if candidates[idx].used {
                            continue;
                        }
                        break Some((score.pair, idx, current));
                    }
                    continue;
                }
                None => break None,
            }
        };

        let Some((pair, idx, _freq)) = maybe_best else {
            break;
        };

        let combined_len = token_lengths[pair.0 as usize] + token_lengths[pair.1 as usize];
        let mut combined_bytes = token_bytes[pair.0 as usize].clone();
        combined_bytes.extend_from_slice(&token_bytes[pair.1 as usize]);
        let new_token_id = TokenId::try_from(token_bytes.len())
            .map_err(|_| anyhow!("vocabulary size exceeded u32::MAX"))?;

        if !allowed_lengths.contains(&combined_len) {
            allowed_lengths.push(combined_len);
        }

        let total_merges = final_apply_merge(
            &mut sequences,
            pair,
            new_token_id,
            combined_len,
            &mut pair_counts,
            &mut heap,
            &token_lengths,
            &allowed_lengths,
        );

        if total_merges == 0 {
            candidates[idx].used = true;
            if let Some(existing) = candidates[idx].pair.take() {
                candidate_lookup.remove(&existing);
            }
            continue;
        }

        token_bytes.push(combined_bytes.clone());
        token_map.insert(combined_bytes, new_token_id);
        token_lengths.push(combined_len);
        merges.push(pair);
        candidates[idx].used = true;
        if let Some(existing) = candidates[idx].pair.take() {
            candidate_lookup.remove(&existing);
        }

        refresh_candidate_pairs(
            &mut candidates,
            &token_map,
            &pair_counts,
            &mut heap,
            &mut candidate_lookup,
            validation_threshold,
        );
    }

    if merges.is_empty() {
        return Ok(None);
    }

    let stats = FinalMergeStats {
        requested: target_merges,
        realised: merges.len(),
        duplicates: 0,
        unresolved: target_merges.saturating_sub(merges.len()),
    };

    Ok(Some(FinalAssembly {
        token_bytes,
        merges,
        stats,
    }))
}

fn refresh_candidate_pairs(
    candidates: &mut [CandidateState],
    token_map: &FxHashMap<Vec<u8>, TokenId>,
    pair_counts: &FxHashMap<Pair, usize>,
    heap: &mut BinaryHeap<FinalPairScore>,
    lookup: &mut FxHashMap<Pair, usize>,
    validation_threshold: usize,
) {
    for (idx, candidate) in candidates.iter_mut().enumerate() {
        if candidate.used {
            continue;
        }
        let left_id = token_map.get(&candidate.key.left).copied();
        let right_id = token_map.get(&candidate.key.right).copied();
        let next_pair = match (left_id, right_id) {
            (Some(left), Some(right)) => Some((left, right)),
            _ => None,
        };
        if next_pair == candidate.pair {
            continue;
        }
        if let Some(existing) = candidate.pair.take() {
            lookup.remove(&existing);
        }
        candidate.pair = next_pair;
        if let Some(pair) = next_pair {
            lookup.insert(pair, idx);
            if let Some(&freq) = pair_counts.get(&pair) {
                if freq > 0 && freq >= validation_threshold {
                    heap.push(FinalPairScore::new(pair, freq));
                }
            }
        }
    }
}

fn final_compute_pair_counts(
    sequences: &[Vec<TokenId>],
    token_lengths: &[usize],
    allowed_lengths: &[usize],
) -> FxHashMap<Pair, usize> {
    sequences
        .par_iter()
        .map(|sequence| {
            let mut local = FxHashMap::default();
            if sequence.len() < 2 {
                return local;
            }
            let mut prev = sequence[0];
            for &current in &sequence[1..] {
                let combined_len = token_lengths[prev as usize] + token_lengths[current as usize];
                if is_allowed_length(combined_len, allowed_lengths) {
                    *local.entry((prev, current)).or_insert(0) += 1;
                }
                prev = current;
            }
            local
        })
        .reduce(FxHashMap::default, |mut acc, local| {
            for (pair, count) in local {
                *acc.entry(pair).or_insert(0) += count;
            }
            acc
        })
}

#[derive(Clone, Copy, Eq, PartialEq)]
struct FinalPairScore {
    frequency: usize,
    pair: Pair,
}

impl FinalPairScore {
    fn new(pair: Pair, frequency: usize) -> Self {
        Self { frequency, pair }
    }
}

impl Ord for FinalPairScore {
    fn cmp(&self, other: &Self) -> Ordering {
        self.frequency
            .cmp(&other.frequency)
            .then_with(|| other.pair.cmp(&self.pair))
    }
}

impl PartialOrd for FinalPairScore {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Default)]
struct FinalMergeAdjustments {
    deltas: FxHashMap<Pair, i64>,
    merges: usize,
}

fn final_accumulate_delta(
    deltas: &mut FxHashMap<Pair, i64>,
    pair: Pair,
    combined_len: usize,
    allowed_lengths: &[usize],
    delta: i64,
) {
    if delta == 0 {
        return;
    }
    if !is_allowed_length(combined_len, allowed_lengths) {
        return;
    }
    *deltas.entry(pair).or_insert(0) += delta;
}

fn final_apply_delta(
    pair_counts: &mut FxHashMap<Pair, usize>,
    heap: &mut BinaryHeap<FinalPairScore>,
    pair: Pair,
    delta: i64,
) {
    if delta == 0 {
        return;
    }
    match delta.cmp(&0) {
        Ordering::Greater => {
            let amount =
                usize::try_from(delta.unsigned_abs()).expect("positive delta must fit in usize");
            let count = pair_counts.entry(pair).or_insert(0);
            *count += amount;
            heap.push(FinalPairScore::new(pair, *count));
        }
        Ordering::Less => {
            let amount =
                usize::try_from(delta.unsigned_abs()).expect("negative delta must fit in usize");
            if let Entry::Occupied(mut occupied) = pair_counts.entry(pair) {
                let current = *occupied.get();
                let new_value = current.saturating_sub(amount);
                if new_value == 0 {
                    occupied.remove();
                } else {
                    *occupied.get_mut() = new_value;
                    heap.push(FinalPairScore::new(pair, new_value));
                }
            }
        }
        Ordering::Equal => {}
    }
}

fn final_token_length_with_new(
    token: TokenId,
    token_lengths: &[usize],
    new_token: TokenId,
    new_token_len: usize,
) -> usize {
    if token == new_token {
        new_token_len
    } else {
        token_lengths[token as usize]
    }
}

fn final_process_sequence(
    sequence: &mut Vec<TokenId>,
    pair: Pair,
    new_token: TokenId,
    new_token_len: usize,
    token_lengths: &[usize],
    allowed_lengths: &[usize],
) -> FinalMergeAdjustments {
    let mut result = FinalMergeAdjustments::default();
    if sequence.len() < 2 {
        return result;
    }

    let mut read = 0usize;
    let mut write = 0usize;
    let original_len = sequence.len();
    let left_len = token_lengths[pair.0 as usize];
    let right_len = token_lengths[pair.1 as usize];

    while read < original_len {
        if read + 1 < original_len && sequence[read] == pair.0 && sequence[read + 1] == pair.1 {
            let prev_token = if write > 0 {
                Some(sequence[write - 1])
            } else {
                None
            };
            let next_token = if read + 2 < original_len {
                Some(sequence[read + 2])
            } else {
                None
            };

            if let Some(prev) = prev_token {
                let prev_len =
                    final_token_length_with_new(prev, token_lengths, new_token, new_token_len);
                let combined = prev_len + left_len;
                final_accumulate_delta(
                    &mut result.deltas,
                    (prev, pair.0),
                    combined,
                    allowed_lengths,
                    -1,
                );
            }
            final_accumulate_delta(
                &mut result.deltas,
                pair,
                left_len + right_len,
                allowed_lengths,
                -1,
            );
            if let Some(next) = next_token {
                let next_len =
                    final_token_length_with_new(next, token_lengths, new_token, new_token_len);
                let combined = right_len + next_len;
                final_accumulate_delta(
                    &mut result.deltas,
                    (pair.1, next),
                    combined,
                    allowed_lengths,
                    -1,
                );
            }

            sequence[write] = new_token;
            write += 1;
            read += 2;
            result.merges += 1;

            if let Some(prev) = prev_token {
                let prev_len =
                    final_token_length_with_new(prev, token_lengths, new_token, new_token_len);
                let combined = prev_len + new_token_len;
                final_accumulate_delta(
                    &mut result.deltas,
                    (prev, new_token),
                    combined,
                    allowed_lengths,
                    1,
                );
            }
            if let Some(next) = next_token {
                let next_len =
                    final_token_length_with_new(next, token_lengths, new_token, new_token_len);
                let combined = new_token_len + next_len;
                final_accumulate_delta(
                    &mut result.deltas,
                    (new_token, next),
                    combined,
                    allowed_lengths,
                    1,
                );
            }
        } else {
            if write != read {
                sequence[write] = sequence[read];
            }
            write += 1;
            read += 1;
        }
    }

    sequence.truncate(write);
    result
}

#[allow(clippy::too_many_arguments)]
fn final_apply_merge(
    sequences: &mut [Vec<TokenId>],
    pair: Pair,
    new_token: TokenId,
    new_token_len: usize,
    pair_counts: &mut FxHashMap<Pair, usize>,
    heap: &mut BinaryHeap<FinalPairScore>,
    token_lengths: &[usize],
    allowed_lengths: &[usize],
) -> usize {
    let aggregate = sequences
        .par_iter_mut()
        .map(|sequence| {
            final_process_sequence(
                sequence,
                pair,
                new_token,
                new_token_len,
                token_lengths,
                allowed_lengths,
            )
        })
        .reduce(FinalMergeAdjustments::default, |mut acc, mut local| {
            acc.merges += local.merges;
            for (pair_key, delta) in local.deltas.drain() {
                *acc.deltas.entry(pair_key).or_insert(0) += delta;
            }
            acc
        });

    for (pair_key, delta) in aggregate.deltas {
        final_apply_delta(pair_counts, heap, pair_key, delta);
    }

    aggregate.merges
}

fn assemble_final_tokenizer(
    ranked: &[(MergeKey, MergeAggregate)],
    limit: usize,
) -> Result<FinalAssembly> {
    let requested = ranked.len().min(limit);
    let mut queue: VecDeque<MergeKey> = ranked
        .iter()
        .take(requested)
        .map(|(key, _)| key.clone())
        .collect();

    if requested > 0 && env::var("BBPE_DEBUG_ASSEMBLY").is_ok() {
        if let Some(head) = queue.front() {
            #[allow(clippy::print_stdout)]
            {
                println!(
                    "assemble_final_tokenizer head merge: left={:?} right={:?}",
                    head.left, head.right
                );
            }
        }
    }

    let mut token_bytes: Vec<Vec<u8>> = (0u8..=u8::MAX).map(|b| vec![b]).collect();
    let mut token_map: FxHashMap<Vec<u8>, TokenId> = FxHashMap::default();
    token_map.reserve(token_bytes.len().saturating_mul(2));
    for (idx, token) in token_bytes.iter().enumerate() {
        token_map.insert(token.clone(), idx as TokenId);
    }

    let mut merges: Vec<Pair> = Vec::with_capacity(requested);
    let mut duplicates = 0usize;
    let mut unresolved = 0usize;

    while !queue.is_empty() && merges.len() < requested {
        let mut progress = false;
        let mut next_queue = VecDeque::with_capacity(queue.len());
        while let Some(candidate) = queue.pop_front() {
            if merges.len() >= requested {
                break;
            }
            let left_id = match token_map.get(&candidate.left) {
                Some(id) => *id,
                None => {
                    next_queue.push_back(candidate);
                    continue;
                }
            };
            let right_id = match token_map.get(&candidate.right) {
                Some(id) => *id,
                None => {
                    next_queue.push_back(candidate);
                    continue;
                }
            };
            let mut combined = candidate.left.clone();
            combined.extend_from_slice(&candidate.right);
            if token_map.contains_key(&combined) {
                duplicates += 1;
                progress = true;
                continue;
            }
            let new_id = u32::try_from(token_bytes.len())
                .map_err(|_| anyhow!("vocabulary size exceeded u32::MAX"))?
                as TokenId;
            token_map.insert(combined.clone(), new_id);
            token_bytes.push(combined);
            merges.push((left_id, right_id));
            progress = true;
        }

        if !progress {
            unresolved += next_queue.len();
            break;
        }
        queue = next_queue;
    }
    unresolved += queue.len();

    let stats = FinalMergeStats {
        requested,
        realised: merges.len(),
        duplicates,
        unresolved,
    };

    Ok(FinalAssembly {
        token_bytes,
        merges,
        stats,
    })
}

fn aggregate_all(data: &[ChunkData], rank_opts: &ChunkRankOptions) -> AggregationResult {
    if data.is_empty() {
        return AggregationResult::default();
    }
    let indices: Vec<usize> = (0..data.len()).collect();
    let aggregate = collect_subset(data, &indices);
    finalize_aggregation(&aggregate, rank_opts)
}

fn aggregate_subset(
    data: &[ChunkData],
    indices: &[usize],
    rank_opts: &ChunkRankOptions,
) -> (
    FxHashMap<MergeKey, MergeAggregateBuilder>,
    AggregationResult,
) {
    let aggregate = collect_subset(data, indices);
    let result = finalize_aggregation(&aggregate, rank_opts);
    (aggregate, result)
}

fn collect_subset(
    data: &[ChunkData],
    indices: &[usize],
) -> FxHashMap<MergeKey, MergeAggregateBuilder> {
    let mut aggregate: FxHashMap<MergeKey, MergeAggregateBuilder> = FxHashMap::default();
    for &idx in indices {
        if let Some(chunk) = data.get(idx) {
            for record in &chunk.records {
                let entry = aggregate
                    .entry(record.key.clone())
                    .or_insert_with(MergeAggregateBuilder::default);
                entry.include(idx, record);
            }
        }
    }
    aggregate
}

fn finalize_aggregation(
    aggregate: &FxHashMap<MergeKey, MergeAggregateBuilder>,
    rank_opts: &ChunkRankOptions,
) -> AggregationResult {
    let mut dropped = 0usize;
    let mut entries = Vec::with_capacity(aggregate.len());
    for (key, builder) in aggregate {
        let agg = builder.as_merge_aggregate();
        if rank_opts.accepts(&agg) {
            entries.push((key.clone(), agg));
        } else {
            dropped += 1;
        }
    }
    entries.sort_by(|a, b| b.1.compare(&a.1, rank_opts));
    AggregationResult {
        total_candidates: aggregate.len(),
        ranked: entries,
        dropped_below_support: dropped,
    }
}

fn accumulate_results(
    combined: &mut FxHashMap<MergeKey, MergeAggregateBuilder>,
    partial: FxHashMap<MergeKey, MergeAggregateBuilder>,
) {
    for (key, aggregate) in partial {
        let entry = combined
            .entry(key)
            .or_insert_with(MergeAggregateBuilder::default);
        entry.extend(aggregate);
    }
}

fn aggregate_boost(
    data: &[ChunkData],
    rank_opts: &ChunkRankOptions,
    global_limit: usize,
    rounds: usize,
    sample_size: usize,
    learning_rate: f64,
    key_to_chunks: &FxHashMap<MergeKey, Vec<usize>>,
    rng: &mut StdRng,
) -> (AggregationResult, Vec<BoostRoundSummary>) {
    let total_chunks = data.len();
    if total_chunks == 0 {
        return (AggregationResult::default(), Vec::new());
    }
    let rounds = rounds.max(1);
    let sample_size = sample_size.clamp(1, total_chunks);
    let mut weights = vec![1.0_f64; total_chunks];
    let mut combined: FxHashMap<MergeKey, MergeAggregateBuilder> = FxHashMap::default();
    let mut diagnostics = Vec::with_capacity(rounds);

    for round in 0..rounds {
        let dist = WeightedIndex::new(&weights).expect("boost weights must remain positive");
        let mut indices = Vec::with_capacity(sample_size);
        for _ in 0..sample_size {
            indices.push(dist.sample(rng));
        }

        let (round_aggregate, round_result) = aggregate_subset(data, &indices, rank_opts);
        accumulate_results(&mut combined, round_aggregate);

        let coverage = compute_coverage(data, &round_result.ranked, global_limit, key_to_chunks);
        let (coverage_mean, coverage_min, coverage_max) = coverage_stats(&coverage);
        let (weight_min, weight_max) = weight_stats(&weights);
        diagnostics.push(BoostRoundSummary {
            round: round + 1,
            sample_size: indices.len(),
            coverage_mean,
            coverage_min,
            coverage_max,
            weight_min,
            weight_max,
        });

        for (idx, cov) in coverage.into_iter().enumerate() {
            let residual = (1.0 - cov).clamp(0.0, 1.0);
            let factor = 1.0 + learning_rate.max(0.0) * residual;
            weights[idx] = (weights[idx] * factor).clamp(1e-6, 1e6);
        }
    }

    (finalize_aggregation(&combined, rank_opts), diagnostics)
}

fn aggregate_sampled(
    data: &[ChunkData],
    rank_opts: &ChunkRankOptions,
    global_limit: usize,
    rounds: usize,
    sample_size: usize,
    key_to_chunks: &FxHashMap<MergeKey, Vec<usize>>,
    rng: &mut StdRng,
) -> (AggregationResult, Vec<SampleRoundSummary>) {
    let total_chunks = data.len();
    if total_chunks == 0 {
        return (AggregationResult::default(), Vec::new());
    }
    let rounds = rounds.max(1);
    let sample_size = sample_size.clamp(1, total_chunks);
    let mut combined: FxHashMap<MergeKey, MergeAggregateBuilder> = FxHashMap::default();
    let mut diagnostics = Vec::with_capacity(rounds);

    for round in 0..rounds {
        let indices = sample(rng, total_chunks, sample_size).into_vec();
        let (round_aggregate, round_result) = aggregate_subset(data, &indices, rank_opts);
        accumulate_results(&mut combined, round_aggregate);

        let coverage = compute_coverage(data, &round_result.ranked, global_limit, key_to_chunks);
        let (coverage_mean, coverage_min, coverage_max) = coverage_stats(&coverage);
        diagnostics.push(SampleRoundSummary {
            round: round + 1,
            sample_size: indices.len(),
            coverage_mean,
            coverage_min,
            coverage_max,
        });
    }

    (finalize_aggregation(&combined, rank_opts), diagnostics)
}

fn compute_coverage(
    data: &[ChunkData],
    ranked: &[(MergeKey, MergeAggregate)],
    top_k: usize,
    key_to_chunks: &FxHashMap<MergeKey, Vec<usize>>,
) -> Vec<f64> {
    let mut coverage = vec![0usize; data.len()];
    let limit = top_k.min(ranked.len());
    for (key, _) in ranked.iter().take(limit) {
        if let Some(indices) = key_to_chunks.get(key) {
            for &idx in indices {
                if idx < coverage.len() {
                    coverage[idx] = coverage[idx].saturating_add(1);
                }
            }
        }
    }
    coverage
        .into_iter()
        .enumerate()
        .map(|(idx, hits)| {
            let denom = data[idx].records.len().max(1);
            hits as f64 / denom as f64
        })
        .collect()
}

fn coverage_stats(values: &[f64]) -> (f64, f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mut min_v = f64::INFINITY;
    let mut max_v = f64::NEG_INFINITY;
    let mut sum = 0.0;
    for &value in values {
        sum += value;
        if value < min_v {
            min_v = value;
        }
        if value > max_v {
            max_v = value;
        }
    }
    (sum / values.len() as f64, min_v, max_v)
}

fn weight_stats(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let mut min_v = f64::INFINITY;
    let mut max_v = f64::NEG_INFINITY;
    for &value in values {
        if value < min_v {
            min_v = value;
        }
        if value > max_v {
            max_v = value;
        }
    }
    (min_v, max_v)
}

fn write_chunk_report(path: &Path, report: &ChunkedReport) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
    }
    let mut file =
        File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    serde_json::to_writer_pretty(&mut file, report)
        .with_context(|| format!("failed to serialise {}", path.display()))?;
    file.flush()
        .with_context(|| format!("failed to flush {}", path.display()))?;
    Ok(())
}

fn split_chunk_for_validation(
    chunk: Vec<u8>,
    fraction: f64,
    rng: &mut StdRng,
) -> (Vec<u8>, Option<Vec<u8>>) {
    if chunk.is_empty() || fraction <= f64::EPSILON {
        return (chunk, None);
    }
    let chunk_len = chunk.len();
    if chunk_len < 2 {
        return (chunk, None);
    }
    let mut eval_len = ((chunk_len as f64) * fraction).round() as usize;
    if eval_len == 0 {
        return (chunk, None);
    }
    if eval_len >= chunk_len {
        eval_len = chunk_len.saturating_sub(1);
        if eval_len == 0 {
            return (chunk, None);
        }
    }
    let max_start = chunk_len - eval_len;
    let start = if max_start == 0 {
        0
    } else {
        rng.gen_range(0..=max_start)
    };

    let mut training = Vec::with_capacity(chunk_len - eval_len);
    training.extend_from_slice(&chunk[..start]);
    training.extend_from_slice(&chunk[start + eval_len..]);
    if training.is_empty() {
        return (chunk, None);
    }

    let validation = chunk[start..start + eval_len].to_vec();
    (training, Some(validation))
}

#[must_use]
fn defaults_chunk_size() -> usize {
    IngestConfig::default().chunk_size
}

#[must_use]
fn bytes_to_mebibytes(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn write_token_sequence<W: Write>(writer: &mut W, tokens: &[u32]) -> Result<()> {
    for (idx, token) in tokens.iter().enumerate() {
        if idx > 0 {
            writer.write_all(b" ")?;
        }
        write!(writer, "{token}")?;
    }
    writer.write_all(b"\n")?;
    Ok(())
}

fn parse_token_list(text: &str) -> Result<Vec<u32>> {
    text.split_whitespace()
        .map(|part| {
            part.parse::<u32>()
                .map_err(|err| anyhow!("invalid token id `{part}`: {err}"))
        })
        .collect()
}
