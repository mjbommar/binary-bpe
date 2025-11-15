use std::collections::hash_map::Entry;
use std::convert::TryFrom;
use std::fmt;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use bbpe::bytes::{bytes_to_latin1, latin1_to_bytes};
use bbpe::config::{IngestConfig, PreprocessorConfig, PreprocessorKind, TrainerConfig};
use bbpe::corpus::{load_binary_corpus, load_jsonl_corpus, stream_binary_corpus, JsonlSpec};
use bbpe::model::{Pair, TokenId};
use bbpe::serialization;
use bbpe::{BinaryTokenizer, BpeModel, Trainer, TrainerArtifacts};
use clap::{ArgAction, Args, Parser, Subcommand, ValueEnum};
use env_logger::Env;
use indicatif::{ProgressBar, ProgressStyle};
use log::info;
use rayon::iter::ParallelBridge;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use rustc_hash::FxHashMap;
use serde::ser::SerializeStruct;
use serde::{Deserialize, Serialize};
use serde_json::{self, json};

const DEFAULT_OUTPUT: &str = "tokenizer.json";
const DEFAULT_MIN_ENTROPY: f64 = 0.20;
const DEFAULT_MAX_ENTROPY: f64 = 7.80;

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
    /// Train per-chunk tokenizers and combine them
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
    /// Files or directories to ingest (optional when using --jsonl)
    #[arg(required = false)]
    inputs: Vec<PathBuf>,

    /// JSONL inputs specified as `PATH:field.path` (repeat flag)
    #[arg(long = "jsonl", value_name = "PATH:FIELD")]
    jsonl_inputs: Vec<JsonlSpec>,

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

    /// Preprocessor applied before merge counting
    #[arg(
        long = "preprocessor",
        value_enum,
        default_value_t = PreprocessorCli::None
    )]
    preprocessor: PreprocessorCli,
    /// Probability that each detected delimiter boundary is preserved (0.0-1.0)
    #[arg(
        long = "preprocessor-probability",
        value_name = "P",
        default_value_t = 1.0
    )]
    preprocessor_probability: f64,
    /// Optional RNG seed controlling probabilistic preprocessing
    #[arg(long = "preprocessor-seed", value_name = "SEED")]
    preprocessor_seed: Option<u64>,

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

    /// Minimum entropy threshold in bits/byte for filtering chunks (e.g., 0.20)
    #[arg(long, value_name = "BITS")]
    min_entropy: Option<f64>,

    /// Maximum entropy threshold in bits/byte for filtering chunks (e.g., 7.0)
    #[arg(long, value_name = "BITS")]
    max_entropy: Option<f64>,

    /// Additional vocabulary sizes to derive within the tokenizer family
    #[arg(long = "family-size", value_name = "SIZE")]
    family_sizes: Vec<usize>,

    /// Output template for derived family members (supports `{size}`)
    #[arg(long = "family-template", value_name = "PATTERN")]
    family_template: Option<String>,
}

#[derive(Args, Debug)]
struct ChunkTrainArgs {
    /// Files or directories to ingest
    #[arg(required = true)]
    inputs: Vec<PathBuf>,

    /// Output path for the combined tokenizer
    #[arg(long, value_name = "PATH", default_value = "chunked-tokenizer.json")]
    output: PathBuf,

    /// Output path for the chunk training report
    #[arg(long, value_name = "PATH", default_value = "chunk_train_report.json")]
    report: PathBuf,

    /// Disable writing the chunk training report JSON
    #[arg(long)]
    no_report: bool,

    /// Target chunk size in bytes
    #[arg(
        long = "chunk-size",
        value_name = "BYTES",
        default_value_t = 33_554_432
    )]
    chunk_size: usize,

    /// Target vocabulary size
    #[arg(long, value_name = "SIZE")]
    vocab_size: Option<usize>,

    /// Minimum frequency for merges
    #[arg(long, value_name = "COUNT")]
    min_frequency: Option<usize>,

    /// Allowed token lengths in bytes (repeat flag)
    #[arg(long = "allowed-length", value_name = "LEN")]
    allowed_lengths: Vec<usize>,

    /// Append additional special tokens
    #[arg(long = "special-token", value_name = "TOKEN")]
    special_tokens: Vec<String>,

    /// Preprocessor applied before merge counting
    #[arg(
        long = "preprocessor",
        value_enum,
        default_value_t = PreprocessorCli::None
    )]
    preprocessor: PreprocessorCli,
    /// Probability that each detected delimiter boundary is preserved (0.0-1.0)
    #[arg(
        long = "preprocessor-probability",
        value_name = "P",
        default_value_t = 1.0
    )]
    preprocessor_probability: f64,
    /// Optional RNG seed controlling probabilistic preprocessing
    #[arg(long = "preprocessor-seed", value_name = "SEED")]
    preprocessor_seed: Option<u64>,

    /// Maximum merge iterations
    #[arg(long, value_name = "COUNT")]
    max_merge_iterations: Option<usize>,

    /// Minimum entropy threshold in bits/byte before training a chunk (default 0.10)
    #[arg(long, value_name = "BITS")]
    min_entropy: Option<f64>,

    /// Maximum entropy threshold in bits/byte before training a chunk (default 7.95)
    #[arg(long, value_name = "BITS")]
    max_entropy: Option<f64>,

    /// Limit Rayon worker threads
    #[arg(long, value_name = "N")]
    threads: Option<usize>,

    /// Disable recursive directory traversal
    #[arg(long)]
    no_recursive: bool,

    /// Follow symlinks during traversal
    #[arg(long)]
    follow_symlinks: bool,

    /// Disable per-chunk progress reporting
    #[arg(long)]
    no_progress: bool,

    /// Combination strategy used to assemble the final vocabulary
    #[arg(long = "combine-mode", value_enum, default_value_t = ChunkCombineMode::Support)]
    combine_mode: ChunkCombineMode,

    /// Handling mode for duplicate chunk contents
    #[arg(long = "duplicates", value_enum, default_value_t = ChunkDuplicateMode::Count)]
    duplicate_mode: ChunkDuplicateMode,

    /// Additional vocabulary sizes to derive within the tokenizer family
    #[arg(long = "family-size", value_name = "SIZE")]
    family_sizes: Vec<usize>,

    /// Output template for derived family members (supports `{size}`)
    #[arg(long = "family-template", value_name = "PATTERN")]
    family_template: Option<String>,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum ChunkCombineMode {
    /// Reuse the first trained chunk without modification.
    First,
    /// Aggregate merges by summed frequency across chunks.
    Frequency,
    /// Prioritise merges that appear in the most chunks.
    Support,
    /// Weight per-chunk merges by entropy-aware scores.
    Entropy,
}

impl ChunkCombineMode {
    fn build(self) -> Box<dyn ChunkCombiner> {
        match self {
            Self::First => Box::new(FirstChunkCombiner),
            Self::Frequency => Box::new(FrequencyUnionCombiner),
            Self::Support => Box::new(SupportUnionCombiner),
            Self::Entropy => Box::new(EntropyFrequencyCombiner),
        }
    }
}

impl fmt::Display for ChunkCombineMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::First => "first",
            Self::Frequency => "frequency",
            Self::Support => "support",
            Self::Entropy => "entropy",
        };
        f.write_str(label)
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum ChunkDuplicateMode {
    /// Count duplicate chunks as independent samples during combination.
    Count,
    /// Only include a single representative for duplicate chunks.
    Unique,
}

impl ChunkDuplicateMode {
    fn counts_duplicates(self) -> bool {
        matches!(self, Self::Count)
    }
}

impl fmt::Display for ChunkDuplicateMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::Count => "count",
            Self::Unique => "unique",
        };
        f.write_str(label)
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum PreprocessorCli {
    /// Disable preprocessing and operate on raw byte sequences.
    None,
    /// Split on ASCII whitespace runs.
    AsciiWhitespace,
    /// Split on Unicode whitespace runs.
    UnicodeWhitespace,
    /// Split on contiguous null-byte runs, useful for binary corpora.
    NullDelimited,
}

impl From<PreprocessorCli> for PreprocessorConfig {
    fn from(value: PreprocessorCli) -> Self {
        let kind = match value {
            PreprocessorCli::None => PreprocessorKind::None,
            PreprocessorCli::AsciiWhitespace => PreprocessorKind::AsciiWhitespace,
            PreprocessorCli::UnicodeWhitespace => PreprocessorKind::UnicodeWhitespace,
            PreprocessorCli::NullDelimited => PreprocessorKind::NullDelimited,
        };
        PreprocessorConfig {
            kind,
            split_probability: 1.0,
            seed: None,
        }
    }
}

impl PreprocessorCli {
    fn into_config(self, probability: f64, seed: Option<u64>) -> PreprocessorConfig {
        let mut cfg: PreprocessorConfig = self.into();
        cfg.split_probability = probability;
        cfg.seed = seed;
        cfg
    }
}

trait ChunkCombiner {
    fn name(&self) -> &'static str;
    fn combine(
        &self,
        chunks: &[ChunkTrainingSnapshot],
        summaries: &[ChunkSummary],
        trainer_cfg: &TrainerConfig,
    ) -> Result<CombinationOutput>;
}

struct CombinationOutput {
    model: BpeModel,
    detail: String,
    stats: CombineStats,
}

struct FirstChunkCombiner;

impl ChunkCombiner for FirstChunkCombiner {
    fn name(&self) -> &'static str {
        "first"
    }

    fn combine(
        &self,
        chunks: &[ChunkTrainingSnapshot],
        _summaries: &[ChunkSummary],
        trainer_cfg: &TrainerConfig,
    ) -> Result<CombinationOutput> {
        let first = chunks
            .first()
            .ok_or_else(|| anyhow!("no chunks available for combination"))?;
        let requested = target_merge_budget(trainer_cfg);
        let realised = first.model.merges().len();
        let detail = format!(
            "reused chunk {} vocabulary ({} merges, {} bytes)",
            first.index,
            first.model.merges().len(),
            first.byte_len
        );
        Ok(CombinationOutput {
            model: first.model.as_ref().clone(),
            detail,
            stats: CombineStats {
                merges_requested: requested,
                merges_realized: realised,
                duplicates_skipped: 0,
                missing_dependencies: 0,
            },
        })
    }
}

struct FrequencyUnionCombiner;

struct SupportUnionCombiner;

struct EntropyFrequencyCombiner;

impl ChunkCombiner for FrequencyUnionCombiner {
    fn name(&self) -> &'static str {
        "frequency"
    }

    fn combine(
        &self,
        _chunks: &[ChunkTrainingSnapshot],
        summaries: &[ChunkSummary],
        trainer_cfg: &TrainerConfig,
    ) -> Result<CombinationOutput> {
        let mut aggregated = aggregate_chunk_merges(summaries);
        aggregated.sort_by(|a, b| {
            b.total_frequency
                .cmp(&a.total_frequency)
                .then_with(|| b.chunk_support.cmp(&a.chunk_support))
                .then_with(|| a.combined_len().cmp(&b.combined_len()))
        });
        let (model, stats) = assemble_aggregated_merges(&aggregated, trainer_cfg)?;
        let detail = format!(
            "frequency union realised {}/{} merges (duplicates {}, missing {})",
            stats.merges_realized,
            stats.merges_requested,
            stats.duplicates_skipped,
            stats.missing_dependencies
        );
        Ok(CombinationOutput {
            model,
            detail,
            stats,
        })
    }
}

impl ChunkCombiner for SupportUnionCombiner {
    fn name(&self) -> &'static str {
        "support"
    }

    fn combine(
        &self,
        _chunks: &[ChunkTrainingSnapshot],
        summaries: &[ChunkSummary],
        trainer_cfg: &TrainerConfig,
    ) -> Result<CombinationOutput> {
        let mut aggregated = aggregate_chunk_merges(summaries);
        aggregated.sort_by(|a, b| {
            b.chunk_support
                .cmp(&a.chunk_support)
                .then_with(|| b.total_frequency.cmp(&a.total_frequency))
                .then_with(|| a.combined_len().cmp(&b.combined_len()))
        });
        let (model, stats) = assemble_aggregated_merges(&aggregated, trainer_cfg)?;
        let detail = format!(
            "support union realised {}/{} merges (duplicates {}, missing {})",
            stats.merges_realized,
            stats.merges_requested,
            stats.duplicates_skipped,
            stats.missing_dependencies
        );
        Ok(CombinationOutput {
            model,
            detail,
            stats,
        })
    }
}

impl ChunkCombiner for EntropyFrequencyCombiner {
    fn name(&self) -> &'static str {
        "entropy"
    }

    fn combine(
        &self,
        _chunks: &[ChunkTrainingSnapshot],
        summaries: &[ChunkSummary],
        trainer_cfg: &TrainerConfig,
    ) -> Result<CombinationOutput> {
        let weights = compute_chunk_weights(summaries);
        let mut aggregated = aggregate_chunk_merges_weighted(summaries, &weights);
        let min_support = if summaries.len() > 1 { 2 } else { 1 };
        aggregated.retain(|merge| merge.chunk_support >= min_support);
        aggregated.sort_by(|a, b| {
            b.weighted_frequency
                .partial_cmp(&a.weighted_frequency)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.chunk_support.cmp(&a.chunk_support))
                .then_with(|| b.total_frequency.cmp(&a.total_frequency))
                .then_with(|| a.combined_len().cmp(&b.combined_len()))
        });
        let (model, stats) = assemble_aggregated_merges(&aggregated, trainer_cfg)?;
        let (weight_min, weight_max) = if weights.is_empty() {
            (0.0, 0.0)
        } else {
            (
                weights.iter().fold(f64::INFINITY, |acc, &w| acc.min(w)),
                weights.iter().fold(f64::NEG_INFINITY, |acc, &w| acc.max(w)),
            )
        };
        let detail = format!(
            "entropy-weighted union realised {}/{} merges (duplicates {}, missing {}, weight range {:.3}-{:.3})",
            stats.merges_realized,
            stats.merges_requested,
            stats.duplicates_skipped,
            stats.missing_dependencies,
            weight_min,
            weight_max
        );
        Ok(CombinationOutput {
            model,
            detail,
            stats,
        })
    }
}

#[derive(Clone)]
struct ChunkTrainingSnapshot {
    index: usize,
    byte_len: usize,
    model: Arc<BpeModel>,
}

#[derive(Debug, Serialize, Clone)]
struct ChunkMergeRecord {
    iteration: usize,
    left: String,
    right: String,
    best_frequency: usize,
    merges_applied: usize,
}

#[derive(Clone)]
struct ChunkSummary {
    index: usize,
    byte_len: usize,
    entropy_bits: f64,
    vocab_size: usize,
    merge_count: usize,
    merges: Arc<Vec<ChunkMergeRecord>>,
}

impl Serialize for ChunkSummary {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut state = serializer.serialize_struct("ChunkSummary", 6)?;
        state.serialize_field("index", &self.index)?;
        state.serialize_field("byte_len", &self.byte_len)?;
        state.serialize_field("entropy_bits", &self.entropy_bits)?;
        state.serialize_field("vocab_size", &self.vocab_size)?;
        state.serialize_field("merge_count", &self.merge_count)?;
        state.serialize_field("merges", &*self.merges)?;
        state.end()
    }
}

#[derive(Clone, Serialize)]
struct CombineStats {
    merges_requested: usize,
    merges_realized: usize,
    duplicates_skipped: usize,
    missing_dependencies: usize,
}

#[derive(Serialize)]
struct ChunkTrainReport {
    chunk_size_bytes: usize,
    trainer_vocab_size: usize,
    trainer_min_frequency: usize,
    trainer_min_entropy: Option<f64>,
    trainer_max_entropy: Option<f64>,
    total_bytes: usize,
    processed_chunks: usize,
    total_chunks: usize,
    duplicate_mode: String,
    duplicate_chunks_reused: usize,
    duplicate_chunks_skipped: usize,
    skipped_low_entropy: usize,
    skipped_high_entropy: usize,
    combine_mode: String,
    combine_detail: String,
    combine_stats: CombineStats,
    final_vocab_size: usize,
    chunks: Vec<ChunkSummary>,
}

#[derive(Hash, Eq, PartialEq)]
struct MergeBytesKey {
    left: Vec<u8>,
    right: Vec<u8>,
}

#[derive(Default)]
struct AggregatedMergeStats {
    total_frequency: usize,
    weighted_frequency: f64,
    chunk_support: usize,
}

struct AggregatedMerge {
    left: Vec<u8>,
    right: Vec<u8>,
    total_frequency: usize,
    weighted_frequency: f64,
    chunk_support: usize,
}

impl AggregatedMerge {
    fn combined_len(&self) -> usize {
        self.left.len() + self.right.len()
    }
}

fn aggregate_chunk_merges(summaries: &[ChunkSummary]) -> Vec<AggregatedMerge> {
    let mut map: FxHashMap<MergeBytesKey, AggregatedMergeStats> = FxHashMap::default();
    for summary in summaries {
        for merge in summary.merges.iter() {
            let key = MergeBytesKey {
                left: latin1_to_bytes(&merge.left),
                right: latin1_to_bytes(&merge.right),
            };
            let entry = map.entry(key).or_default();
            entry.total_frequency = entry.total_frequency.saturating_add(merge.best_frequency);
            entry.weighted_frequency += merge.best_frequency as f64;
            entry.chunk_support = entry.chunk_support.saturating_add(1);
        }
    }
    map.into_iter()
        .map(|(key, stats)| AggregatedMerge {
            left: key.left,
            right: key.right,
            total_frequency: stats.total_frequency,
            weighted_frequency: stats.weighted_frequency,
            chunk_support: stats.chunk_support,
        })
        .collect()
}

fn aggregate_chunk_merges_weighted(
    summaries: &[ChunkSummary],
    weights: &[f64],
) -> Vec<AggregatedMerge> {
    let mut map: FxHashMap<MergeBytesKey, AggregatedMergeStats> = FxHashMap::default();
    let fallback_weight = if summaries.is_empty() {
        0.0
    } else {
        1.0 / summaries.len() as f64
    };
    for summary in summaries {
        let weight = weights
            .get(summary.index)
            .copied()
            .unwrap_or(fallback_weight);
        for merge in summary.merges.iter() {
            let key = MergeBytesKey {
                left: latin1_to_bytes(&merge.left),
                right: latin1_to_bytes(&merge.right),
            };
            let entry = map.entry(key).or_default();
            entry.total_frequency = entry.total_frequency.saturating_add(merge.best_frequency);
            entry.weighted_frequency += (merge.best_frequency as f64) * weight;
            entry.chunk_support = entry.chunk_support.saturating_add(1);
        }
    }
    map.into_iter()
        .map(|(key, stats)| AggregatedMerge {
            left: key.left,
            right: key.right,
            total_frequency: stats.total_frequency,
            weighted_frequency: stats.weighted_frequency,
            chunk_support: stats.chunk_support,
        })
        .collect()
}

fn assemble_aggregated_merges(
    candidates: &[AggregatedMerge],
    trainer_cfg: &TrainerConfig,
) -> Result<(BpeModel, CombineStats)> {
    let requested = target_merge_budget(trainer_cfg);
    let mut token_bytes: Vec<Vec<u8>> = (0u8..=u8::MAX).map(|b| vec![b]).collect();
    let mut token_map: FxHashMap<Vec<u8>, TokenId> = FxHashMap::default();
    for (idx, token) in token_bytes.iter().enumerate() {
        token_map.insert(token.clone(), idx as TokenId);
    }

    let mut merges: Vec<Pair> = Vec::with_capacity(requested);
    let mut duplicates = 0usize;
    let mut missing = 0usize;

    let mut attempts = vec![0usize; candidates.len()];
    let mut queue: std::collections::VecDeque<usize> = (0..candidates.len()).collect();

    while !queue.is_empty() && merges.len() < requested {
        let mut progress = false;
        let pass_len = queue.len();
        for _ in 0..pass_len {
            if merges.len() >= requested {
                break;
            }
            let idx = queue.pop_front().expect("queue not empty");
            attempts[idx] = attempts[idx].saturating_add(1);
            let candidate = &candidates[idx];

            let left_id = match token_map.get(&candidate.left) {
                Some(id) => *id,
                None => {
                    if attempts[idx] <= requested {
                        queue.push_back(idx);
                    } else {
                        missing = missing.saturating_add(1);
                    }
                    continue;
                }
            };
            let right_id = match token_map.get(&candidate.right) {
                Some(id) => *id,
                None => {
                    if attempts[idx] <= requested {
                        queue.push_back(idx);
                    } else {
                        missing = missing.saturating_add(1);
                    }
                    continue;
                }
            };

            let mut combined = candidate.left.clone();
            combined.extend_from_slice(&candidate.right);
            if token_map.contains_key(&combined) {
                duplicates = duplicates.saturating_add(1);
                continue;
            }

            let new_id = u32::try_from(token_bytes.len())
                .map_err(|_| anyhow!("vocabulary size exceeded u32::MAX"))?
                as TokenId;
            token_bytes.push(combined.clone());
            token_map.insert(combined, new_id);
            merges.push((left_id, right_id));
            progress = true;
        }

        if !progress {
            // Remaining candidates are unresolved; count as missing dependencies.
            missing = missing.saturating_add(queue.len());
            break;
        }
    }

    let stats = CombineStats {
        merges_requested: requested,
        merges_realized: merges.len(),
        duplicates_skipped: duplicates,
        missing_dependencies: missing,
    };
    let model = BpeModel::new(token_bytes, merges, trainer_cfg.clone());
    Ok((model, stats))
}

fn target_merge_budget(cfg: &TrainerConfig) -> usize {
    let base_vocab = 256usize + cfg.special_tokens.len();
    cfg.target_vocab_size.saturating_sub(base_vocab)
}

fn compute_entropy_bits(data: &[u8]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut counts = [0usize; 256];
    for &byte in data {
        counts[byte as usize] = counts[byte as usize].saturating_add(1);
    }
    let total = data.len() as f64;
    counts
        .iter()
        .filter(|&&count| count > 0)
        .map(|&count| {
            let p = count as f64 / total;
            -p * p.log2()
        })
        .sum()
}

fn compute_chunk_weights(summaries: &[ChunkSummary]) -> Vec<f64> {
    if summaries.is_empty() {
        return Vec::new();
    }
    let avg_len = summaries
        .iter()
        .map(|summary| summary.byte_len as f64)
        .sum::<f64>()
        / summaries.len() as f64;
    let mut raw_weights = Vec::with_capacity(summaries.len());
    for summary in summaries {
        let entropy_norm = (summary.entropy_bits / 8.0).clamp(0.1, 1.0);
        let length_ratio = if avg_len > 0.0 {
            (summary.byte_len as f64 / avg_len).sqrt()
        } else {
            1.0
        };
        let length_adjusted = length_ratio.clamp(0.5, 2.0);
        raw_weights.push(entropy_norm * length_adjusted);
    }
    let total_weight = raw_weights.iter().sum::<f64>();
    if total_weight == 0.0 {
        let uniform = 1.0 / summaries.len() as f64;
        return vec![uniform; summaries.len()];
    }
    raw_weights
        .into_iter()
        .map(|weight| weight / total_weight)
        .collect()
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

    if args.inputs.is_empty() && args.jsonl_inputs.is_empty() {
        return Err(anyhow!(
            "provide at least one filesystem input or --jsonl <PATH:FIELD>"
        ));
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
    cfg = cfg.preprocessor(
        args.preprocessor
            .into_config(args.preprocessor_probability, args.preprocessor_seed),
    );
    let trainer_cfg = cfg.build()?;

    let ingest_cfg = IngestConfig {
        chunk_size: args.chunk_size.unwrap_or(defaults_chunk_size()),
        recursive: !args.no_recursive,
        follow_symlinks: args.follow_symlinks,
    };

    let mut sequences: Vec<Vec<u8>> = Vec::new();

    if !args.inputs.is_empty() {
        let mut binary_sequences = load_binary_corpus(&args.inputs, &ingest_cfg)
            .with_context(|| "failed to load binary corpus")?;
        let corpus_bytes: usize = binary_sequences.iter().map(|seq| seq.len()).sum();
        info!(
            "loaded {} filesystem sequences totalling {:.2} MiB",
            binary_sequences.len(),
            bytes_to_mebibytes(corpus_bytes)
        );
        sequences.append(&mut binary_sequences);
    }

    if !args.jsonl_inputs.is_empty() {
        let mut jsonl_sequences =
            load_jsonl_corpus(&args.jsonl_inputs).with_context(|| "failed to load JSONL corpus")?;
        let jsonl_bytes: usize = jsonl_sequences.iter().map(|seq| seq.len()).sum();
        info!(
            "loaded {} sequences from {} JSONL file(s) totalling {:.2} MiB",
            jsonl_sequences.len(),
            args.jsonl_inputs.len(),
            bytes_to_mebibytes(jsonl_bytes)
        );
        sequences.append(&mut jsonl_sequences);
    }

    if sequences.is_empty() {
        return Err(anyhow!(
            "no training data loaded; check file paths or JSONL specifications"
        ));
    }

    let corpus_bytes: usize = sequences.iter().map(|seq| seq.len()).sum();
    info!(
        "combined corpus: {} sequences ({:.2} MiB)",
        sequences.len(),
        bytes_to_mebibytes(corpus_bytes)
    );

    // Apply entropy filtering if thresholds are specified
    if args.min_entropy.is_some() || args.max_entropy.is_some() {
        let min_entropy = args.min_entropy.unwrap_or(0.0).max(0.0);
        let max_entropy = args.max_entropy.unwrap_or(8.0).min(8.0);

        if min_entropy > max_entropy {
            return Err(anyhow!(
                "min-entropy threshold ({min_entropy}) cannot exceed max-entropy ({max_entropy})"
            ));
        }

        let original_count = sequences.len();
        let mut filtered_low = 0usize;
        let mut filtered_high = 0usize;

        sequences.retain(|seq| {
            let entropy = compute_entropy_bits(seq);
            let below_min = entropy < min_entropy;
            let above_max = entropy > max_entropy;

            if below_min {
                filtered_low += 1;
            }
            if above_max {
                filtered_high += 1;
            }

            !below_min && !above_max
        });

        let filtered_bytes: usize = sequences.iter().map(|seq| seq.len()).sum();

        info!(
            "entropy filtering: kept {} / {} sequences ({:.2} MiB), filtered {} low / {} high",
            sequences.len(),
            original_count,
            bytes_to_mebibytes(filtered_bytes),
            filtered_low,
            filtered_high
        );

        if sequences.is_empty() {
            return Err(anyhow!(
                "all sequences were filtered out by entropy thresholds (min: {min_entropy}, max: {max_entropy})"
            ));
        }
    }

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

    let trainer = Arc::new(Trainer::new(trainer_cfg.clone()));
    let start = Instant::now();
    let artifacts = trainer.train_from_sequences(sequences)?;
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

    emit_family_models(
        &artifacts.model,
        &args.family_sizes,
        args.family_template.as_deref(),
        &args.output,
        args.pretty,
    )?;

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

    let min_entropy = args.min_entropy.unwrap_or(DEFAULT_MIN_ENTROPY).max(0.0);
    let max_entropy = args.max_entropy.unwrap_or(DEFAULT_MAX_ENTROPY).min(8.0);

    if min_entropy > max_entropy {
        return Err(anyhow!(
            "min-entropy threshold ({min_entropy}) cannot exceed max-entropy ({max_entropy})"
        ));
    }

    if let Some(threads) = args.threads {
        ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .context("unable to configure Rayon thread pool")?;
    }

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
    cfg = cfg
        .preprocessor(
            args.preprocessor
                .into_config(args.preprocessor_probability, args.preprocessor_seed),
        )
        .max_merge_iterations(args.max_merge_iterations)
        .show_progress(false)
        .plateau_stop_enabled(false);
    let trainer_cfg = cfg.build()?;
    let trainer = Trainer::new(trainer_cfg.clone());

    let ingest_cfg = IngestConfig {
        chunk_size: args.chunk_size,
        recursive: !args.no_recursive,
        follow_symlinks: args.follow_symlinks,
    };

    let mut chunk_summaries: Vec<ChunkSummary> = Vec::new();
    let mut snapshots: Vec<ChunkTrainingSnapshot> = Vec::new();

    let chunk_stream = stream_binary_corpus(&args.inputs, &ingest_cfg)
        .with_context(|| "failed to initialise corpus stream")?;
    let total_chunks = chunk_stream.total_chunks();
    let total_bytes = chunk_stream.total_bytes();
    info!(
        "discovered {} chunk candidates totalling {:.2} MiB",
        total_chunks,
        bytes_to_mebibytes(total_bytes)
    );

    let progress = if args.no_progress || total_chunks == 0 {
        None
    } else {
        let pb = ProgressBar::new(total_chunks as u64);
        let style = ProgressStyle::with_template(
            "{spinner:.cyan} chunk {pos}/{len} [{elapsed_precise}] {wide_msg}",
        )
        .unwrap()
        .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏");
        pb.set_style(style);
        pb.enable_steady_tick(Duration::from_millis(80));
        Some(pb)
    };

    let progress_for_workers = progress.clone();

    let min_entropy = args.min_entropy;
    let max_entropy = args.max_entropy;

    let skipped_low_entropy_counter = Arc::new(AtomicUsize::new(0));
    let skipped_high_entropy_counter = Arc::new(AtomicUsize::new(0));

    #[derive(Debug)]
    struct ChunkResultCore {
        byte_len: usize,
        entropy_bits: f64,
        vocab_size: usize,
        merge_count: usize,
        merges: Arc<Vec<ChunkMergeRecord>>,
        model: Arc<BpeModel>,
    }

    #[derive(Debug)]
    struct ChunkMeta {
        first_index: usize,
        skipped: bool,
    }

    struct CacheEntry {
        first_index: usize,
        handle: Arc<OnceLock<Option<Arc<ChunkResultCore>>>>,
    }

    let cache = Arc::new(Mutex::new(FxHashMap::<blake3::Hash, CacheEntry>::default()));
    let result_slots = Arc::new(
        (0..total_chunks)
            .map(|_| OnceLock::<Option<Arc<ChunkResultCore>>>::new())
            .collect::<Vec<_>>(),
    );
    let meta_slots = Arc::new(
        (0..total_chunks)
            .map(|_| OnceLock::<ChunkMeta>::new())
            .collect::<Vec<_>>(),
    );

    let cache_ref = Arc::clone(&cache);
    let result_slots_ref = Arc::clone(&result_slots);
    let meta_slots_ref = Arc::clone(&meta_slots);

    let skipped_low_entropy_ref = Arc::clone(&skipped_low_entropy_counter);
    let skipped_high_entropy_ref = Arc::clone(&skipped_high_entropy_counter);
    chunk_stream
        .enumerate()
        .par_bridge()
        .try_for_each(move |(index, chunk_res)| -> Result<()> {
            let chunk = chunk_res
                .with_context(|| format!("failed to read chunk {index}"))?;
            let byte_len = chunk.len();
            let entropy_bits = compute_entropy_bits(&chunk);
            let chunk_hash = blake3::hash(&chunk);

            let (first_index, handle) = {
                let mut guard = cache_ref.lock().expect("chunk cache poisoned");
                match guard.entry(chunk_hash) {
                    Entry::Vacant(slot) => {
                        let cursor = Arc::new(OnceLock::new());
                        slot.insert(CacheEntry {
                            first_index: index,
                            handle: cursor.clone(),
                        });
                        (index, cursor)
                    }
                    Entry::Occupied(slot) => {
                        let cached = slot.get();
                        (cached.first_index, cached.handle.clone())
                    }
                }
            };

            let mut skipped = false;

            if first_index == index {
                let below_min = min_entropy.is_some_and(|min| entropy_bits < min);
                let above_max = max_entropy.is_some_and(|max| entropy_bits > max);

                if below_min {
                    skipped_low_entropy_ref.fetch_add(1, Ordering::Relaxed);
                }
                if above_max {
                    skipped_high_entropy_ref.fetch_add(1, Ordering::Relaxed);
                }

                if below_min || above_max {
                    handle
                        .set(None)
                        .map_err(|_| anyhow!("chunk result initialised multiple times"))?;
                    result_slots_ref[index]
                        .set(None)
                        .map_err(|_| anyhow!("chunk result already written"))?;
                    skipped = true;

                    if let Some(pb) = progress_for_workers.as_ref() {
                        pb.inc(1);
                        pb.suspend(|| {
                            info!(
                                "chunk {}/{} ({:.2} MiB) entropy {:.2} bits filtered by thresholds",
                                index + 1,
                                total_chunks,
                                bytes_to_mebibytes(byte_len),
                                entropy_bits
                            );
                        });
                    } else {
                        info!(
                            "chunk {}/{} ({:.2} MiB) entropy {:.2} bits filtered by thresholds",
                            index + 1,
                            total_chunks,
                            bytes_to_mebibytes(byte_len),
                            entropy_bits
                        );
                    }
                } else {
                    if let Some(pb) = progress_for_workers.as_ref() {
                        pb.set_message(format!(
                            "training chunk {}/{} ({:.2} MiB)",
                            index + 1,
                            total_chunks,
                            bytes_to_mebibytes(byte_len)
                        ));
                    }

                    let sequences = vec![chunk];
                    let artifacts = trainer
                        .train_from_sequences(sequences)
                        .with_context(|| format!("failed to train chunk {index}"))?;
                    let TrainerArtifacts { model, metrics } = artifacts;

                    let model = Arc::new(model);
                    let vocab_size = model.vocab_size();
                    let token_bytes = model.token_bytes();
                    let merges = model.merges();
                    let iteration_metrics = metrics.iterations;

                    let mut merge_records_vec = Vec::with_capacity(merges.len());
                    for (merge_idx, &(left_id, right_id)) in merges.iter().enumerate() {
                        let left = bytes_to_latin1(&token_bytes[left_id as usize]);
                        let right = bytes_to_latin1(&token_bytes[right_id as usize]);
                        let metrics = iteration_metrics.get(merge_idx);
                        let (best_frequency, merges_applied) = metrics
                            .map(|m| (m.best_frequency, m.merges_applied))
                            .unwrap_or((0, 0));
                        merge_records_vec.push(ChunkMergeRecord {
                            iteration: merge_idx + 1,
                            left,
                            right,
                            best_frequency,
                            merges_applied,
                        });
                    }

                    let merge_records = Arc::new(merge_records_vec);
                    let merge_count = merge_records.len();

                    let result = Arc::new(ChunkResultCore {
                        byte_len,
                        entropy_bits,
                        vocab_size,
                        merge_count,
                        merges: Arc::clone(&merge_records),
                        model: Arc::clone(&model),
                    });

                    handle
                        .set(Some(result.clone()))
                        .map_err(|_| anyhow!("chunk result initialised multiple times"))?;
                    result_slots_ref[index]
                        .set(Some(result.clone()))
                        .map_err(|_| anyhow!("chunk result already written"))?;

                    if let Some(pb) = progress_for_workers.as_ref() {
                        pb.inc(1);
                        pb.suspend(|| {
                            info!(
                                "chunk {}/{} ({:.2} MiB) => vocab {} ({} merges, entropy {:.2} bits)",
                                index + 1,
                                total_chunks,
                                bytes_to_mebibytes(byte_len),
                                vocab_size,
                                merge_count,
                                entropy_bits
                            );
                        });
                    } else {
                        info!(
                            "chunk {}/{} ({:.2} MiB) => vocab {} ({} merges, entropy {:.2} bits)",
                            index + 1,
                            total_chunks,
                            bytes_to_mebibytes(byte_len),
                            vocab_size,
                            merge_count,
                            entropy_bits
                        );
                    }
                }
            } else {
                let result_option = loop {
                    if let Some(result) = handle.get() {
                        break result.clone();
                    }
                    std::thread::yield_now();
                };

                match result_option {
                    Some(arc) => {
                        result_slots_ref[index]
                            .set(Some(arc.clone()))
                            .map_err(|_| anyhow!("duplicate chunk result already written"))?;

                        let hash_hex = chunk_hash.to_hex().to_string();
                        let hash_preview = hash_hex.get(..12).unwrap_or(&hash_hex);

                        if let Some(pb) = progress_for_workers.as_ref() {
                            pb.inc(1);
                            pb.suspend(|| {
                                info!(
                                    "chunk {}/{} duplicate of chunk {} (hash {}...) => vocab {} ({} merges, entropy {:.2} bits)",
                                    index + 1,
                                    total_chunks,
                                    first_index + 1,
                                    hash_preview,
                                    arc.vocab_size,
                                    arc.merge_count,
                                    arc.entropy_bits
                                );
                            });
                        } else {
                            info!(
                                "chunk {}/{} duplicate of chunk {} (hash {}...) => vocab {} ({} merges, entropy {:.2} bits)",
                                index + 1,
                                total_chunks,
                                first_index + 1,
                                hash_preview,
                                arc.vocab_size,
                                arc.merge_count,
                                arc.entropy_bits
                            );
                        }
                    }
                    None => {
                        skipped = true;
                        result_slots_ref[index]
                            .set(None)
                            .map_err(|_| anyhow!("duplicate chunk result already written"))?;

                        if let Some(pb) = progress_for_workers.as_ref() {
                            pb.inc(1);
                            pb.suspend(|| {
                                info!(
                                    "chunk {}/{} duplicate of chunk {} skipped due to entropy filter",
                                    index + 1,
                                    total_chunks,
                                    first_index + 1
                                );
                            });
                        } else {
                            info!(
                                "chunk {}/{} duplicate of chunk {} skipped due to entropy filter",
                                index + 1,
                                total_chunks,
                                first_index + 1
                            );
                        }
                    }
                }
            }

            meta_slots_ref[index]
                .set(ChunkMeta { first_index, skipped })
                .ok();

            Ok(())
        })?;

    let result_slots = Arc::try_unwrap(result_slots)
        .expect("result slots still referenced")
        .into_iter()
        .map(|slot| slot.into_inner().unwrap_or(None))
        .collect::<Vec<_>>();
    let meta_slots = Arc::try_unwrap(meta_slots)
        .expect("metadata slots still referenced")
        .into_iter()
        .map(|slot| slot.into_inner().expect("missing chunk metadata"))
        .collect::<Vec<_>>();

    let mut duplicate_chunks_reused = 0usize;
    let mut duplicate_chunks_skipped = 0usize;

    for (index, (meta, result)) in meta_slots.iter().zip(result_slots.iter()).enumerate() {
        if meta.skipped {
            continue;
        }

        let result = match result {
            Some(result) => result,
            None => continue,
        };

        let is_duplicate = meta.first_index != index;
        if is_duplicate && !args.duplicate_mode.counts_duplicates() {
            duplicate_chunks_skipped = duplicate_chunks_skipped.saturating_add(1);
            continue;
        }
        if is_duplicate {
            duplicate_chunks_reused = duplicate_chunks_reused.saturating_add(1);
        }

        let next_index = chunk_summaries.len();
        let summary = ChunkSummary {
            index: next_index,
            byte_len: result.byte_len,
            entropy_bits: result.entropy_bits,
            vocab_size: result.vocab_size,
            merge_count: result.merge_count,
            merges: Arc::clone(&result.merges),
        };

        let snapshot = ChunkTrainingSnapshot {
            index: next_index,
            byte_len: result.byte_len,
            model: Arc::clone(&result.model),
        };

        chunk_summaries.push(summary);
        snapshots.push(snapshot);
    }

    let skipped_low_entropy = skipped_low_entropy_counter.load(Ordering::Relaxed);
    let skipped_high_entropy = skipped_high_entropy_counter.load(Ordering::Relaxed);

    if skipped_low_entropy > 0 || skipped_high_entropy > 0 {
        info!(
            "filtered {skipped_low_entropy} chunk(s) below min-entropy and {skipped_high_entropy} chunk(s) above max-entropy"
        );
    }

    if let Some(pb) = progress {
        pb.finish_with_message(format!(
            "processed {} chunks (trained {} unique)",
            total_chunks,
            snapshots.len()
        ));
    }

    if duplicate_chunks_reused > 0 || duplicate_chunks_skipped > 0 {
        info!(
            "deduplicated {} chunk(s) using {} mode (reused {}, skipped {})",
            duplicate_chunks_reused + duplicate_chunks_skipped,
            args.duplicate_mode,
            duplicate_chunks_reused,
            duplicate_chunks_skipped
        );
    }

    if snapshots.is_empty() {
        return Err(anyhow!("no chunks were produced from the provided inputs"));
    }

    let combiner = args.combine_mode.build();
    info!(
        "combining {} chunks using {} strategy",
        snapshots.len(),
        combiner.name()
    );
    let combination = combiner.combine(&snapshots, &chunk_summaries, &trainer_cfg)?;
    let CombinationOutput {
        model: final_model,
        detail: combine_detail,
        stats: combine_stats,
    } = combination;

    if let Some(parent) = args.output.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
    }
    final_model
        .save_huggingface(&args.output)
        .with_context(|| {
            format!(
                "failed to save combined tokenizer to {}",
                args.output.display()
            )
        })?;

    emit_family_models(
        &final_model,
        &args.family_sizes,
        args.family_template.as_deref(),
        &args.output,
        false,
    )?;

    let combined_chunks = snapshots.len();

    if args.no_report {
        // Free heavy per-chunk summaries before printing to reduce peak memory.
        drop(chunk_summaries);
        drop(snapshots);

        println!(
            "🧪 chunk-train combined {} chunks ({} bytes across {} processed chunks) using {} -> {}",
            combined_chunks,
            total_bytes,
            total_chunks,
            args.combine_mode,
            args.output.display()
        );
        println!("   combine detail: {combine_detail}");
        println!(
            "   duplicate mode: {} (reused {}, skipped {})",
            args.duplicate_mode, duplicate_chunks_reused, duplicate_chunks_skipped
        );
        if args.min_entropy.is_some() || args.max_entropy.is_some() {
            let min_display = args
                .min_entropy
                .map(|v| format!("{v:.4}"))
                .unwrap_or_else(|| "none".to_string());
            let max_display = args
                .max_entropy
                .map(|v| format!("{v:.4}"))
                .unwrap_or_else(|| "none".to_string());
            println!(
                "   entropy filter: min {min_display} max {max_display} (skipped low {skipped_low_entropy}, high {skipped_high_entropy})"
            );
        }
        println!("   chunk report disabled (--no-report)");
    } else {
        let report = ChunkTrainReport {
            chunk_size_bytes: args.chunk_size,
            trainer_vocab_size: trainer_cfg.target_vocab_size,
            trainer_min_frequency: trainer_cfg.min_frequency,
            trainer_min_entropy: args.min_entropy,
            trainer_max_entropy: args.max_entropy,
            total_bytes,
            processed_chunks: total_chunks,
            total_chunks: combined_chunks,
            duplicate_mode: args.duplicate_mode.to_string(),
            duplicate_chunks_reused,
            duplicate_chunks_skipped,
            skipped_low_entropy,
            skipped_high_entropy,
            combine_mode: args.combine_mode.to_string(),
            combine_detail: combine_detail.clone(),
            combine_stats: combine_stats.clone(),
            final_vocab_size: final_model.vocab_size(),
            chunks: chunk_summaries,
        };

        if let Some(parent) = args.report.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)
                    .with_context(|| format!("failed to create {}", parent.display()))?;
            }
        }
        let mut report_file = File::create(&args.report).with_context(|| {
            format!("failed to write chunk report to {}", args.report.display())
        })?;
        serde_json::to_writer_pretty(&mut report_file, &report)
            .context("failed to serialise chunk report")?;
        report_file.write_all(b"\n").with_context(|| {
            format!("failed to finalise chunk report {}", args.report.display())
        })?;

        println!(
            "🧪 chunk-train combined {} chunks ({} bytes across {} processed chunks) using {} -> {}",
            report.total_chunks,
            report.total_bytes,
            report.processed_chunks,
            args.combine_mode,
            args.output.display()
        );
        println!("   combine detail: {}", report.combine_detail);
        println!(
            "   duplicate mode: {} (reused {}, skipped {})",
            report.duplicate_mode, report.duplicate_chunks_reused, report.duplicate_chunks_skipped
        );
        if args.min_entropy.is_some() || args.max_entropy.is_some() {
            let min_display = args
                .min_entropy
                .map(|v| format!("{v:.4}"))
                .unwrap_or_else(|| "none".to_string());
            let max_display = args
                .max_entropy
                .map(|v| format!("{v:.4}"))
                .unwrap_or_else(|| "none".to_string());
            println!(
                "   entropy filter: min {min_display} max {max_display} (skipped low {skipped_low_entropy}, high {skipped_high_entropy})"
            );
        }
        println!("   report written to {}", args.report.display());
    }

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

fn emit_family_models(
    model: &BpeModel,
    family_sizes: &[usize],
    template: Option<&str>,
    base_output: &Path,
    pretty: bool,
) -> Result<()> {
    if family_sizes.is_empty() {
        return Ok(());
    }
    let mut requested = family_sizes.to_vec();
    requested.sort_unstable();
    requested.dedup();
    if requested.is_empty() {
        return Ok(());
    }

    let mut derived_paths = Vec::new();
    for size in requested {
        let derived = model
            .derive_with_vocab(size)
            .with_context(|| format!("failed to derive vocabulary of size {size}"))?;
        let output_path = resolve_family_output_path(base_output, template, size);
        if let Some(parent) = output_path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).with_context(|| {
                    format!(
                        "failed to create family output directory {}",
                        parent.display()
                    )
                })?;
            }
        }
        derived.save_huggingface(&output_path).with_context(|| {
            format!(
                "failed to save derived tokenizer for vocab {size} to {}",
                output_path.display()
            )
        })?;
        if pretty {
            let pretty_json = derived.to_huggingface_json(true)?;
            fs::write(&output_path, pretty_json).with_context(|| {
                format!(
                    "failed to pretty print derived tokenizer {}",
                    output_path.display()
                )
            })?;
        }
        derived_paths.push((size, output_path));
    }

    if !derived_paths.is_empty() {
        println!("📚 derived tokenizer family members:");
        for (size, path) in derived_paths {
            println!("   vocab {size} -> {}", path.display());
        }
    }

    Ok(())
}

fn resolve_family_output_path(base: &Path, template: Option<&str>, size: usize) -> PathBuf {
    if let Some(pattern) = template {
        let rendered = pattern.replace("{size}", &size.to_string());
        let candidate = PathBuf::from(rendered);
        if candidate.is_relative() {
            let parent = base
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .map(Path::to_path_buf)
                .unwrap_or_else(|| PathBuf::from("."));
            if parent.as_os_str().is_empty() || parent == Path::new(".") {
                candidate
            } else {
                parent.join(candidate)
            }
        } else {
            candidate
        }
    } else {
        default_family_output_path(base, size)
    }
}

fn default_family_output_path(base: &Path, size: usize) -> PathBuf {
    let parent = base
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let stem = base
        .file_stem()
        .and_then(|s| s.to_str())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| "tokenizer".to_string());
    let extension = base
        .extension()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("json");
    let filename = if extension.is_empty() {
        format!("{stem}-{size}")
    } else {
        format!("{stem}-{size}.{}", extension)
    };
    if parent.as_os_str().is_empty() || parent == Path::new(".") {
        PathBuf::from(filename)
    } else {
        parent.join(filename)
    }
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
