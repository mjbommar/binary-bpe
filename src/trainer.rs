//! Core training loop responsible for producing `tokenizer.json` artefacts.

use std::cmp::Ordering;
use std::collections::{hash_map::Entry, BinaryHeap};
use std::convert::TryFrom;
use std::time::Instant;
use std::{fmt, path::Path};

use log::info;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::bytes::is_allowed_length;
use crate::config::{IngestConfig, TrainerBuilder, TrainerConfig};
use crate::corpus::load_binary_corpus;
use crate::error::{BbpeError, Result};
use crate::metrics::{sample_rss_kb, IterationMetrics, StopReason, TrainingMetrics};
use crate::model::{BpeModel, Pair, TokenId};

/// High-level façade configuring and executing BPE training runs.
#[derive(Debug, Clone)]
pub struct Trainer {
    cfg: TrainerConfig,
}

/// Artifacts returned after a training session completes.
#[must_use]
#[derive(Debug, Clone)]
pub struct TrainerArtifacts {
    /// Trained BPE model.
    pub model: BpeModel,
    /// Detailed metrics captured during training.
    pub metrics: TrainingMetrics,
}

impl Trainer {
    /// Creates a new trainer for the supplied configuration.
    #[must_use]
    pub fn new(cfg: TrainerConfig) -> Self {
        Self { cfg }
    }

    /// Returns a [`TrainerBuilder`] with default settings.
    #[must_use]
    pub fn builder() -> TrainerBuilder {
        TrainerConfig::builder()
    }

    /// Returns an immutable reference to the underlying configuration.
    #[must_use]
    pub fn config(&self) -> &TrainerConfig {
        &self.cfg
    }

    /// Trains a model by loading files from disk according to [`IngestConfig`].
    pub fn train_from_paths<P: AsRef<Path>>(
        &self,
        inputs: &[P],
        ingest: &IngestConfig,
    ) -> Result<TrainerArtifacts> {
        let sequences = load_binary_corpus(inputs, ingest)?;
        self.train_from_sequences(&sequences)
    }

    /// Trains a model from in-memory byte sequences.
    pub fn train_from_sequences(&self, sequences: &[Vec<u8>]) -> Result<TrainerArtifacts> {
        if sequences.is_empty() {
            return Err(BbpeError::InvalidConfig(
                "training requires at least one non-empty sequence".into(),
            ));
        }
        self.cfg.validate()?;

        let base_vocab = 256usize;
        let special_count = self.cfg.special_tokens.len();
        let max_new_tokens = self.cfg.target_vocab_size - base_vocab - special_count;
        if max_new_tokens == 0 {
            return Err(BbpeError::InvalidConfig(
                "target vocab leaves no room for merges".into(),
            ));
        }

        let allowed_lengths = self.cfg.allowed_token_lengths.clone();
        let mut working_sequences: Vec<Vec<TokenId>> = sequences
            .iter()
            .map(|seq| seq.iter().map(|&b| TokenId::from(b)).collect())
            .collect();

        let mut token_bytes: Vec<Vec<u8>> = (0u8..=u8::MAX).map(|b| vec![b]).collect();
        let mut token_lengths: Vec<usize> = vec![1; token_bytes.len()];
        let mut merges: Vec<Pair> = Vec::with_capacity(max_new_tokens);

        let mut pair_counts =
            compute_pair_counts(&working_sequences, &token_lengths, &allowed_lengths);
        let mut heap = BinaryHeap::with_capacity(pair_counts.len().max(1));
        for (&pair, &count) in &pair_counts {
            if count >= self.cfg.min_frequency {
                heap.push(PairScore::new(pair, count));
            }
        }

        let mut iteration = 0usize;
        let mut metrics = TrainingMetrics::new(max_new_tokens.min(16_384));
        let training_start = Instant::now();
        let plateau_floor = self.cfg.plateau_frequency_floor.max(self.cfg.min_frequency);
        let plateau_stop_enabled = self.cfg.plateau_stop_enabled && self.cfg.plateau_patience > 0;
        let mut plateau_streak = 0usize;
        let mut initial_frequency: Option<usize> = None;

        while merges.len() < max_new_tokens {
            if let Some(max_iters) = self.cfg.max_merge_iterations {
                if iteration >= max_iters {
                    metrics.stop_reason = StopReason::MaxIterationsReached;
                    break;
                }
            }

            let iteration_start = Instant::now();
            let best_candidate = loop {
                match heap.pop() {
                    Some(score) => {
                        let current = pair_counts.get(&score.pair).copied().unwrap_or(0);
                        if current == 0 || current != score.frequency {
                            continue;
                        }
                        if current < self.cfg.min_frequency {
                            continue;
                        }
                        break Some((score.pair, current));
                    }
                    None => break None,
                }
            };

            let Some((best_pair, frequency)) = best_candidate else {
                metrics.stop_reason = StopReason::NoEligiblePairs;
                break;
            };

            if initial_frequency.is_none() {
                initial_frequency = Some(frequency);
            }

            let freq_low = frequency <= plateau_floor
                || initial_frequency.is_some_and(|init| {
                    (frequency as u128) * (self.cfg.plateau_frequency_divisor as u128)
                        <= init as u128
                });

            if plateau_stop_enabled {
                if freq_low {
                    plateau_streak += 1;
                    if plateau_streak >= self.cfg.plateau_patience {
                        metrics.stop_reason = StopReason::PlateauReached;
                        break;
                    }
                } else {
                    plateau_streak = 0;
                }
            }

            let combined_len =
                token_lengths[best_pair.0 as usize] + token_lengths[best_pair.1 as usize];
            let mut new_token = Vec::with_capacity(combined_len);
            new_token.extend_from_slice(&token_bytes[best_pair.0 as usize]);
            new_token.extend_from_slice(&token_bytes[best_pair.1 as usize]);
            let new_token_id = TokenId::try_from(token_bytes.len())
                .map_err(|_| BbpeError::Internal("vocabulary size exceeded u32::MAX".into()))?;

            let total_merges = apply_merge(
                &mut working_sequences,
                best_pair,
                new_token_id,
                combined_len,
                &mut pair_counts,
                &mut heap,
                &token_lengths,
                &allowed_lengths,
            );

            if total_merges == 0 {
                metrics.stop_reason = StopReason::NoEligiblePairs;
                break;
            }

            token_bytes.push(new_token);
            token_lengths.push(combined_len);
            merges.push(best_pair);
            iteration += 1;

            if self.cfg.show_progress {
                info!(
                    "iter {:>6} freq {:>8} merges {:>8} distinct_pairs {:>8} vocab {:>8}",
                    iteration,
                    frequency,
                    total_merges,
                    pair_counts.len(),
                    base_vocab + iteration
                );
            }

            let iteration_metrics = IterationMetrics {
                iteration,
                best_frequency: frequency,
                merges_applied: total_merges,
                distinct_pairs: pair_counts.len(),
                elapsed_iteration: iteration_start.elapsed(),
                elapsed_total: training_start.elapsed(),
                rss_kb: sample_rss_kb(),
            };
            metrics.iterations.push(iteration_metrics);
        }

        if metrics.iterations.len() == max_new_tokens {
            metrics.stop_reason = StopReason::TargetVocabReached;
        }
        let total_duration = training_start.elapsed();
        metrics.total_duration = total_duration;

        if self.cfg.show_progress {
            info!(
                "completed {} merges in {:.2?}; vocab size {}",
                merges.len(),
                total_duration,
                token_bytes.len() + self.cfg.special_tokens.len()
            );
        }

        let model = BpeModel::new(token_bytes, merges, self.cfg.clone());
        Ok(TrainerArtifacts { model, metrics })
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct PairScore {
    frequency: usize,
    pair: Pair,
}

impl PairScore {
    fn new(pair: Pair, frequency: usize) -> Self {
        Self { frequency, pair }
    }
}

impl Ord for PairScore {
    fn cmp(&self, other: &Self) -> Ordering {
        self.frequency
            .cmp(&other.frequency)
            .then_with(|| other.pair.cmp(&self.pair))
    }
}

impl PartialOrd for PairScore {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn compute_pair_counts(
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

#[derive(Default)]
struct MergeAdjustments {
    deltas: FxHashMap<Pair, i64>,
    merges: usize,
}

fn accumulate_delta(
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

fn apply_delta(
    pair_counts: &mut FxHashMap<Pair, usize>,
    heap: &mut BinaryHeap<PairScore>,
    pair: Pair,
    delta: i64,
) {
    if delta == 0 {
        return;
    }
    match delta.cmp(&0) {
        Ordering::Greater => {
            let amount = usize::try_from(delta.unsigned_abs())
                .expect("positive delta magnitude must fit in usize");
            let count = pair_counts.entry(pair).or_insert(0);
            *count += amount;
            heap.push(PairScore::new(pair, *count));
        }
        Ordering::Less => {
            let amount = usize::try_from(delta.unsigned_abs())
                .expect("negative delta magnitude must fit in usize");
            if let Entry::Occupied(mut occupied) = pair_counts.entry(pair) {
                let current = *occupied.get();
                let new_value = current.saturating_sub(amount);
                if new_value == 0 {
                    occupied.remove();
                } else {
                    *occupied.get_mut() = new_value;
                    heap.push(PairScore::new(pair, new_value));
                }
            }
        }
        Ordering::Equal => {}
    }
}

fn token_length_with_new(
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

fn process_sequence(
    sequence: &mut Vec<TokenId>,
    pair: Pair,
    new_token: TokenId,
    new_token_len: usize,
    token_lengths: &[usize],
    allowed_lengths: &[usize],
) -> MergeAdjustments {
    let mut result = MergeAdjustments::default();
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
                let prev_len = token_length_with_new(prev, token_lengths, new_token, new_token_len);
                let combined = prev_len + left_len;
                accumulate_delta(
                    &mut result.deltas,
                    (prev, pair.0),
                    combined,
                    allowed_lengths,
                    -1,
                );
            }
            accumulate_delta(
                &mut result.deltas,
                pair,
                left_len + right_len,
                allowed_lengths,
                -1,
            );
            if let Some(next) = next_token {
                let next_len = token_length_with_new(next, token_lengths, new_token, new_token_len);
                let combined = right_len + next_len;
                accumulate_delta(
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
                let prev_len = token_length_with_new(prev, token_lengths, new_token, new_token_len);
                let combined = prev_len + new_token_len;
                accumulate_delta(
                    &mut result.deltas,
                    (prev, new_token),
                    combined,
                    allowed_lengths,
                    1,
                );
            }
            if let Some(next) = next_token {
                let next_len = token_length_with_new(next, token_lengths, new_token, new_token_len);
                let combined = new_token_len + next_len;
                accumulate_delta(
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
fn apply_merge(
    sequences: &mut [Vec<TokenId>],
    pair: Pair,
    new_token: TokenId,
    new_token_len: usize,
    pair_counts: &mut FxHashMap<Pair, usize>,
    heap: &mut BinaryHeap<PairScore>,
    token_lengths: &[usize],
    allowed_lengths: &[usize],
) -> usize {
    let aggregate = sequences
        .par_iter_mut()
        .map(|sequence| {
            process_sequence(
                sequence,
                pair,
                new_token,
                new_token_len,
                token_lengths,
                allowed_lengths,
            )
        })
        .reduce(MergeAdjustments::default, |mut acc, mut local| {
            acc.merges += local.merges;
            for (pair_key, delta) in local.deltas.drain() {
                *acc.deltas.entry(pair_key).or_insert(0) += delta;
            }
            acc
        });

    for (pair_key, delta) in aggregate.deltas {
        apply_delta(pair_counts, heap, pair_key, delta);
    }

    aggregate.merges
}

impl fmt::Display for TrainerArtifacts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "BPE model with vocab size {}", self.model.vocab_size())?;
        writeln!(f, "Stop reason: {:?}", self.metrics.stop_reason)?;
        writeln!(f, "Total duration: {:?}", self.metrics.total_duration)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::BinaryTokenizer;
    use crate::serialization;
    use tempfile::tempdir;

    fn trainer(min_frequency: usize, vocab_size: usize) -> Trainer {
        let cfg = TrainerConfig::builder()
            .min_frequency(min_frequency)
            .target_vocab_size(vocab_size)
            .special_tokens(Vec::<String>::new())
            .show_progress(false)
            .build()
            .unwrap();
        Trainer::new(cfg)
    }

    #[test]
    fn trainer_produces_merges() {
        let sequences = vec![
            vec![0x10, 0x20, 0x10, 0x20, 0x10, 0x20],
            vec![0x10, 0x20, 0x30, 0x40],
            vec![0x10, 0x20, 0x10, 0x20],
        ];
        let trainer = trainer(2, 270);
        let artefacts = trainer.train_from_sequences(&sequences).unwrap();
        assert!(!artefacts.model.merges().is_empty());
        assert!(!artefacts.metrics.iterations.is_empty());
    }

    #[test]
    fn tokenizer_round_trip() {
        let sequences = vec![vec![0xAA, 0xBB, 0xAA, 0xBB], vec![0xAA, 0xBB, 0xCC, 0xDD]];
        let trainer = trainer(2, 264);
        let artefacts = trainer.train_from_sequences(&sequences).unwrap();
        let bin_tok = BinaryTokenizer::from_model(&artefacts.model).unwrap();
        let encoded = bin_tok
            .encode_bytes(&[0xAA, 0xBB, 0xAA, 0xBB], false)
            .unwrap();
        let decoded = bin_tok.decode_to_bytes(&encoded, false).unwrap();
        assert_eq!(decoded, vec![0xAA, 0xBB, 0xAA, 0xBB]);
    }

    #[test]
    fn tokenizer_saves_huggingface_json() {
        let sequences = vec![vec![0, 1, 2, 3, 4, 5, 6, 7, 8]];
        let trainer = trainer(1, 258);
        let artefacts = trainer.train_from_sequences(&sequences).unwrap();
        let dir = tempdir().unwrap();
        let path = dir.path().join("tokenizer.json");
        artefacts.model.save_huggingface(&path).unwrap();
        let tokenizer = serialization::load_tokenizer(&path).unwrap();
        assert_eq!(
            tokenizer.get_vocab_size(false),
            artefacts.model.vocab_size()
        );
    }
}
