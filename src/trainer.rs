//! Core training loop responsible for producing `tokenizer.json` artefacts.

use std::cmp::Ordering;
use std::collections::{hash_map::Entry, BinaryHeap};
use std::convert::TryFrom;
use std::time::Instant;
use std::{fmt, path::Path};

use log::info;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::bytes::{
    contains_ascii_letter, ends_with_ascii_whitespace, is_all_ascii_whitespace,
    is_ascii_whitespace, starts_with_ascii_whitespace,
};
use crate::config::{IngestConfig, TrainerBuilder, TrainerConfig};
use crate::corpus::{stream_binary_corpus, stream_jsonl_corpus, JsonlSpec};
use crate::error::{BbpeError, Result};
use crate::metrics::{sample_rss_kb, IterationMetrics, StopReason, TrainingMetrics};
use crate::model::{BpeModel, Pair, TokenId};
use crate::preprocess::PreprocessorRunner;
use crate::special_tokens;

const PAIR_CACHE_MULTIPLIER: usize = 32;
const MIN_PAIR_CACHE: usize = 10_000;

mod word;
use word::Word;

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

/// Iterator wrapper that optionally tracks the total number of available sequences.
pub struct SequenceStream<I> {
    iter: I,
    length_hint: Option<usize>,
}

impl<I> SequenceStream<I> {
    /// Creates a stream without a known length.
    #[must_use]
    pub fn new(iter: I) -> Self {
        Self {
            iter,
            length_hint: None,
        }
    }

    /// Creates a stream with the provided exact length hint.
    #[must_use]
    pub fn with_length_hint(iter: I, length_hint: usize) -> Self {
        Self {
            iter,
            length_hint: Some(length_hint),
        }
    }

    /// Returns the optional length hint if one was provided.
    #[must_use]
    pub fn len_hint(&self) -> Option<usize> {
        self.length_hint
    }
}

impl<I> Iterator for SequenceStream<I>
where
    I: Iterator<Item = Result<Vec<u8>>>,
{
    type Item = Result<Vec<u8>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
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
        let chunk_stream = stream_binary_corpus(inputs, ingest)?;
        let length_hint = chunk_stream.total_chunks();
        let stream = SequenceStream::with_length_hint(chunk_stream, length_hint);
        self.train_from_stream(stream)
    }

    /// Trains a model by extracting string fields from newline-delimited JSON (JSONL) files.
    pub fn train_from_jsonl(&self, specs: &[JsonlSpec]) -> Result<TrainerArtifacts> {
        let stream = stream_jsonl_corpus(specs)?;
        self.train_from_stream(SequenceStream::new(stream))
    }

    /// Trains a model from in-memory byte sequences.
    pub fn train_from_sequences(&self, sequences: Vec<Vec<u8>>) -> Result<TrainerArtifacts> {
        let length = sequences.len();
        let iter = sequences.into_iter().map(Ok::<_, BbpeError>);
        self.train_from_stream(SequenceStream::with_length_hint(iter, length))
    }

    /// Streams byte sequences into the trainer without buffering the entire corpus.
    pub fn train_from_stream<I>(&self, stream: SequenceStream<I>) -> Result<TrainerArtifacts>
    where
        I: Iterator<Item = Result<Vec<u8>>> + Send,
    {
        self.cfg.validate()?;
        let length_hint = stream.len_hint();

        let mut runner = PreprocessorRunner::new(self.cfg.preprocessor.clone());
        let leading_specials = special_tokens::leading_tokens();
        let trailing_specials = special_tokens::trailing_tokens(
            self.cfg.reasoning_tokens_enabled,
            &self.cfg.special_tokens,
        );
        let leading_count = leading_specials.len();
        let trailing_count = trailing_specials.len();
        let byte_vocab = 256usize;
        let initial_vocab = leading_count + byte_vocab + trailing_count;
        if self.cfg.target_vocab_size < initial_vocab {
            return Err(BbpeError::InvalidConfig(
                "target vocab smaller than required specials + byte alphabet".into(),
            ));
        }
        let max_new_tokens = self.cfg.target_vocab_size - initial_vocab;
        if max_new_tokens == 0 {
            return Err(BbpeError::InvalidConfig(
                "target vocab leaves no room for merges".into(),
            ));
        }
        let byte_offset =
            TokenId::try_from(leading_count).expect("leading special token count fits in TokenId");

        if self.cfg.show_progress {
            if let Some(hint) = length_hint {
                info!("streaming approximately {hint} corpus chunks");
            } else {
                info!("streaming corpus with unknown length");
            }
        }

        let mut aggregated_sequences: FxHashMap<Vec<TokenId>, u64> = FxHashMap::default();
        let mut observed_sequences = 0usize;

        for chunk in stream {
            let chunk = chunk?;
            if chunk.is_empty() {
                continue;
            }
            observed_sequences += 1;
            runner.process(&chunk, |segment| {
                let bytes = segment.as_ref();
                if bytes.is_empty() {
                    return;
                }
                let tokens: Vec<TokenId> = bytes
                    .iter()
                    .map(|&b| byte_offset + TokenId::from(b))
                    .collect();
                *aggregated_sequences.entry(tokens).or_insert(0) += 1;
            });
        }

        if observed_sequences == 0 {
            return Err(BbpeError::InvalidConfig(
                "training requires at least one non-empty sequence".into(),
            ));
        }
        if aggregated_sequences.is_empty() {
            return Err(BbpeError::InvalidConfig(
                "training requires at least one non-empty sequence after preprocessing".into(),
            ));
        }

        let allowed_lengths = self.cfg.allowed_token_lengths.clone();
        let mut token_bytes: Vec<Vec<u8>> = Vec::with_capacity(self.cfg.target_vocab_size);
        let mut token_lengths: Vec<usize> = Vec::with_capacity(self.cfg.target_vocab_size);
        let mut token_lookup: FxHashSet<Vec<u8>> =
            FxHashSet::with_capacity_and_hasher(self.cfg.target_vocab_size, Default::default());
        for token in leading_specials {
            let bytes = token.as_bytes().to_vec();
            token_lengths.push(bytes.len());
            token_lookup.insert(bytes.clone());
            token_bytes.push(bytes);
        }
        for byte in 0u8..=u8::MAX {
            let bytes = vec![byte];
            token_lengths.push(bytes.len());
            token_lookup.insert(bytes.clone());
            token_bytes.push(bytes);
        }
        for token in &trailing_specials {
            let bytes = token.as_bytes().to_vec();
            token_lengths.push(bytes.len());
            token_lookup.insert(bytes.clone());
            token_bytes.push(bytes);
        }
        let mut merges: Vec<Pair> = Vec::with_capacity(max_new_tokens);

        let (mut words, counts): (Vec<Word>, Vec<u64>) = {
            let mut local_words = Vec::with_capacity(aggregated_sequences.len());
            let mut local_counts = Vec::with_capacity(aggregated_sequences.len());
            for (tokens, count) in aggregated_sequences {
                local_words.push(Word::from_tokens(tokens));
                local_counts.push(count);
            }
            (local_words, local_counts)
        };

        let (mut pair_counts, mut pair_positions) =
            compute_pair_stats(&words, &counts, &allowed_lengths);
        let mut heap = BinaryHeap::with_capacity(pair_counts.len().max(1));
        for (&pair, &count) in &pair_counts {
            if count >= self.cfg.min_frequency {
                heap.push(PairScore::new(pair, count));
            }
        }
        let pair_cache_cap = self
            .cfg
            .target_vocab_size
            .saturating_mul(PAIR_CACHE_MULTIPLIER)
            .max(MIN_PAIR_CACHE);
        prune_pair_cache(
            &mut pair_counts,
            &mut pair_positions,
            &mut heap,
            pair_cache_cap,
            self.cfg.min_frequency,
        );

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

            if ends_with_ascii_whitespace(&new_token) && !is_all_ascii_whitespace(&new_token) {
                pair_counts.remove(&best_pair);
                continue;
            }
            if self.cfg.require_letter_whitespace_merges
                && new_token.iter().copied().any(is_ascii_whitespace)
                && !is_all_ascii_whitespace(&new_token)
                && !contains_ascii_letter(&new_token)
            {
                pair_counts.remove(&best_pair);
                continue;
            }
            if self.cfg.forbid_leading_whitespace_merges
                && starts_with_ascii_whitespace(&new_token)
                && !is_all_ascii_whitespace(&new_token)
            {
                pair_counts.remove(&best_pair);
                continue;
            }
            if !token_lookup.insert(new_token.clone()) {
                pair_counts.remove(&best_pair);
                continue;
            }

            let new_token_id = TokenId::try_from(token_bytes.len())
                .map_err(|_| BbpeError::Internal("vocabulary size exceeded u32::MAX".into()))?;

            let total_merges = apply_merge(
                &mut words,
                &counts,
                best_pair,
                new_token_id,
                combined_len,
                &mut pair_counts,
                &mut heap,
                &allowed_lengths,
                &mut pair_positions,
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
                    initial_vocab + iteration
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

            prune_pair_cache(
                &mut pair_counts,
                &mut pair_positions,
                &mut heap,
                pair_cache_cap,
                self.cfg.min_frequency,
            );
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
                token_bytes.len()
            );
        }

        pad_vocabulary_with_whitespace(
            &mut token_bytes,
            &mut token_lengths,
            &mut token_lookup,
            self.cfg.target_vocab_size,
        );

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

#[allow(clippy::too_many_arguments)]
fn apply_merge(
    words: &mut [Word],
    counts: &[u64],
    pair: Pair,
    new_token: TokenId,
    new_token_len: usize,
    pair_counts: &mut FxHashMap<Pair, usize>,
    heap: &mut BinaryHeap<PairScore>,
    allowed_lengths: &[usize],
    pair_positions: &mut FxHashMap<Pair, FxHashSet<usize>>,
) -> usize {
    let positions = pair_positions.remove(&pair).unwrap_or_default();
    if positions.is_empty() {
        return 0;
    }

    let mut aggregate_deltas: FxHashMap<Pair, i64> = FxHashMap::default();
    let mut total_merges = 0usize;

    for idx in positions {
        if idx >= words.len() {
            continue;
        }
        let result = words[idx].merge(pair.0, pair.1, new_token, new_token_len, allowed_lengths);
        if result.merges == 0 {
            continue;
        }
        let weight_usize: usize = counts[idx]
            .try_into()
            .expect("sequence count must fit in usize");
        let weight_i64 = counts[idx] as i64;
        total_merges += result.merges * weight_usize;

        for (pair_key, delta) in result.deltas {
            let weighted = (delta as i64) * weight_i64;
            if weighted == 0 {
                continue;
            }
            *aggregate_deltas.entry(pair_key).or_insert(0) += weighted;
            if delta > 0 {
                pair_positions.entry(pair_key).or_default().insert(idx);
            }
        }
    }

    for (pair_key, delta) in aggregate_deltas {
        apply_delta(pair_counts, heap, pair_key, delta);
    }

    total_merges
}

fn compute_pair_stats(
    words: &[Word],
    counts: &[u64],
    allowed_lengths: &[usize],
) -> (FxHashMap<Pair, usize>, FxHashMap<Pair, FxHashSet<usize>>) {
    let mut pair_counts: FxHashMap<Pair, usize> = FxHashMap::default();
    let mut pair_positions: FxHashMap<Pair, FxHashSet<usize>> = FxHashMap::default();

    for (idx, word) in words.iter().enumerate() {
        if !word.has_pairs() {
            continue;
        }
        let weight: usize = counts[idx]
            .try_into()
            .expect("sequence count must fit in usize");
        word.for_each_pair(allowed_lengths, |pair| {
            *pair_counts.entry(pair).or_insert(0) += weight;
            pair_positions.entry(pair).or_default().insert(idx);
        });
    }

    (pair_counts, pair_positions)
}

const WHITESPACE_PADDING_BYTES: [u8; 4] = [b' ', b'\t', b'\n', b'\r'];

fn pad_vocabulary_with_whitespace(
    token_bytes: &mut Vec<Vec<u8>>,
    token_lengths: &mut Vec<usize>,
    token_lookup: &mut FxHashSet<Vec<u8>>,
    target_vocab_size: usize,
) {
    if token_bytes.len() >= target_vocab_size {
        return;
    }
    let mut needed = target_vocab_size - token_bytes.len();
    let mut run_len = 2usize;
    while needed > 0 {
        for &byte in &WHITESPACE_PADDING_BYTES {
            if needed == 0 {
                break;
            }
            let candidate = vec![byte; run_len];
            if token_lookup.insert(candidate.clone()) {
                token_lengths.push(candidate.len());
                token_bytes.push(candidate);
                needed -= 1;
            }
        }
        run_len += 1;
    }

    while token_bytes.len() < target_vocab_size {
        let mut candidate = vec![0u8, (token_bytes.len() & 0xFF) as u8];
        while !token_lookup.insert(candidate.clone()) {
            candidate[1] = candidate[1].wrapping_add(1);
        }
        token_lengths.push(candidate.len());
        token_bytes.push(candidate);
    }
}

impl fmt::Display for TrainerArtifacts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "BPE model with vocab size {}", self.model.vocab_size())?;
        writeln!(f, "Stop reason: {:?}", self.metrics.stop_reason)?;
        writeln!(f, "Total duration: {:?}", self.metrics.total_duration)?;
        Ok(())
    }
}

fn prune_pair_cache(
    pair_counts: &mut FxHashMap<Pair, usize>,
    pair_positions: &mut FxHashMap<Pair, FxHashSet<usize>>,
    heap: &mut BinaryHeap<PairScore>,
    max_pairs: usize,
    min_frequency: usize,
) {
    if pair_counts.len() <= max_pairs {
        return;
    }
    let mut entries: Vec<(Pair, usize)> = pair_counts
        .iter()
        .filter(|(_, &count)| count >= min_frequency)
        .map(|(pair, &count)| (*pair, count))
        .collect();
    if entries.len() <= max_pairs {
        return;
    }
    entries.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    entries.truncate(max_pairs);
    let keep: FxHashSet<Pair> = entries.iter().map(|(pair, _)| *pair).collect();
    pair_counts.retain(|pair, _| keep.contains(pair));
    pair_positions.retain(|pair, _| keep.contains(pair));
    heap.clear();
    for (pair, freq) in entries {
        heap.push(PairScore::new(pair, freq));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytes::{
        contains_ascii_letter, ends_with_ascii_whitespace, is_all_ascii_whitespace,
        is_ascii_whitespace,
    };
    use crate::model::BinaryTokenizer;
    use crate::serialization;
    use rustc_hash::FxHashSet;
    use tempfile::tempdir;

    fn trainer(min_frequency: usize, vocab_size: usize) -> Trainer {
        let cfg = TrainerConfig::builder()
            .min_frequency(min_frequency)
            .target_vocab_size(vocab_size)
            .special_tokens(Vec::<String>::new())
            .reasoning_tokens_enabled(false)
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
        let artefacts = trainer.train_from_sequences(sequences).unwrap();
        assert!(!artefacts.model.merges().is_empty());
        assert!(!artefacts.metrics.iterations.is_empty());
    }

    #[test]
    fn trainer_streams_data_incrementally() {
        let sequences = vec![
            vec![0x01, 0x02, 0x01, 0x02],
            vec![0x01, 0x02, 0x03, 0x04],
            vec![0x02, 0x03, 0x02, 0x03],
        ];
        let trainer = trainer(2, 270);
        let stream = SequenceStream::new(sequences.into_iter().map(Ok::<_, BbpeError>));
        let artefacts = trainer.train_from_stream(stream).unwrap();
        assert!(!artefacts.model.merges().is_empty());
    }

    #[test]
    fn tokenizer_round_trip() {
        let sequences = vec![vec![0xAA, 0xBB, 0xAA, 0xBB], vec![0xAA, 0xBB, 0xCC, 0xDD]];
        let trainer = trainer(2, 264);
        let artefacts = trainer.train_from_sequences(sequences).unwrap();
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
        let trainer = trainer(1, 270);
        let artefacts = trainer.train_from_sequences(sequences).unwrap();
        let dir = tempdir().unwrap();
        let path = dir.path().join("tokenizer.json");
        artefacts.model.save_huggingface(&path).unwrap();
        let tokenizer = serialization::load_tokenizer(&path).unwrap();
        assert_eq!(tokenizer.get_vocab_size(true), artefacts.model.vocab_size());
    }

    #[test]
    fn merges_do_not_leave_trailing_whitespace_tokens() {
        let sequences = vec![b"alpha beta gamma ".to_vec(); 64];
        let trainer = trainer(2, 270);
        let artefacts = trainer.train_from_sequences(sequences).unwrap();
        for token in artefacts.model.token_bytes() {
            if token.is_empty() {
                continue;
            }
            if ends_with_ascii_whitespace(token) {
                assert!(is_all_ascii_whitespace(token));
            }
        }
    }

    #[test]
    fn whitespace_merges_require_letters_enforced() {
        let sequences = vec![b"123 456 ".to_vec(); 64];
        let cfg = TrainerConfig::builder()
            .min_frequency(2)
            .target_vocab_size(270)
            .special_tokens(Vec::<String>::new())
            .reasoning_tokens_enabled(false)
            .show_progress(false)
            .require_letter_whitespace_merges(true)
            .build()
            .unwrap();
        let trainer = Trainer::new(cfg);
        let artefacts = trainer.train_from_sequences(sequences).unwrap();
        for token in artefacts.model.token_bytes() {
            if token.is_empty() {
                continue;
            }
            let has_ws = token.iter().copied().any(is_ascii_whitespace);
            if has_ws && !is_all_ascii_whitespace(token) {
                assert!(contains_ascii_letter(token));
            }
        }
    }

    #[test]
    fn leading_whitespace_merges_are_forbidden() {
        let sequences = vec![b" foo".to_vec(); 64];
        let cfg = TrainerConfig::builder()
            .min_frequency(2)
            .target_vocab_size(270)
            .special_tokens(Vec::<String>::new())
            .reasoning_tokens_enabled(false)
            .show_progress(false)
            .forbid_leading_whitespace_merges(true)
            .build()
            .unwrap();
        let trainer = Trainer::new(cfg);
        let artefacts = trainer.train_from_sequences(sequences).unwrap();
        for token in artefacts.model.token_bytes() {
            if token.is_empty() {
                continue;
            }
            let starts_ws = token.first().copied().is_some_and(is_ascii_whitespace);
            if starts_ws && !is_all_ascii_whitespace(token) {
                panic!("token {token:?} should not start with whitespace");
            }
        }
    }

    #[test]
    fn trainer_pads_vocab_with_whitespace_tokens() {
        let sequences = vec![b"ab".to_vec()];
        let cfg = TrainerConfig::builder()
            .min_frequency(10)
            .target_vocab_size(267)
            .special_tokens(Vec::<String>::new())
            .reasoning_tokens_enabled(false)
            .show_progress(false)
            .max_merge_iterations(Some(0))
            .build()
            .unwrap();
        let trainer = Trainer::new(cfg);
        let artefacts = trainer.train_from_sequences(sequences).unwrap();
        assert_eq!(artefacts.model.vocab_size(), 267);
        let extras = artefacts.model.token_bytes();
        let extras = &extras[extras.len().saturating_sub(4)..];
        assert_eq!(
            extras,
            &[
                b"  ".to_vec(),
                b"\t\t".to_vec(),
                b"\n\n".to_vec(),
                b"\r\r".to_vec()
            ]
        );
    }

    #[test]
    fn trainer_avoids_duplicate_special_token_bytes() {
        let glyph = "●".as_bytes().to_vec();
        let mut sequence = Vec::new();
        for _ in 0..64 {
            sequence.extend_from_slice(&glyph);
        }
        let cfg = TrainerConfig::builder()
            .min_frequency(1)
            .target_vocab_size(400)
            .special_tokens(Vec::<String>::new())
            .reasoning_tokens_enabled(true)
            .show_progress(false)
            .build()
            .unwrap();
        let trainer = Trainer::new(cfg);
        let artefacts = trainer
            .train_from_sequences(vec![sequence])
            .expect("training succeeds with repeated reasoning glyph");
        let mut seen = FxHashSet::default();
        for token in artefacts.model.token_bytes() {
            assert!(
                seen.insert(token.clone()),
                "duplicate token bytes detected: {:?}",
                token
            );
        }
    }
}
