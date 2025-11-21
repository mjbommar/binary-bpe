//! Preprocessing helpers that transform raw byte sequences before training.

use std::borrow::Cow;

use bstr::ByteSlice;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::bytes::is_ascii_whitespace;
use crate::config::{PreprocessorConfig, PreprocessorKind};

/// Borrowed-or-owned wrapper returned by [`apply_preprocessor`].
#[derive(Debug)]
pub enum PreprocessedSequences<'a> {
    /// References the original input sequences when no preprocessing is required.
    Borrowed(&'a [Vec<u8>]),
    /// Owns the transformed sequences produced by a preprocessor.
    Owned(Vec<Vec<u8>>),
}

impl<'a> PreprocessedSequences<'a> {
    /// Returns the underlying sequences slice regardless of ownership.
    #[must_use]
    pub fn as_slice(&self) -> &[Vec<u8>] {
        match self {
            Self::Borrowed(seqs) => seqs,
            Self::Owned(seqs) => seqs.as_slice(),
        }
    }
}

/// Stateful helper that incrementally applies preprocessing to streaming sequences.
#[derive(Debug)]
pub struct PreprocessorRunner {
    cfg: PreprocessorConfig,
    rng: Option<StdRng>,
    enabled: bool,
}

impl PreprocessorRunner {
    /// Creates a new runner for the provided configuration.
    #[must_use]
    pub fn new(cfg: PreprocessorConfig) -> Self {
        let enabled = !matches!(cfg.kind, PreprocessorKind::None) && cfg.split_probability > 0.0;
        let rng = if enabled {
            Some(match cfg.seed {
                Some(seed) => StdRng::seed_from_u64(seed),
                None => StdRng::from_entropy(),
            })
        } else {
            None
        };
        Self { cfg, rng, enabled }
    }

    /// Returns true when preprocessing will emit new sequences instead of borrowing.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Processes a single sequence, invoking the supplied closure for each emitted chunk.
    pub fn process<'a, F>(&'a mut self, sequence: &'a [u8], mut emit: F)
    where
        F: FnMut(Cow<'a, [u8]>),
    {
        if sequence.is_empty() {
            return;
        }
        if !self.enabled {
            emit(Cow::Borrowed(sequence));
            return;
        }
        let rng = self
            .rng
            .as_mut()
            .expect("runner should maintain RNG when enabled");
        match self.cfg.kind {
            PreprocessorKind::None => emit(Cow::Borrowed(sequence)),
            PreprocessorKind::AsciiWhitespace => {
                split_ascii_sequence(sequence, self.cfg.split_probability, rng, emit)
            }
            PreprocessorKind::UnicodeWhitespace => {
                split_unicode_sequence(sequence, self.cfg.split_probability, rng, emit)
            }
            PreprocessorKind::NullDelimited => {
                split_null_sequence(sequence, self.cfg.split_probability, rng, emit)
            }
        }
    }
}

/// Applies the configured preprocessor to an input corpus, returning either a borrowed or owned
/// slice of sequences depending on whether any transformation was required.
#[must_use]
pub fn apply_preprocessor<'a>(
    cfg: &PreprocessorConfig,
    sequences: &'a [Vec<u8>],
) -> PreprocessedSequences<'a> {
    let mut runner = PreprocessorRunner::new(cfg.clone());
    if !runner.is_enabled() {
        return PreprocessedSequences::Borrowed(sequences);
    }

    let mut owned = Vec::new();
    for seq in sequences {
        runner.process(seq, |chunk| {
            if chunk.is_empty() {
                return;
            }
            owned.push(chunk.as_ref().to_vec());
        });
    }
    PreprocessedSequences::Owned(owned)
}

fn split_ascii_sequence<'a, R, F>(sequence: &'a [u8], probability: f64, rng: &mut R, emit: F)
where
    R: Rng + ?Sized,
    F: FnMut(Cow<'a, [u8]>),
{
    split_by_predicate(
        sequence,
        is_ascii_whitespace,
        probability,
        rng,
        SplitBehavior::Whitespace,
        emit,
    );
}

fn split_unicode_sequence<'a, R, F>(sequence: &'a [u8], probability: f64, rng: &mut R, emit: F)
where
    R: Rng + ?Sized,
    F: FnMut(Cow<'a, [u8]>),
{
    if sequence.is_empty() {
        return;
    }
    let segments = collect_unicode_segments(sequence);
    merge_segments(
        sequence,
        segments,
        probability,
        rng,
        SplitBehavior::Whitespace,
        emit,
    );
}

fn split_null_sequence<'a, R, F>(sequence: &'a [u8], probability: f64, rng: &mut R, emit: F)
where
    R: Rng + ?Sized,
    F: FnMut(Cow<'a, [u8]>),
{
    split_by_predicate(
        sequence,
        |byte| byte == 0,
        probability,
        rng,
        SplitBehavior::Generic,
        emit,
    );
}

fn split_by_predicate<'a, R, P, F>(
    sequence: &'a [u8],
    predicate: P,
    probability: f64,
    rng: &mut R,
    behavior: SplitBehavior,
    emit: F,
) where
    R: Rng + ?Sized,
    P: Fn(u8) -> bool + Copy,
    F: FnMut(Cow<'a, [u8]>),
{
    if sequence.is_empty() {
        return;
    }
    let segments = collect_segments(sequence, predicate);
    merge_segments(sequence, segments, probability, rng, behavior, emit);
}

type Segment = (usize, usize, bool);

fn collect_segments<F>(sequence: &[u8], predicate: F) -> Vec<Segment>
where
    F: Fn(u8) -> bool,
{
    let mut segments = Vec::new();
    if sequence.is_empty() {
        return segments;
    }
    let mut start = 0usize;
    let mut current_kind = predicate(sequence[0]);
    for idx in 1..=sequence.len() {
        let boundary = if idx == sequence.len() {
            true
        } else {
            predicate(sequence[idx]) != current_kind
        };
        if boundary && idx > start {
            segments.push((start, idx, current_kind));
            if idx < sequence.len() {
                start = idx;
                current_kind = predicate(sequence[idx]);
            }
        }
    }
    segments
}

fn collect_unicode_segments(sequence: &[u8]) -> Vec<Segment> {
    let mut segments = Vec::new();
    let mut run_start = 0usize;
    let mut run_kind: Option<bool> = None;
    let mut previous_end = 0usize;

    for (start, end, ch) in sequence.char_indices() {
        let is_whitespace = ch.is_whitespace();
        match run_kind {
            Some(kind) if kind == is_whitespace => {
                previous_end = end;
            }
            Some(kind) => {
                if previous_end > run_start {
                    segments.push((run_start, previous_end, kind));
                }
                run_start = start;
                run_kind = Some(is_whitespace);
                previous_end = end;
            }
            None => {
                run_start = start;
                run_kind = Some(is_whitespace);
                previous_end = end;
            }
        }
    }

    if let Some(kind) = run_kind {
        let final_end = sequence.len().max(previous_end);
        if final_end > run_start {
            segments.push((run_start, final_end, kind));
        }
    }

    segments
}

fn merge_segments<'a, R, F>(
    sequence: &'a [u8],
    segments: Vec<Segment>,
    probability: f64,
    rng: &mut R,
    behavior: SplitBehavior,
    emit: F,
) where
    R: Rng + ?Sized,
    F: FnMut(Cow<'a, [u8]>),
{
    match behavior {
        SplitBehavior::Generic => {
            merge_segments_generic(sequence, segments, probability, rng, emit)
        }
        SplitBehavior::Whitespace => {
            merge_segments_whitespace(sequence, segments, probability, rng, emit)
        }
    }
}

fn merge_segments_generic<'a, R, F>(
    sequence: &'a [u8],
    segments: Vec<Segment>,
    probability: f64,
    rng: &mut R,
    mut emit: F,
) where
    R: Rng + ?Sized,
    F: FnMut(Cow<'a, [u8]>),
{
    if segments.is_empty() {
        return;
    }
    let mut buffer = sequence[segments[0].0..segments[0].1].to_vec();
    for (start, end, _) in segments.into_iter().skip(1) {
        let bytes = &sequence[start..end];
        if should_split(probability, rng) {
            emit(Cow::Owned(std::mem::take(&mut buffer)));
            buffer = bytes.to_vec();
        } else {
            buffer.extend_from_slice(bytes);
        }
    }
    emit(Cow::Owned(buffer));
}

fn merge_segments_whitespace<'a, R, F>(
    sequence: &'a [u8],
    segments: Vec<Segment>,
    probability: f64,
    rng: &mut R,
    mut emit: F,
) where
    R: Rng + ?Sized,
    F: FnMut(Cow<'a, [u8]>),
{
    if segments.is_empty() {
        return;
    }

    let mut buffer: Vec<u8> = Vec::new();
    let mut pending_whitespace: Option<Vec<u8>> = None;
    let mut idx = 0usize;

    while idx < segments.len() {
        let (start, end, is_whitespace) = segments[idx];
        let bytes = &sequence[start..end];

        if !is_whitespace {
            if let Some(mut pending) = pending_whitespace.take() {
                if buffer.is_empty() {
                    emit(Cow::Owned(std::mem::take(&mut pending)));
                } else {
                    buffer.extend_from_slice(&pending);
                }
            }
            buffer.extend_from_slice(bytes);
            idx += 1;
            continue;
        }

        let has_left = !buffer.is_empty();
        let has_right = segments
            .get(idx + 1)
            .map(|(_, _, next_kind)| !next_kind)
            .unwrap_or(false);

        if has_left && has_right && !should_split(probability, rng) {
            if let Some(existing) = &mut pending_whitespace {
                existing.extend_from_slice(bytes);
            } else {
                pending_whitespace = Some(bytes.to_vec());
            }
            idx += 1;
            continue;
        }

        if let Some(pending) = pending_whitespace.take() {
            if !buffer.is_empty() {
                emit(Cow::Owned(std::mem::take(&mut buffer)));
            }
            emit(Cow::Owned(pending));
        }

        if !buffer.is_empty() {
            emit(Cow::Owned(std::mem::take(&mut buffer)));
        }
        emit(Cow::Owned(bytes.to_vec()));
        idx += 1;
    }

    if let Some(pending) = pending_whitespace.take() {
        if !buffer.is_empty() {
            emit(Cow::Owned(std::mem::take(&mut buffer)));
        }
        emit(Cow::Owned(pending));
    } else if !buffer.is_empty() {
        emit(Cow::Owned(buffer));
    }
}

#[derive(Clone, Copy)]
enum SplitBehavior {
    Generic,
    Whitespace,
}

fn should_split<R: Rng + ?Sized>(probability: f64, rng: &mut R) -> bool {
    if probability >= 1.0 {
        return true;
    }
    if probability <= 0.0 {
        return false;
    }
    rng.gen_bool(probability)
}

#[cfg(test)]
fn chunk_is_unicode_whitespace(bytes: &[u8]) -> bool {
    bstr::BStr::new(bytes).chars().all(|ch| ch.is_whitespace())
}

#[cfg(test)]
fn ends_with_unicode_whitespace(bytes: &[u8]) -> bool {
    bstr::BStr::new(bytes)
        .chars()
        .next_back()
        .map(|ch| ch.is_whitespace())
        .unwrap_or(false)
}

#[cfg(test)]
fn chunk_has_mixed_unicode_content(bytes: &[u8]) -> bool {
    let mut has_whitespace = false;
    let mut has_non_whitespace = false;
    for ch in bstr::BStr::new(bytes).chars() {
        if ch.is_whitespace() {
            has_whitespace = true;
        } else {
            has_non_whitespace = true;
        }
        if has_whitespace && has_non_whitespace {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_whitespace_splits_and_preserves_runs() {
        let seq = b"foo  bar\tbaz".to_vec();
        let cfg = PreprocessorConfig {
            kind: PreprocessorKind::AsciiWhitespace,
            split_probability: 1.0,
            seed: Some(42),
        };
        let processed = match apply_preprocessor(&cfg, &[seq]) {
            PreprocessedSequences::Owned(data) => data,
            _ => panic!("expected owned sequences"),
        };
        assert_eq!(
            processed,
            vec![
                b"foo".to_vec(),
                b"  ".to_vec(),
                b"bar".to_vec(),
                b"\t".to_vec(),
                b"baz".to_vec()
            ]
        );
    }

    #[test]
    fn ascii_whitespace_probabilistic_merges_keep_suffixes_clean() {
        let seq = b"with respect to hygiene".to_vec();
        let cfg = PreprocessorConfig {
            kind: PreprocessorKind::AsciiWhitespace,
            split_probability: 0.3,
            seed: Some(7),
        };
        let processed = match apply_preprocessor(&cfg, &[seq]) {
            PreprocessedSequences::Owned(data) => data,
            _ => panic!("expected owned sequences"),
        };

        assert!(
            processed.iter().any(
                |chunk| chunk.contains(&b' ') && chunk.iter().any(|b| !is_ascii_whitespace(*b))
            ),
            "expected merged multi-word chunk"
        );
        for chunk in &processed {
            if let Some(last) = chunk.last() {
                if is_ascii_whitespace(*last) {
                    assert!(
                        chunk.iter().all(|b| is_ascii_whitespace(*b)),
                        "token {chunk:?} ends with whitespace but is not purely whitespace"
                    );
                }
            }
        }
    }

    #[test]
    fn ascii_whitespace_merges_create_internal_spacing() {
        let seq = b"alpha beta gamma".to_vec();
        let cfg = PreprocessorConfig {
            kind: PreprocessorKind::AsciiWhitespace,
            split_probability: 0.000001,
            seed: Some(1),
        };
        let processed = match apply_preprocessor(&cfg, std::slice::from_ref(&seq)) {
            PreprocessedSequences::Owned(data) => data,
            _ => panic!("expected owned sequences"),
        };
        assert_eq!(processed.len(), 1);
        let merged = &processed[0];
        assert_eq!(merged, &seq);
        assert!(!merged.first().is_some_and(|b| is_ascii_whitespace(*b)));
        assert!(!merged.last().is_some_and(|b| is_ascii_whitespace(*b)));
        assert!(merged.windows(3).any(|w| w == b"a b"));
    }

    #[test]
    fn unicode_whitespace_combines_multi_byte_runs() {
        let seq = "hi\u{2003}\u{2003}there\u{00A0}you".as_bytes().to_vec();
        let cfg = PreprocessorConfig {
            kind: PreprocessorKind::UnicodeWhitespace,
            split_probability: 1.0,
            seed: Some(7),
        };
        let processed = match apply_preprocessor(&cfg, &[seq]) {
            PreprocessedSequences::Owned(data) => data,
            _ => panic!("expected owned sequences"),
        };
        assert_eq!(
            processed,
            vec![
                "hi".as_bytes().to_vec(),
                "\u{2003}\u{2003}".as_bytes().to_vec(),
                "there".as_bytes().to_vec(),
                "\u{00A0}".as_bytes().to_vec(),
                "you".as_bytes().to_vec()
            ]
        );
    }

    #[test]
    fn unicode_whitespace_merges_do_not_emit_trailing_separators() {
        let seq = "foo\u{2003}bar\u{2003}baz".as_bytes().to_vec();
        let cfg = PreprocessorConfig {
            kind: PreprocessorKind::UnicodeWhitespace,
            split_probability: 0.4,
            seed: Some(77),
        };
        let processed = match apply_preprocessor(&cfg, &[seq]) {
            PreprocessedSequences::Owned(data) => data,
            _ => panic!("expected owned sequences"),
        };

        assert!(processed
            .iter()
            .any(|chunk| chunk_has_mixed_unicode_content(chunk)));
        for chunk in &processed {
            if ends_with_unicode_whitespace(chunk) {
                assert!(
                    chunk_is_unicode_whitespace(chunk),
                    "token {chunk:?} should not have trailing unicode whitespace"
                );
            }
        }
    }

    #[test]
    fn null_delimited_splits_binary_streams() {
        let seq = b"\x00alpha\x00\x00beta".to_vec();
        let cfg = PreprocessorConfig {
            kind: PreprocessorKind::NullDelimited,
            split_probability: 1.0,
            seed: Some(99),
        };
        let processed = match apply_preprocessor(&cfg, &[seq]) {
            PreprocessedSequences::Owned(data) => data,
            _ => panic!("expected owned sequences"),
        };
        assert_eq!(
            processed,
            vec![
                b"\x00".to_vec(),
                b"alpha".to_vec(),
                b"\x00\x00".to_vec(),
                b"beta".to_vec()
            ]
        );
    }

    #[test]
    fn probability_zero_returns_borrowed_sequences() {
        let seq = b"foo bar".to_vec();
        let cfg = PreprocessorConfig {
            kind: PreprocessorKind::AsciiWhitespace,
            split_probability: 0.0,
            seed: None,
        };
        match apply_preprocessor(&cfg, std::slice::from_ref(&seq)) {
            PreprocessedSequences::Borrowed(slice) => assert_eq!(slice[0], seq),
            _ => panic!("probability zero should borrow original data"),
        }
    }

    #[test]
    fn probabilistic_preprocessor_is_seeded() {
        let seq = b"alpha beta gamma".to_vec();
        let cfg = PreprocessorConfig {
            kind: PreprocessorKind::AsciiWhitespace,
            split_probability: 0.5,
            seed: Some(123),
        };
        let processed = match apply_preprocessor(&cfg, &[seq]) {
            PreprocessedSequences::Owned(data) => data,
            _ => panic!("expected owned sequences"),
        };
        assert_eq!(
            processed,
            vec![
                b"alpha".to_vec(),
                b" ".to_vec(),
                b"beta".to_vec(),
                b" ".to_vec(),
                b"gamma".to_vec()
            ]
        );
    }

    #[test]
    fn apply_preprocessor_falls_back_to_borrowed() {
        let seqs = vec![b"foo".to_vec()];
        match apply_preprocessor(&PreprocessorConfig::default(), &seqs) {
            PreprocessedSequences::Borrowed(slice) => assert_eq!(slice.len(), 1),
            _ => panic!("expected borrowed sequences"),
        }
    }

    #[test]
    fn preprocessor_runner_passthrough_when_disabled() {
        let cfg = PreprocessorConfig::default();
        let mut runner = PreprocessorRunner::new(cfg);
        let mut seen = Vec::new();
        let data = b"hello world".to_vec();
        runner.process(&data, |segment| seen.push(segment.to_vec()));
        assert_eq!(seen, vec![b"hello world".to_vec()]);
    }
}
