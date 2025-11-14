//! Preprocessing helpers that transform raw byte sequences before training.

use bstr::ByteSlice;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

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

/// Applies the configured preprocessor to an input corpus, returning either a borrowed or owned
/// slice of sequences depending on whether any transformation was required.
#[must_use]
pub fn apply_preprocessor<'a>(
    cfg: &PreprocessorConfig,
    sequences: &'a [Vec<u8>],
) -> PreprocessedSequences<'a> {
    let probability = cfg.split_probability.clamp(0.0, 1.0);
    if matches!(cfg.kind, PreprocessorKind::None) || probability == 0.0 {
        return PreprocessedSequences::Borrowed(sequences);
    }

    let mut rng = match cfg.seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_entropy(),
    };

    match cfg.kind {
        PreprocessorKind::None => PreprocessedSequences::Borrowed(sequences),
        PreprocessorKind::AsciiWhitespace => {
            PreprocessedSequences::Owned(split_ascii_sequences(sequences, probability, &mut rng))
        }
        PreprocessorKind::UnicodeWhitespace => {
            PreprocessedSequences::Owned(split_unicode_sequences(sequences, probability, &mut rng))
        }
        PreprocessorKind::NullDelimited => {
            PreprocessedSequences::Owned(split_null_delimited(sequences, probability, &mut rng))
        }
    }
}

fn split_ascii_sequences<R: Rng + ?Sized>(
    sequences: &[Vec<u8>],
    probability: f64,
    rng: &mut R,
) -> Vec<Vec<u8>> {
    split_by_predicate(sequences, is_ascii_whitespace, probability, rng)
}

fn split_unicode_sequences<R: Rng + ?Sized>(
    sequences: &[Vec<u8>],
    probability: f64,
    rng: &mut R,
) -> Vec<Vec<u8>> {
    let mut result = Vec::new();
    for seq in sequences {
        split_unicode_runs(seq, probability, rng, &mut result);
    }
    result
}

fn split_null_delimited<R: Rng + ?Sized>(
    sequences: &[Vec<u8>],
    probability: f64,
    rng: &mut R,
) -> Vec<Vec<u8>> {
    split_by_predicate(sequences, |byte| byte == 0, probability, rng)
}

fn split_by_predicate<R, F>(
    sequences: &[Vec<u8>],
    predicate: F,
    probability: f64,
    rng: &mut R,
) -> Vec<Vec<u8>>
where
    R: Rng + ?Sized,
    F: Fn(u8) -> bool + Copy,
{
    let mut result = Vec::new();
    for seq in sequences {
        split_runs(seq, predicate, probability, rng, &mut result);
    }
    result
}

fn is_ascii_whitespace(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\n' | b'\r' | 0x0B | 0x0C)
}

type Segment = (usize, usize, bool);

fn split_runs<R, F>(
    sequence: &[u8],
    predicate: F,
    probability: f64,
    rng: &mut R,
    out: &mut Vec<Vec<u8>>,
) where
    R: Rng + ?Sized,
    F: Fn(u8) -> bool,
{
    if sequence.is_empty() {
        return;
    }
    let mut segments = Vec::new();
    let mut start = 0usize;
    let mut current_kind = predicate(sequence[0]);
    for idx in 1..=sequence.len() {
        let boundary = if idx == sequence.len() {
            true
        } else {
            predicate(sequence[idx]) != current_kind
        };
        if boundary {
            if idx > start {
                segments.push((start, idx, current_kind));
            }
            if idx < sequence.len() {
                start = idx;
                current_kind = predicate(sequence[idx]);
            }
        }
    }
    merge_segments(sequence, segments, probability, rng, out);
}

fn split_unicode_runs<R: Rng + ?Sized>(
    sequence: &[u8],
    probability: f64,
    rng: &mut R,
    out: &mut Vec<Vec<u8>>,
) {
    if sequence.is_empty() {
        return;
    }

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

    merge_segments(sequence, segments, probability, rng, out);
}

fn merge_segments<R: Rng + ?Sized>(
    sequence: &[u8],
    segments: Vec<Segment>,
    probability: f64,
    rng: &mut R,
    out: &mut Vec<Vec<u8>>,
) {
    if segments.is_empty() {
        return;
    }
    let mut buffer = sequence[segments[0].0..segments[0].1].to_vec();
    for (start, end, _) in segments.into_iter().skip(1) {
        let bytes = &sequence[start..end];
        if should_split(probability, rng) {
            out.push(std::mem::take(&mut buffer));
            buffer = bytes.to_vec();
        } else {
            buffer.extend_from_slice(bytes);
        }
    }
    out.push(buffer);
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
        match apply_preprocessor(&cfg, &[seq.clone()]) {
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
                b"beta ".to_vec(),
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
}
