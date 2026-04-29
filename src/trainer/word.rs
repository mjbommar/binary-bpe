use crate::bytes::is_allowed_length;
use crate::model::{Pair, TokenId};

/// Outcome of merging a specific pair within a [`Word`].
#[derive(Default)]
pub(crate) struct MergeOutcome {
    /// Number of pair occurrences replaced inside the word.
    pub merges: usize,
    /// Pair count deltas emitted by the merge. Negative values represent removals,
    /// positive values represent newly formed adjacent pairs.
    pub deltas: Vec<(Pair, i32)>,
}

#[derive(Clone, Copy)]
struct Symbol {
    token: TokenId,
    len: u32,
}

impl Symbol {
    fn new(token: TokenId, len: usize) -> Self {
        Self {
            token,
            len: len as u32,
        }
    }

    #[inline]
    fn len_usize(self) -> usize {
        self.len as usize
    }
}

/// Compact linked-list style representation of a tokenized sequence.
#[derive(Clone, Default)]
pub(crate) struct Word {
    symbols: Vec<Symbol>,
}

impl Word {
    /// Builds a word from an owned token sequence. Each base token initially spans one byte.
    pub(crate) fn from_tokens(tokens: Vec<TokenId>) -> Self {
        let symbols = tokens
            .into_iter()
            .map(|token| Symbol::new(token, 1))
            .collect();
        Self { symbols }
    }

    /// Returns true when the word contains at least two symbols.
    pub(crate) fn has_pairs(&self) -> bool {
        self.symbols.len() >= 2
    }

    /// Invokes the provided closure for each adjacent token pair whose combined byte length is allowed.
    pub(crate) fn for_each_pair<F>(&self, allowed_lengths: &[usize], mut f: F)
    where
        F: FnMut(Pair),
    {
        for window in self.symbols.windows(2) {
            let combined = window[0].len_usize() + window[1].len_usize();
            if is_allowed_length(combined, allowed_lengths) {
                f((window[0].token, window[1].token));
            }
        }
    }

    /// Applies the selected merge pair throughout the word and returns the resulting deltas.
    ///
    /// Uses two-pointer compaction: a single linear pass over `symbols` rewrites in
    /// place with O(N) work, instead of O(N²) from repeated `Vec::remove`.
    pub(crate) fn merge(
        &mut self,
        left: TokenId,
        right: TokenId,
        replacement: TokenId,
        replacement_len: usize,
        allowed_lengths: &[usize],
    ) -> MergeOutcome {
        let mut outcome = MergeOutcome::default();
        let len = self.symbols.len();
        if len < 2 {
            return outcome;
        }

        let mut write: usize = 0;
        let mut read: usize = 0;

        while read < len {
            // Try to merge at `read`.
            if read + 1 < len
                && self.symbols[read].token == left
                && self.symbols[read + 1].token == right
            {
                let left_symbol = self.symbols[read];
                let right_symbol = self.symbols[read + 1];
                let combined_len = left_symbol.len_usize() + right_symbol.len_usize();
                if is_allowed_length(combined_len, allowed_lengths) {
                    // prev is whatever is currently at write-1 (post any earlier
                    // merges in this same pass), matching the original semantics
                    // where the array shifted left after each remove().
                    let prev_symbol = if write > 0 {
                        Some(self.symbols[write - 1])
                    } else {
                        None
                    };
                    let next_symbol = if read + 2 < len {
                        Some(self.symbols[read + 2])
                    } else {
                        None
                    };

                    if let Some(prev) = prev_symbol {
                        let combined = prev.len_usize() + left_symbol.len_usize();
                        if is_allowed_length(combined, allowed_lengths) {
                            outcome.deltas.push(((prev.token, left_symbol.token), -1));
                        }
                    }
                    outcome
                        .deltas
                        .push(((left_symbol.token, right_symbol.token), -1));
                    if let Some(next) = next_symbol {
                        let combined = right_symbol.len_usize() + next.len_usize();
                        if is_allowed_length(combined, allowed_lengths) {
                            outcome.deltas.push(((right_symbol.token, next.token), -1));
                        }
                    }

                    let merged = Symbol::new(replacement, replacement_len);
                    if let Some(prev) = prev_symbol {
                        let combined = prev.len_usize() + replacement_len;
                        if is_allowed_length(combined, allowed_lengths) {
                            outcome.deltas.push(((prev.token, replacement), 1));
                        }
                    }
                    if let Some(next) = next_symbol {
                        let combined = replacement_len + next.len_usize();
                        if is_allowed_length(combined, allowed_lengths) {
                            outcome.deltas.push(((replacement, next.token), 1));
                        }
                    }

                    self.symbols[write] = merged;
                    write += 1;
                    read += 2;
                    outcome.merges += 1;
                    continue;
                }
            }

            // No merge: copy through.
            if write != read {
                self.symbols[write] = self.symbols[read];
            }
            write += 1;
            read += 1;
        }

        self.symbols.truncate(write);
        outcome
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_replaces_all_pairs() {
        let mut word = Word::from_tokens(vec![1, 2, 1, 2, 3]);
        assert!(word.has_pairs());
        let result = word.merge(1, 2, 99, 2, &[2, 3, 4]);
        assert_eq!(result.merges, 2);
        assert!(result.deltas.contains(&((1, 2), -1)));
    }

    #[test]
    fn merge_respects_allowed_lengths() {
        let mut word = Word::from_tokens(vec![1, 2, 3]);
        let outcome = word.merge(1, 2, 4, 2, &[1]);
        assert_eq!(outcome.merges, 0);
        assert!(outcome.deltas.is_empty());
    }

    #[test]
    fn enumerate_pairs_honors_allowed_lengths() {
        let word = Word::from_tokens(vec![1, 2, 3]);
        let mut pairs = Vec::new();
        word.for_each_pair(&[2], |pair| pairs.push(pair));
        assert_eq!(pairs, vec![(1, 2), (2, 3)]);
    }
}
