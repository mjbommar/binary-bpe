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
    len: usize,
}

impl Symbol {
    fn new(token: TokenId, len: usize) -> Self {
        Self { token, len }
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
            let combined = window[0].len + window[1].len;
            if is_allowed_length(combined, allowed_lengths) {
                f((window[0].token, window[1].token));
            }
        }
    }

    /// Applies the selected merge pair throughout the word and returns the resulting deltas.
    pub(crate) fn merge(
        &mut self,
        left: TokenId,
        right: TokenId,
        replacement: TokenId,
        replacement_len: usize,
        allowed_lengths: &[usize],
    ) -> MergeOutcome {
        let mut outcome = MergeOutcome::default();
        if self.symbols.len() < 2 {
            return outcome;
        }

        let mut i = 0usize;
        while i + 1 < self.symbols.len() {
            if self.symbols[i].token == left && self.symbols[i + 1].token == right {
                let left_symbol = self.symbols[i];
                let right_symbol = self.symbols[i + 1];
                let combined_len = left_symbol.len + right_symbol.len;
                if !is_allowed_length(combined_len, allowed_lengths) {
                    i += 1;
                    continue;
                }

                let prev_symbol = if i > 0 {
                    Some(self.symbols[i - 1])
                } else {
                    None
                };
                let next_symbol = if i + 2 < self.symbols.len() {
                    Some(self.symbols[i + 2])
                } else {
                    None
                };

                // Remove affected adjacency counts.
                if let Some(prev) = prev_symbol {
                    let combined = prev.len + left_symbol.len;
                    if is_allowed_length(combined, allowed_lengths) {
                        outcome.deltas.push(((prev.token, left_symbol.token), -1));
                    }
                }
                outcome
                    .deltas
                    .push(((left_symbol.token, right_symbol.token), -1));
                if let Some(next) = next_symbol {
                    let combined = right_symbol.len + next.len;
                    if is_allowed_length(combined, allowed_lengths) {
                        outcome.deltas.push(((right_symbol.token, next.token), -1));
                    }
                }

                // Merge the pair in place.
                self.symbols[i] = Symbol::new(replacement, replacement_len);
                self.symbols.remove(i + 1);
                outcome.merges += 1;

                // Emit adjacencies formed with the merged token.
                if let Some(prev) = prev_symbol {
                    let combined = prev.len + replacement_len;
                    if is_allowed_length(combined, allowed_lengths) {
                        outcome.deltas.push(((prev.token, replacement), 1));
                    }
                }
                if let Some(next) = next_symbol {
                    let combined = replacement_len + next.len;
                    if is_allowed_length(combined, allowed_lengths) {
                        outcome.deltas.push(((replacement, next.token), 1));
                    }
                }
            }
            i += 1;
        }

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
        let mut collected = Vec::new();
        word.for_each_pair(&[3], |pair| collected.push(pair));
        assert!(collected.is_empty());
        word.for_each_pair(&[2], |pair| collected.push(pair));
        assert_eq!(collected.len(), 2);
    }
}
