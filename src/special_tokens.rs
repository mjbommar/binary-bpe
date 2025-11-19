//! Helpers for loading and arranging special token inventories.

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use serde::Deserialize;

const SPECIAL_TOKEN_SPEC: &str = include_str!("../references/special_tokens.json");

#[derive(Debug, Clone, Deserialize)]
struct RawSpecialTokens(HashMap<String, u32>);

#[derive(Debug)]
struct SpecialTokenSets {
    leading: Vec<String>,
    reasoning: Vec<String>,
}

static TOKEN_SETS: OnceLock<SpecialTokenSets> = OnceLock::new();

fn token_sets() -> &'static SpecialTokenSets {
    TOKEN_SETS.get_or_init(|| {
        let raw: RawSpecialTokens =
            serde_json::from_str(SPECIAL_TOKEN_SPEC).expect("invalid special token JSON");
        let mut entries: Vec<(String, u32)> = raw.0.into_iter().collect();
        entries.sort_by_key(|(_, id)| *id);

        let mut leading = Vec::new();
        let mut reasoning = Vec::new();
        for (token, id) in entries {
            if id <= 6 {
                leading.push(token);
            } else {
                reasoning.push(token);
            }
        }
        SpecialTokenSets { leading, reasoning }
    })
}

/// Returns the always-on Category 3 tokens which must occupy IDs starting at zero.
pub fn leading_tokens() -> &'static [String] {
    &token_sets().leading
}

/// Returns the Category 4 reasoning/argument tokens in their canonical order.
pub fn reasoning_tokens() -> &'static [String] {
    &token_sets().reasoning
}

/// Builds the trailing special token inventory (placed immediately after the byte alphabet) by
/// optionally including reasoning tokens and then appending any user-provided extras.
pub fn trailing_tokens(reasoning_enabled: bool, extras: &[String]) -> Vec<String> {
    let mut tokens = Vec::new();
    if reasoning_enabled {
        tokens.extend(reasoning_tokens().iter().cloned());
    }
    for token in extras {
        if !tokens.iter().any(|existing| existing == token) {
            tokens.push(token.clone());
        }
    }
    tokens
}

/// Deduplicates tokens in-place while preserving the first occurrence ordering.
pub fn dedup_in_place(tokens: &mut Vec<String>) {
    let mut seen = HashSet::with_capacity(tokens.len());
    tokens.retain(|token| seen.insert(token.clone()));
}
