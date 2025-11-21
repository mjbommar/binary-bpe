//! Utilities for converting between binary data and serialized token strings.

use std::collections::{hash_map::Entry, HashMap, HashSet};
use std::sync::OnceLock;

use crate::special_tokens;

/// Byte-string encodings supported by persisted tokenizers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ByteEncoding {
    /// Legacy Latin-1 mapping with private-use overrides used prior to 0.6.2.
    Legacy,
    /// GPT-2 style byte-level alphabet compatible with Hugging Face's byte fallback logic.
    Gpt2,
}

fn byte_overrides() -> &'static (HashMap<u8, char>, HashMap<char, u8>) {
    static OVERRIDES: OnceLock<(HashMap<u8, char>, HashMap<char, u8>)> = OnceLock::new();
    OVERRIDES.get_or_init(|| {
        let mut forward = HashMap::new();
        let mut reverse = HashMap::new();
        let mut next = 0xE000u32;
        for token in special_tokens::leading_tokens()
            .iter()
            .chain(special_tokens::reasoning_tokens().iter())
        {
            if token.chars().count() == 1 {
                let ch = token.chars().next().unwrap();
                if (ch as u32) <= 0xFF {
                    let byte = ch as u8;
                    if let Entry::Vacant(slot) = forward.entry(byte) {
                        let replacement =
                            std::char::from_u32(next).expect("replacement code point valid");
                        next += 1;
                        slot.insert(replacement);
                        reverse.insert(replacement, byte);
                    }
                }
            }
        }
        (forward, reverse)
    })
}

fn byte_level_tables() -> &'static ([char; 256], HashMap<char, u8>) {
    static TABLES: OnceLock<([char; 256], HashMap<char, u8>)> = OnceLock::new();
    TABLES.get_or_init(|| {
        let mut forward = ['\0'; 256];
        let mut reverse = HashMap::new();
        let mut bs: Vec<u8> = (b'!'..=b'~').collect();
        bs.extend(b'\xA1'..=b'\xAC');
        bs.extend(b'\xAE'..=b'\xFF');
        let mut cs: Vec<u32> = bs.iter().map(|&b| u32::from(b)).collect();
        let mut seen: HashSet<u8> = bs.iter().copied().collect();
        let mut n = 0u32;
        for byte in 0u8..=255 {
            if seen.contains(&byte) {
                continue;
            }
            bs.push(byte);
            cs.push(256 + n);
            seen.insert(byte);
            n += 1;
        }
        for (byte, codepoint) in bs.into_iter().zip(cs.into_iter()) {
            let ch = std::char::from_u32(codepoint).expect("valid byte-level code point");
            forward[byte as usize] = ch;
            reverse.insert(ch, byte);
        }
        (forward, reverse)
    })
}

fn encode_legacy(bytes: &[u8]) -> String {
    let (forward, _) = byte_overrides();
    bytes
        .iter()
        .map(|&b| forward.get(&b).copied().unwrap_or(b as char))
        .collect()
}

fn decode_legacy(text: &str) -> Vec<u8> {
    let (_, reverse) = byte_overrides();
    text.chars()
        .map(|c| reverse.get(&c).copied().unwrap_or(c as u8))
        .collect()
}

fn encode_gpt2(bytes: &[u8]) -> String {
    let (forward, _) = byte_level_tables();
    bytes.iter().map(|&b| forward[b as usize]).collect()
}

fn decode_gpt2(text: &str) -> Vec<u8> {
    let (_, reverse) = byte_level_tables();
    text.chars()
        .map(|c| {
            reverse
                .get(&c)
                .copied()
                .unwrap_or_else(|| panic!("unknown byte-level code point: U+{:04X}", c as u32))
        })
        .collect()
}

/// Converts raw bytes into a serialized string representation.
#[must_use]
pub fn bytes_to_string(bytes: &[u8], encoding: ByteEncoding) -> String {
    match encoding {
        ByteEncoding::Legacy => encode_legacy(bytes),
        ByteEncoding::Gpt2 => encode_gpt2(bytes),
    }
}

/// Converts a serialized string produced by `bytes_to_string` back to raw bytes.
#[must_use]
pub fn string_to_bytes(text: &str, encoding: ByteEncoding) -> Vec<u8> {
    match encoding {
        ByteEncoding::Legacy => decode_legacy(text),
        ByteEncoding::Gpt2 => decode_gpt2(text),
    }
}

/// Backwards-compatible alias for the default (GPT-2) serialization.
#[must_use]
pub fn bytes_to_latin1(bytes: &[u8]) -> String {
    bytes_to_string(bytes, ByteEncoding::Gpt2)
}

/// Backwards-compatible alias for decoding the default serialization.
#[must_use]
pub fn latin1_to_bytes(text: &str) -> Vec<u8> {
    string_to_bytes(text, ByteEncoding::Gpt2)
}

/// Converts bytes using the legacy Latin-1 mapping.
#[must_use]
pub fn legacy_bytes_to_latin1(bytes: &[u8]) -> String {
    bytes_to_string(bytes, ByteEncoding::Legacy)
}

/// Converts legacy Latin-1 strings back to bytes.
#[must_use]
pub fn legacy_latin1_to_bytes(text: &str) -> Vec<u8> {
    string_to_bytes(text, ByteEncoding::Legacy)
}

/// Returns `true` for the ASCII whitespace bytes we treat as delimiters.
#[inline]
#[must_use]
pub fn is_ascii_whitespace(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\n' | b'\r' | 0x0B | 0x0C)
}

/// Returns `true` when *all* bytes in the slice are ASCII whitespace.
#[inline]
#[must_use]
pub fn is_all_ascii_whitespace(bytes: &[u8]) -> bool {
    !bytes.is_empty() && bytes.iter().all(|&b| is_ascii_whitespace(b))
}

/// Returns `true` when the slice ends with an ASCII whitespace byte.
#[inline]
#[must_use]
pub fn ends_with_ascii_whitespace(bytes: &[u8]) -> bool {
    bytes.last().copied().is_some_and(is_ascii_whitespace)
}

/// Returns `true` when the slice starts with ASCII whitespace.
#[inline]
#[must_use]
pub fn starts_with_ascii_whitespace(bytes: &[u8]) -> bool {
    bytes.first().copied().is_some_and(is_ascii_whitespace)
}

/// Returns `true` when at least one ASCII alphabetic byte is present.
#[inline]
#[must_use]
pub fn contains_ascii_letter(bytes: &[u8]) -> bool {
    bytes.iter().any(|&b| b.is_ascii_alphabetic())
}

/// Returns `true` when the provided length is included in the allowed merge lengths list.
#[inline]
#[must_use]
pub fn is_allowed_length(len: usize, allowed: &[usize]) -> bool {
    allowed.contains(&len)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latin1_transcoding_round_trip() {
        let bytes: Vec<u8> = (0..=u8::MAX).collect();
        let latin1 = bytes_to_latin1(&bytes);
        let restored = latin1_to_bytes(&latin1);
        assert_eq!(restored, bytes);
    }

    #[test]
    fn legacy_round_trip() {
        let bytes: Vec<u8> = (0..=u8::MAX).collect();
        let latin1 = legacy_bytes_to_latin1(&bytes);
        let restored = legacy_latin1_to_bytes(&latin1);
        assert_eq!(restored, bytes);
    }

    #[test]
    fn allowed_length_checks_subset() {
        let allowed = [1, 2, 4, 8];
        assert!(is_allowed_length(4, &allowed));
        assert!(!is_allowed_length(3, &allowed));
    }

    #[test]
    fn ascii_whitespace_helpers() {
        assert!(is_ascii_whitespace(b' '));
        assert!(is_ascii_whitespace(b'\t'));
        assert!(ends_with_ascii_whitespace(b"foo \r"));
        assert!(starts_with_ascii_whitespace(b" foo"));
        assert!(is_all_ascii_whitespace(b"\t\n"));
        assert!(!is_all_ascii_whitespace(b"foo"));
        assert!(!ends_with_ascii_whitespace(b"foo"));
        assert!(!starts_with_ascii_whitespace(b"foo"));
        assert!(contains_ascii_letter(b"foo"));
        assert!(!contains_ascii_letter(b"123"));
    }
}
