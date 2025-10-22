//! Utilities for converting between binary data and Latin-1 text.

/// Converts raw bytes into a Latin-1 string representation used for Hugging Face compatibility.
#[must_use]
pub fn bytes_to_latin1(bytes: &[u8]) -> String {
    bytes.iter().map(|&b| b as char).collect()
}

/// Converts a Latin-1 string produced by `bytes_to_latin1` back to raw bytes.
#[must_use]
pub fn latin1_to_bytes(text: &str) -> Vec<u8> {
    text.chars().map(|c| c as u8).collect()
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
    fn allowed_length_checks_subset() {
        let allowed = [1, 2, 4, 8];
        assert!(is_allowed_length(4, &allowed));
        assert!(!is_allowed_length(3, &allowed));
    }
}
