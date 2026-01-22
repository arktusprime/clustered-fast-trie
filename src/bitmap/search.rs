//! Search operations for finding set bits in bitmap.

use crate::bitmap::{leading_zeros, popcount, trailing_zeros};

/// Find first set bit (minimum).
///
/// # Arguments
/// * `bitmap` - Reference to 4-word bitmap
///
/// # Returns
/// Index of first set bit, or None if bitmap is empty
///
/// # Performance
/// O(1) - uses CPU intrinsics (TZCNT) for fast bit scanning
#[inline]
pub fn first_set_bit(bitmap: &[u64; 4]) -> Option<u8> {
    for (word_idx, &word) in bitmap.iter().enumerate() {
        if word != 0 {
            let bit_in_word = trailing_zeros(word) as usize;
            return Some((word_idx * 64 + bit_in_word) as u8);
        }
    }
    None
}

/// Find last set bit (maximum).
///
/// # Arguments
/// * `bitmap` - Reference to 4-word bitmap
///
/// # Returns
/// Index of last set bit, or None if bitmap is empty
///
/// # Performance
/// O(1) - uses CPU intrinsics (LZCNT) for fast bit scanning
#[inline]
pub fn last_set_bit(bitmap: &[u64; 4]) -> Option<u8> {
    for (word_idx, &word) in bitmap.iter().enumerate().rev() {
        if word != 0 {
            let bit_in_word = 63 - leading_zeros(word) as usize;
            return Some((word_idx * 64 + bit_in_word) as u8);
        }
    }
    None
}

/// Count all set bits in bitmap.
///
/// # Arguments
/// * `bitmap` - Reference to 4-word bitmap
///
/// # Returns
/// Total number of set bits (0-256)
///
/// # Performance
/// O(1) - uses CPU POPCNT instruction for fast counting
#[inline]
pub fn count_all(bitmap: &[u64; 4]) -> u32 {
    popcount(bitmap[0]) + popcount(bitmap[1]) + popcount(bitmap[2]) + popcount(bitmap[3])
}

/// Find next set bit after the given index.
///
/// # Arguments
/// * `bitmap` - Reference to 4-word bitmap
/// * `after` - Index to search after (0-255)
///
/// # Returns
/// Index of next set bit, or None if no set bits found
///
/// # Performance
/// O(1) - uses CPU intrinsics (TZCNT) for fast bit scanning
#[inline]
pub fn next_set_bit(bitmap: &[u64; 4], after: u8) -> Option<u8> {
    let start = after as usize + 1;
    if start >= 256 {
        return None;
    }

    let start_word = start / 64;
    let start_bit = start % 64;

    // Check remaining bits in start word
    if start_bit > 0 {
        let mask = !((1u64 << start_bit) - 1);
        let masked = bitmap[start_word] & mask;
        if masked != 0 {
            let bit_in_word = trailing_zeros(masked) as usize;
            return Some((start_word * 64 + bit_in_word) as u8);
        }
    }

    // Check subsequent words
    for word_idx in (start_word + 1)..4 {
        if bitmap[word_idx] != 0 {
            let bit_in_word = trailing_zeros(bitmap[word_idx]) as usize;
            return Some((word_idx * 64 + bit_in_word) as u8);
        }
    }

    None
}

/// Find previous set bit before the given index.
///
/// # Arguments
/// * `bitmap` - Reference to 4-word bitmap
/// * `before` - Index to search before (0-255)
///
/// # Returns
/// Index of previous set bit, or None if no set bits found
///
/// # Performance
/// O(1) - uses CPU intrinsics (LZCNT) for fast bit scanning
#[inline]
pub fn prev_set_bit(bitmap: &[u64; 4], before: u8) -> Option<u8> {
    if before == 0 {
        return None;
    }

    let end = (before as usize).saturating_sub(1);
    let end_word = end / 64;
    let end_bit = end % 64;

    // Check bits up to end_bit in end word
    let mask = (1u64 << (end_bit + 1)) - 1;
    let masked = bitmap[end_word] & mask;
    if masked != 0 {
        let bit_in_word = 63 - leading_zeros(masked) as usize;
        return Some((end_word * 64 + bit_in_word) as u8);
    }

    // Check previous words
    for word_idx in (0..end_word).rev() {
        if bitmap[word_idx] != 0 {
            let bit_in_word = 63 - leading_zeros(bitmap[word_idx]) as usize;
            return Some((word_idx * 64 + bit_in_word) as u8);
        }
    }

    None
}

/// Count set bits in range [from, to).
///
/// # Arguments
/// * `bitmap` - Reference to 4-word bitmap
/// * `from` - Start index (inclusive, 0-255)
/// * `to` - End index (exclusive, 0-256)
///
/// # Returns
/// Number of set bits in range
///
/// # Performance
/// O(1) - uses CPU POPCNT instruction for fast counting
#[inline]
pub fn count_range(bitmap: &[u64; 4], from: u8, to: u16) -> u32 {
    if from as u16 >= to {
        return 0;
    }

    let to = to.min(256) as usize;
    let from = from as usize;

    let from_word = from / 64;
    let to_word = (to - 1) / 64;

    if from_word == to_word {
        // Same word
        let from_bit = from % 64;
        let to_bit = to % 64;
        let mask = if to_bit == 0 {
            !0u64 << from_bit
        } else {
            ((!0u64) << from_bit) & ((1u64 << to_bit) - 1)
        };
        return popcount(bitmap[from_word] & mask);
    }

    let mut count = 0u32;

    // First word: from_bit to 63
    let from_bit = from % 64;
    let first_mask = !0u64 << from_bit;
    count += popcount(bitmap[from_word] & first_mask);

    // Middle words: all bits
    for w in (from_word + 1)..to_word {
        count += popcount(bitmap[w]);
    }

    // Last word: 0 to to_bit
    let to_bit = to % 64;
    if to_bit > 0 {
        let last_mask = (1u64 << to_bit) - 1;
        count += popcount(bitmap[to_word] & last_mask);
    } else if to_word < 4 {
        count += popcount(bitmap[to_word]);
    }

    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitmap::{set_bit, set_range};

    #[test]
    fn test_first_set_bit() {
        // Empty bitmap
        let bitmap = [0u64; 4];
        assert_eq!(first_set_bit(&bitmap), None);

        // First word
        let mut bitmap = [0u64; 4];
        set_bit(&mut bitmap, 5);
        assert_eq!(first_set_bit(&bitmap), Some(5));

        // Multiple bits - should return first
        set_bit(&mut bitmap, 10);
        set_bit(&mut bitmap, 100);
        assert_eq!(first_set_bit(&bitmap), Some(5));

        // Second word
        let mut bitmap = [0u64; 4];
        set_bit(&mut bitmap, 67);
        assert_eq!(first_set_bit(&bitmap), Some(67));

        // Last word
        let mut bitmap = [0u64; 4];
        set_bit(&mut bitmap, 255);
        assert_eq!(first_set_bit(&bitmap), Some(255));
    }

    #[test]
    fn test_last_set_bit() {
        // Empty bitmap
        let bitmap = [0u64; 4];
        assert_eq!(last_set_bit(&bitmap), None);

        // Last word
        let mut bitmap = [0u64; 4];
        set_bit(&mut bitmap, 200);
        assert_eq!(last_set_bit(&bitmap), Some(200));

        // Multiple bits - should return last
        set_bit(&mut bitmap, 5);
        set_bit(&mut bitmap, 100);
        assert_eq!(last_set_bit(&bitmap), Some(200));

        // First word
        let mut bitmap = [0u64; 4];
        set_bit(&mut bitmap, 10);
        assert_eq!(last_set_bit(&bitmap), Some(10));

        // Bit 255 (last possible)
        let mut bitmap = [0u64; 4];
        set_bit(&mut bitmap, 255);
        assert_eq!(last_set_bit(&bitmap), Some(255));
    }

    #[test]
    fn test_count_all() {
        // Empty bitmap
        let bitmap = [0u64; 4];
        assert_eq!(count_all(&bitmap), 0);

        // Single bit
        let mut bitmap = [0u64; 4];
        set_bit(&mut bitmap, 42);
        assert_eq!(count_all(&bitmap), 1);

        // Multiple bits
        set_bit(&mut bitmap, 5);
        set_bit(&mut bitmap, 100);
        set_bit(&mut bitmap, 200);
        assert_eq!(count_all(&bitmap), 4);

        // Range
        let mut bitmap = [0u64; 4];
        set_range(&mut bitmap, 10, 20);
        assert_eq!(count_all(&bitmap), 10);

        // Full bitmap
        let bitmap = [!0u64; 4];
        assert_eq!(count_all(&bitmap), 256);
    }

    #[test]
    fn test_next_set_bit() {
        let mut bitmap = [0u64; 4];

        // Set some bits
        set_bit(&mut bitmap, 5);
        set_bit(&mut bitmap, 67);
        set_bit(&mut bitmap, 200);

        // Find next bits
        assert_eq!(next_set_bit(&bitmap, 0), Some(5));
        assert_eq!(next_set_bit(&bitmap, 5), Some(67));
        assert_eq!(next_set_bit(&bitmap, 67), Some(200));
        assert_eq!(next_set_bit(&bitmap, 200), None);

        // Edge cases
        assert_eq!(next_set_bit(&bitmap, 255), None);
        assert_eq!(next_set_bit(&bitmap, 4), Some(5));
        assert_eq!(next_set_bit(&bitmap, 66), Some(67));
    }

    #[test]
    fn test_prev_set_bit() {
        let mut bitmap = [0u64; 4];

        // Set some bits
        set_bit(&mut bitmap, 5);
        set_bit(&mut bitmap, 67);
        set_bit(&mut bitmap, 200);

        // Find previous bits
        assert_eq!(prev_set_bit(&bitmap, 201), Some(200));
        assert_eq!(prev_set_bit(&bitmap, 200), Some(67));
        assert_eq!(prev_set_bit(&bitmap, 67), Some(5));
        assert_eq!(prev_set_bit(&bitmap, 5), None);

        // Edge cases
        assert_eq!(prev_set_bit(&bitmap, 0), None);
        assert_eq!(prev_set_bit(&bitmap, 6), Some(5));
        assert_eq!(prev_set_bit(&bitmap, 68), Some(67));
    }

    #[test]
    fn test_count_range() {
        let mut bitmap = [0u64; 4];

        // Set range [10, 20)
        set_range(&mut bitmap, 10, 20);

        // Count exact range
        assert_eq!(count_range(&bitmap, 10, 20), 10);

        // Count subrange
        assert_eq!(count_range(&bitmap, 12, 18), 6);

        // Count outside range
        assert_eq!(count_range(&bitmap, 0, 10), 0);
        assert_eq!(count_range(&bitmap, 20, 30), 0);

        // Count overlapping
        assert_eq!(count_range(&bitmap, 5, 15), 5);
        assert_eq!(count_range(&bitmap, 15, 25), 5);

        // Cross word boundary
        let mut bitmap = [0u64; 4];
        set_range(&mut bitmap, 60, 70);
        assert_eq!(count_range(&bitmap, 60, 70), 10);

        // Empty range
        assert_eq!(count_range(&bitmap, 10, 10), 0);
    }
}
