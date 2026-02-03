//! Search operations for finding set bits in bitmap.

use crate::atomic::{AtomicU64, Ordering};
use crate::bitmap::{leading_zeros, popcount, trailing_zeros};

/// Find minimum set bit.
///
/// # Arguments
/// * `bitmap` - Reference to 4-word atomic bitmap
///
/// # Returns
/// Index of minimum set bit, or None if bitmap is empty
///
/// # Performance
/// O(1) - uses CPU intrinsics (TZCNT) for fast bit scanning
#[inline]
pub fn min_bit(bitmap: &[AtomicU64; 4]) -> Option<u8> {
    for (word_idx, word) in bitmap.iter().enumerate() {
        let value = word.load(Ordering::Acquire);
        if value != 0 {
            let bit_in_word = trailing_zeros(value) as usize;
            return Some((word_idx * 64 + bit_in_word) as u8);
        }
    }
    None
}

/// Find maximum set bit.
///
/// # Arguments
/// * `bitmap` - Reference to 4-word atomic bitmap
///
/// # Returns
/// Index of maximum set bit, or None if bitmap is empty
///
/// # Performance
/// O(1) - uses CPU intrinsics (LZCNT) for fast bit scanning
#[inline]
pub fn max_bit(bitmap: &[AtomicU64; 4]) -> Option<u8> {
    for (word_idx, word) in bitmap.iter().enumerate().rev() {
        let value = word.load(Ordering::Acquire);
        if value != 0 {
            let bit_in_word = 63 - leading_zeros(value) as usize;
            return Some((word_idx * 64 + bit_in_word) as u8);
        }
    }
    None
}

/// Count all set bits in bitmap.
///
/// # Arguments
/// * `bitmap` - Reference to 4-word atomic bitmap
///
/// # Returns
/// Total number of set bits (0-256)
///
/// # Performance
/// O(1) - uses CPU POPCNT instruction for fast counting
#[inline]
#[allow(dead_code)]
pub fn count_bits(bitmap: &[AtomicU64; 4]) -> u32 {
    let w0 = bitmap[0].load(Ordering::Acquire);
    let w1 = bitmap[1].load(Ordering::Acquire);
    let w2 = bitmap[2].load(Ordering::Acquire);
    let w3 = bitmap[3].load(Ordering::Acquire);
    popcount(w0) + popcount(w1) + popcount(w2) + popcount(w3)
}

/// Find next set bit after the given index.
///
/// # Arguments
/// * `bitmap` - Reference to 4-word atomic bitmap
/// * `after` - Index to search after (0-255)
///
/// # Returns
/// Index of next set bit, or None if no set bits found
///
/// # Performance
/// O(1) - uses CPU intrinsics (TZCNT) for fast bit scanning
#[inline]
pub fn next_set_bit(bitmap: &[AtomicU64; 4], after: u8) -> Option<u8> {
    let start = after as usize + 1;
    if start >= 256 {
        return None;
    }

    let start_word = start / 64;
    let start_bit = start % 64;

    // Check remaining bits in start word
    let mask = !((1u64 << start_bit) - 1);
    let value = bitmap[start_word].load(Ordering::Acquire);
    let masked = value & mask;
    if masked != 0 {
        let bit_in_word = trailing_zeros(masked) as usize;
        return Some((start_word * 64 + bit_in_word) as u8);
    }

    // Check subsequent words
    #[allow(clippy::needless_range_loop)]
    for word_idx in (start_word + 1)..4 {
        let value = bitmap[word_idx].load(Ordering::Acquire);
        if value != 0 {
            let bit_in_word = trailing_zeros(value) as usize;
            return Some((word_idx * 64 + bit_in_word) as u8);
        }
    }

    None
}

/// Find previous set bit before the given index.
///
/// # Arguments
/// * `bitmap` - Reference to 4-word atomic bitmap
/// * `before` - Index to search before (0-255)
///
/// # Returns
/// Index of previous set bit, or None if no set bits found
///
/// # Performance
/// O(1) - uses CPU intrinsics (LZCNT) for fast bit scanning
#[inline]
pub fn prev_set_bit(bitmap: &[AtomicU64; 4], before: u8) -> Option<u8> {
    if before == 0 {
        return None;
    }

    let end = (before as usize).saturating_sub(1);
    let end_word = end / 64;
    let end_bit = end % 64;

    // Check bits up to end_bit in end word
    let mask = (1u64 << (end_bit + 1)) - 1;
    let value = bitmap[end_word].load(Ordering::Acquire);
    let masked = value & mask;
    if masked != 0 {
        let bit_in_word = 63 - leading_zeros(masked) as usize;
        return Some((end_word * 64 + bit_in_word) as u8);
    }

    // Check previous words
    for word_idx in (0..end_word).rev() {
        let value = bitmap[word_idx].load(Ordering::Acquire);
        if value != 0 {
            let bit_in_word = 63 - leading_zeros(value) as usize;
            return Some((word_idx * 64 + bit_in_word) as u8);
        }
    }

    None
}

/// Count set bits in range [from, to).
///
/// # Arguments
/// * `bitmap` - Reference to 4-word atomic bitmap
/// * `from` - Start index (inclusive, 0-255)
/// * `to` - End index (exclusive, 0-256)
///
/// # Returns
/// Number of set bits in range
///
/// # Performance
/// O(1) - uses CPU POPCNT instruction for fast counting
#[inline]
#[allow(dead_code)]
pub fn count_range(bitmap: &[AtomicU64; 4], from: u8, to: u16) -> u32 {
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
        let value = bitmap[from_word].load(Ordering::Acquire);
        return popcount(value & mask);
    }

    let mut count = 0u32;

    // First word: from_bit to 63
    let from_bit = from % 64;
    let first_mask = !0u64 << from_bit;
    let first_value = bitmap[from_word].load(Ordering::Acquire);
    count += popcount(first_value & first_mask);

    // Middle words: all bits
    #[allow(clippy::needless_range_loop)]
    for w in (from_word + 1)..to_word {
        count += popcount(bitmap[w].load(Ordering::Acquire));
    }

    // Last word: 0 to to_bit
    let to_bit = to % 64;
    if to_bit > 0 {
        let last_mask = (1u64 << to_bit) - 1;
        let last_value = bitmap[to_word].load(Ordering::Acquire);
        count += popcount(last_value & last_mask);
    } else if to_word < 4 {
        count += popcount(bitmap[to_word].load(Ordering::Acquire));
    }

    count
}

/// Find minimum set bit with seqlock protection.
///
/// Uses seqlock protocol to ensure consistent read during concurrent bulk modifications.
///
/// # Arguments
/// * `seq` - Sequence counter for seqlock protocol
/// * `bitmap` - Reference to 4-word atomic bitmap
///
/// # Returns
/// Index of minimum set bit, or None if bitmap is empty
///
/// # Performance
/// O(1) - ~5% overhead from seqlock without collisions
///
/// # Thread Safety
/// Lock-free read operation. Retries if concurrent bulk modification detected.
#[inline]
#[allow(dead_code)]
pub fn min_bit_seqlock(seq: &AtomicU64, bitmap: &[AtomicU64; 4]) -> Option<u8> {
    loop {
        let seq_before = seq.load(Ordering::Acquire);
        if seq_before & 1 != 0 {
            core::hint::spin_loop();
            continue;
        }
        let result = min_bit(bitmap);
        let seq_after = seq.load(Ordering::Acquire);
        if seq_before == seq_after {
            return result;
        }
    }
}

/// Find maximum set bit with seqlock protection.
///
/// Uses seqlock protocol to ensure consistent read during concurrent bulk modifications.
///
/// # Arguments
/// * `seq` - Sequence counter for seqlock protocol
/// * `bitmap` - Reference to 4-word atomic bitmap
///
/// # Returns
/// Index of maximum set bit, or None if bitmap is empty
///
/// # Performance
/// O(1) - ~5% overhead from seqlock without collisions
///
/// # Thread Safety
/// Lock-free read operation. Retries if concurrent bulk modification detected.
#[inline]
#[allow(dead_code)]
pub fn max_bit_seqlock(seq: &AtomicU64, bitmap: &[AtomicU64; 4]) -> Option<u8> {
    loop {
        let seq_before = seq.load(Ordering::Acquire);
        if seq_before & 1 != 0 {
            core::hint::spin_loop();
            continue;
        }
        let result = max_bit(bitmap);
        let seq_after = seq.load(Ordering::Acquire);
        if seq_before == seq_after {
            return result;
        }
    }
}

/// Count all set bits with seqlock protection.
///
/// Uses seqlock protocol to ensure consistent read during concurrent bulk modifications.
///
/// # Arguments
/// * `seq` - Sequence counter for seqlock protocol
/// * `bitmap` - Reference to 4-word atomic bitmap
///
/// # Returns
/// Total number of set bits (0-256)
///
/// # Performance
/// O(1) - ~5% overhead from seqlock without collisions
///
/// # Thread Safety
/// Lock-free read operation. Retries if concurrent bulk modification detected.
#[inline]
#[allow(dead_code)]
pub fn count_bits_seqlock(seq: &AtomicU64, bitmap: &[AtomicU64; 4]) -> u32 {
    loop {
        let seq_before = seq.load(Ordering::Acquire);
        if seq_before & 1 != 0 {
            core::hint::spin_loop();
            continue;
        }
        let result = count_bits(bitmap);
        let seq_after = seq.load(Ordering::Acquire);
        if seq_before == seq_after {
            return result;
        }
    }
}

/// Find next set bit after the given index with seqlock protection.
///
/// Uses seqlock protocol to ensure consistent read during concurrent bulk modifications.
///
/// # Arguments
/// * `seq` - Sequence counter for seqlock protocol
/// * `bitmap` - Reference to 4-word atomic bitmap
/// * `after` - Index to search after (0-255)
///
/// # Returns
/// Index of next set bit, or None if no set bits found
///
/// # Performance
/// O(1) - ~5% overhead from seqlock without collisions
///
/// # Thread Safety
/// Lock-free read operation. Retries if concurrent bulk modification detected.
#[inline]
#[allow(dead_code)]
pub fn next_set_bit_seqlock(seq: &AtomicU64, bitmap: &[AtomicU64; 4], after: u8) -> Option<u8> {
    loop {
        let seq_before = seq.load(Ordering::Acquire);
        if seq_before & 1 != 0 {
            core::hint::spin_loop();
            continue;
        }
        let result = next_set_bit(bitmap, after);
        let seq_after = seq.load(Ordering::Acquire);
        if seq_before == seq_after {
            return result;
        }
    }
}

/// Find previous set bit before the given index with seqlock protection.
///
/// Uses seqlock protocol to ensure consistent read during concurrent bulk modifications.
///
/// # Arguments
/// * `seq` - Sequence counter for seqlock protocol
/// * `bitmap` - Reference to 4-word atomic bitmap
/// * `before` - Index to search before (0-255)
///
/// # Returns
/// Index of previous set bit, or None if no set bits found
///
/// # Performance
/// O(1) - ~5% overhead from seqlock without collisions
///
/// # Thread Safety
/// Lock-free read operation. Retries if concurrent bulk modification detected.
#[inline]
#[allow(dead_code)]
pub fn prev_set_bit_seqlock(seq: &AtomicU64, bitmap: &[AtomicU64; 4], before: u8) -> Option<u8> {
    loop {
        let seq_before = seq.load(Ordering::Acquire);
        if seq_before & 1 != 0 {
            core::hint::spin_loop();
            continue;
        }
        let result = prev_set_bit(bitmap, before);
        let seq_after = seq.load(Ordering::Acquire);
        if seq_before == seq_after {
            return result;
        }
    }
}

/// Count set bits in range [from, to) with seqlock protection.
///
/// Uses seqlock protocol to ensure consistent read during concurrent bulk modifications.
///
/// # Arguments
/// * `seq` - Sequence counter for seqlock protocol
/// * `bitmap` - Reference to 4-word atomic bitmap
/// * `from` - Start index (inclusive, 0-255)
/// * `to` - End index (exclusive, 0-256)
///
/// # Returns
/// Number of set bits in range
///
/// # Performance
/// O(1) - ~5% overhead from seqlock without collisions
///
/// # Thread Safety
/// Lock-free read operation. Retries if concurrent bulk modification detected.
#[inline]
#[allow(dead_code)]
pub fn count_range_seqlock(seq: &AtomicU64, bitmap: &[AtomicU64; 4], from: u8, to: u16) -> u32 {
    loop {
        let seq_before = seq.load(Ordering::Acquire);
        if seq_before & 1 != 0 {
            core::hint::spin_loop();
            continue;
        }
        let result = count_range(bitmap, from, to);
        let seq_after = seq.load(Ordering::Acquire);
        if seq_before == seq_after {
            return result;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitmap::{set_bit, set_range};

    #[test]
    fn test_min_bit() {
        const INIT: AtomicU64 = AtomicU64::new(0);

        // Empty bitmap
        let bitmap = [INIT; 4];
        assert_eq!(min_bit(&bitmap), None);

        // First word
        let bitmap = [INIT; 4];
        set_bit(&bitmap, 5);
        assert_eq!(min_bit(&bitmap), Some(5));

        // Multiple bits - should return first
        set_bit(&bitmap, 10);
        set_bit(&bitmap, 100);
        assert_eq!(min_bit(&bitmap), Some(5));

        // Second word
        let bitmap = [INIT; 4];
        set_bit(&bitmap, 67);
        assert_eq!(min_bit(&bitmap), Some(67));

        // Last word
        let bitmap = [INIT; 4];
        set_bit(&bitmap, 255);
        assert_eq!(min_bit(&bitmap), Some(255));
    }

    #[test]
    fn test_max_bit() {
        const INIT: AtomicU64 = AtomicU64::new(0);

        // Empty bitmap
        let bitmap = [INIT; 4];
        assert_eq!(max_bit(&bitmap), None);

        // Last word
        let bitmap = [INIT; 4];
        set_bit(&bitmap, 200);
        assert_eq!(max_bit(&bitmap), Some(200));

        // Multiple bits - should return last
        set_bit(&bitmap, 5);
        set_bit(&bitmap, 100);
        assert_eq!(max_bit(&bitmap), Some(200));

        // First word
        let bitmap = [INIT; 4];
        set_bit(&bitmap, 10);
        assert_eq!(max_bit(&bitmap), Some(10));

        // Bit 255 (last possible)
        let bitmap = [INIT; 4];
        set_bit(&bitmap, 255);
        assert_eq!(max_bit(&bitmap), Some(255));
    }

    #[test]
    fn test_count_bits() {
        const INIT_ZERO: AtomicU64 = AtomicU64::new(0);
        const INIT_FULL: AtomicU64 = AtomicU64::new(!0);

        // Empty bitmap
        let bitmap = [INIT_ZERO; 4];
        assert_eq!(count_bits(&bitmap), 0);

        // Single bit
        let bitmap = [INIT_ZERO; 4];
        set_bit(&bitmap, 42);
        assert_eq!(count_bits(&bitmap), 1);

        // Multiple bits
        set_bit(&bitmap, 5);
        set_bit(&bitmap, 100);
        set_bit(&bitmap, 200);
        assert_eq!(count_bits(&bitmap), 4);

        // Range
        let bitmap = [INIT_ZERO; 4];
        set_range(&bitmap, 10, 20);
        assert_eq!(count_bits(&bitmap), 10);

        // Full bitmap
        let bitmap = [INIT_FULL; 4];
        assert_eq!(count_bits(&bitmap), 256);
    }

    #[test]
    fn test_next_set_bit() {
        const INIT: AtomicU64 = AtomicU64::new(0);
        let bitmap = [INIT; 4];

        // Set some bits
        set_bit(&bitmap, 5);
        set_bit(&bitmap, 67);
        set_bit(&bitmap, 200);

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
        const INIT: AtomicU64 = AtomicU64::new(0);
        let bitmap = [INIT; 4];

        // Set some bits
        set_bit(&bitmap, 5);
        set_bit(&bitmap, 67);
        set_bit(&bitmap, 200);

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
        const INIT: AtomicU64 = AtomicU64::new(0);
        let bitmap = [INIT; 4];

        // Set range [10, 20)
        set_range(&bitmap, 10, 20);

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
        let bitmap = [INIT; 4];
        set_range(&bitmap, 60, 70);
        assert_eq!(count_range(&bitmap, 60, 70), 10);

        // Empty range
        assert_eq!(count_range(&bitmap, 10, 10), 0);
    }
}
