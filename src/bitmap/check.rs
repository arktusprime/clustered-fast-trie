//! Check operations for validating bitmap state.

/// Check if all bits in range [from, to) are set.
///
/// # Arguments
/// * `bitmap` - Reference to 4-word bitmap
/// * `from` - Start index (inclusive, 0-255)
/// * `to` - End index (exclusive, 0-256)
///
/// # Returns
/// `true` if all bits in range are set, `false` otherwise
///
/// # Performance
/// O(1) - processes up to 4 words with bitwise operations
#[inline]
pub fn is_range_set(bitmap: &[u64; 4], from: u8, to: u16) -> bool {
    if from as u16 >= to {
        return true; // Empty range
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
        return (bitmap[from_word] & mask) == mask;
    }

    // First word: from_bit to 63
    let from_bit = from % 64;
    let first_mask = !0u64 << from_bit;
    if (bitmap[from_word] & first_mask) != first_mask {
        return false;
    }

    // Middle words: all bits
    for w in (from_word + 1)..to_word {
        if bitmap[w] != !0u64 {
            return false;
        }
    }

    // Last word: 0 to to_bit
    let to_bit = to % 64;
    if to_bit > 0 {
        let last_mask = (1u64 << to_bit) - 1;
        if (bitmap[to_word] & last_mask) != last_mask {
            return false;
        }
    } else if to_word < 4 {
        if bitmap[to_word] != !0u64 {
            return false;
        }
    }

    true
}

/// Check if all specified bits are set.
///
/// # Arguments
/// * `bitmap` - Reference to 4-word bitmap
/// * `indices` - Slice of bit indices to check (0-255)
///
/// # Returns
/// `true` if all specified bits are set, `false` otherwise
///
/// # Performance
/// O(n) where n = indices.len(), with stable performance regardless of index order
#[inline]
pub fn are_bits_set(bitmap: &[u64; 4], indices: &[u8]) -> bool {
    if indices.is_empty() {
        return true; // Empty set
    }

    // Accumulate required masks for all 4 words
    let mut masks = [0u64; 4];

    for &idx in indices {
        let word = (idx / 64) as usize;
        let bit = idx % 64;
        masks[word] |= 1u64 << bit;
    }

    // Check all masks
    (bitmap[0] & masks[0]) == masks[0]
        && (bitmap[1] & masks[1]) == masks[1]
        && (bitmap[2] & masks[2]) == masks[2]
        && (bitmap[3] & masks[3]) == masks[3]
}

/// Check if bitmap is empty (no bits set).
///
/// # Arguments
/// * `bitmap` - Reference to 4-word bitmap
///
/// # Returns
/// `true` if no bits are set, `false` otherwise
///
/// # Performance
/// O(1) - checks all 4 words with bitwise OR
#[inline]
pub fn is_empty(bitmap: &[u64; 4]) -> bool {
    bitmap[0] == 0 && bitmap[1] == 0 && bitmap[2] == 0 && bitmap[3] == 0
}

/// Check if bitmap is full (all bits set).
///
/// # Arguments
/// * `bitmap` - Reference to 4-word bitmap
///
/// # Returns
/// `true` if all bits are set, `false` otherwise
///
/// # Performance
/// O(1) - checks all 4 words with bitwise AND
#[inline]
pub fn is_full(bitmap: &[u64; 4]) -> bool {
    bitmap[0] == !0u64 && bitmap[1] == !0u64 && bitmap[2] == !0u64 && bitmap[3] == !0u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitmap::{clear_bit, set_bit, set_bits, set_range};

    #[test]
    fn test_is_range_set() {
        let mut bitmap = [0u64; 4];

        // Set range [10, 20)
        set_range(&mut bitmap, 10, 20);

        // Check exact range
        assert!(is_range_set(&bitmap, 10, 20));

        // Check subrange
        assert!(is_range_set(&bitmap, 12, 18));

        // Check outside range
        assert!(!is_range_set(&bitmap, 9, 20));
        assert!(!is_range_set(&bitmap, 10, 21));

        // Check cross-word boundary
        let mut bitmap = [0u64; 4];
        set_range(&mut bitmap, 60, 70);
        assert!(is_range_set(&bitmap, 60, 70));
        assert!(is_range_set(&bitmap, 62, 68));
        assert!(!is_range_set(&bitmap, 59, 70));

        // Check full range
        let bitmap = [!0u64; 4];
        assert!(is_range_set(&bitmap, 0, 256));

        // Check empty range
        assert!(is_range_set(&bitmap, 10, 10));
    }

    #[test]
    fn test_are_bits_set() {
        let mut bitmap = [0u64; 4];

        // Set specific bits
        let indices = [5, 10, 15, 42, 100, 200, 255];
        set_bits(&mut bitmap, &indices);

        // Check all are set
        assert!(are_bits_set(&bitmap, &indices));

        // Check subset
        assert!(are_bits_set(&bitmap, &[5, 10, 15]));
        assert!(are_bits_set(&bitmap, &[100, 200]));

        // Check with missing bit
        assert!(!are_bits_set(&bitmap, &[5, 6, 10]));
        assert!(!are_bits_set(&bitmap, &[99, 100]));

        // Check empty set
        assert!(are_bits_set(&bitmap, &[]));

        // Check single bit
        assert!(are_bits_set(&bitmap, &[42]));
        assert!(!are_bits_set(&bitmap, &[43]));
    }

    #[test]
    fn test_is_empty() {
        let bitmap = [0u64; 4];
        assert!(is_empty(&bitmap));

        let mut bitmap = [0u64; 4];
        set_bit(&mut bitmap, 42);
        assert!(!is_empty(&bitmap));

        let bitmap = [!0u64; 4];
        assert!(!is_empty(&bitmap));
    }

    #[test]
    fn test_is_full() {
        let bitmap = [!0u64; 4];
        assert!(is_full(&bitmap));

        let mut bitmap = [!0u64; 4];
        clear_bit(&mut bitmap, 42);
        assert!(!is_full(&bitmap));

        let bitmap = [0u64; 4];
        assert!(!is_full(&bitmap));
    }
}
