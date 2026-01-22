//! Bulk operations for setting/clearing multiple bits efficiently.

/// Set a range of bits [from, to) in the bitmap.
///
/// # Arguments
/// * `bitmap` - Mutable reference to 4-word bitmap
/// * `from` - Start index (inclusive, 0-255)
/// * `to` - End index (exclusive, 0-256)
///
/// # Performance
/// O(1) - processes up to 4 words with bitwise operations
#[inline]
pub fn set_range(bitmap: &mut [u64; 4], from: u8, to: u16) {
    if from as u16 >= to {
        return;
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
        bitmap[from_word] |= mask;
    } else {
        // First word: from_bit to 63
        let from_bit = from % 64;
        bitmap[from_word] |= !0u64 << from_bit;

        // Middle words: all bits
        for w in (from_word + 1)..to_word {
            bitmap[w] = !0u64;
        }

        // Last word: 0 to to_bit
        let to_bit = to % 64;
        if to_bit > 0 {
            bitmap[to_word] |= (1u64 << to_bit) - 1;
        } else if to_word < 4 {
            bitmap[to_word] = !0u64;
        }
    }
}

/// Clear a range of bits [from, to) in the bitmap.
///
/// # Arguments
/// * `bitmap` - Mutable reference to 4-word bitmap
/// * `from` - Start index (inclusive, 0-255)
/// * `to` - End index (exclusive, 0-256)
///
/// # Performance
/// O(1) - processes up to 4 words with bitwise operations
#[inline]
pub fn clear_range(bitmap: &mut [u64; 4], from: u8, to: u16) {
    if from as u16 >= to {
        return;
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
        bitmap[from_word] &= !mask;
    } else {
        // First word: from_bit to 63
        let from_bit = from % 64;
        bitmap[from_word] &= !(!0u64 << from_bit);

        // Middle words: all bits
        for w in (from_word + 1)..to_word {
            bitmap[w] = 0;
        }

        // Last word: 0 to to_bit
        let to_bit = to % 64;
        if to_bit > 0 {
            bitmap[to_word] &= !((1u64 << to_bit) - 1);
        } else if to_word < 4 {
            bitmap[to_word] = 0;
        }
    }
}

/// Set multiple bits at specified indices.
///
/// Accumulates masks for all words, then applies in one pass.
/// Works efficiently with both sorted and unsorted indices.
///
/// # Arguments
/// * `bitmap` - Mutable reference to 4-word bitmap
/// * `indices` - Slice of bit indices to set (0-255)
///
/// # Performance
/// O(n) where n = indices.len(), with stable performance regardless of index order
#[inline]
pub fn set_bits(bitmap: &mut [u64; 4], indices: &[u8]) {
    if indices.is_empty() {
        return;
    }

    // Accumulate masks for all 4 words
    let mut masks = [0u64; 4];

    for &idx in indices {
        let word = (idx / 64) as usize;
        let bit = idx % 64;
        masks[word] |= 1u64 << bit;
    }

    // Apply all masks (SIMD-friendly)
    bitmap[0] |= masks[0];
    bitmap[1] |= masks[1];
    bitmap[2] |= masks[2];
    bitmap[3] |= masks[3];
}

/// Clear multiple bits at specified indices.
///
/// Accumulates masks for all words, then applies in one pass.
/// Works efficiently with both sorted and unsorted indices.
///
/// # Arguments
/// * `bitmap` - Mutable reference to 4-word bitmap
/// * `indices` - Slice of bit indices to clear (0-255)
///
/// # Performance
/// O(n) where n = indices.len(), with stable performance regardless of index order
#[inline]
pub fn clear_bits(bitmap: &mut [u64; 4], indices: &[u8]) {
    if indices.is_empty() {
        return;
    }

    // Accumulate masks for all 4 words
    let mut masks = [0u64; 4];

    for &idx in indices {
        let word = (idx / 64) as usize;
        let bit = idx % 64;
        masks[word] |= 1u64 << bit;
    }

    // Apply all masks (SIMD-friendly)
    bitmap[0] &= !masks[0];
    bitmap[1] &= !masks[1];
    bitmap[2] &= !masks[2];
    bitmap[3] &= !masks[3];
}

/// Set all 256 bits in the bitmap.
///
/// # Arguments
/// * `bitmap` - Mutable reference to 4-word bitmap
///
/// # Performance
/// O(1) - SIMD-friendly, 4 independent stores
#[inline]
pub fn set_all(bitmap: &mut [u64; 4]) {
    bitmap[0] = !0u64;
    bitmap[1] = !0u64;
    bitmap[2] = !0u64;
    bitmap[3] = !0u64;
}

/// Clear all 256 bits in the bitmap.
///
/// # Arguments
/// * `bitmap` - Mutable reference to 4-word bitmap
///
/// # Performance
/// O(1) - SIMD-friendly, 4 independent stores
#[inline]
pub fn clear_all(bitmap: &mut [u64; 4]) {
    bitmap[0] = 0;
    bitmap[1] = 0;
    bitmap[2] = 0;
    bitmap[3] = 0;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitmap::{is_set, set_bit};

    #[test]
    fn test_set_range() {
        let mut bitmap = [0u64; 4];

        // Single word range
        set_range(&mut bitmap, 10, 20);
        for i in 10..20 {
            assert!(is_set(&bitmap, i));
        }
        assert!(!is_set(&bitmap, 9));
        assert!(!is_set(&bitmap, 20));

        // Cross word boundary
        let mut bitmap = [0u64; 4];
        set_range(&mut bitmap, 60, 70);
        for i in 60..70 {
            assert!(is_set(&bitmap, i));
        }

        // Full range
        let mut bitmap = [0u64; 4];
        set_range(&mut bitmap, 0, 256);
        for i in 0..=255 {
            assert!(is_set(&bitmap, i));
        }
    }

    #[test]
    fn test_clear_range() {
        let mut bitmap = [!0u64; 4];

        // Single word range
        clear_range(&mut bitmap, 10, 20);
        for i in 10..20 {
            assert!(!is_set(&bitmap, i));
        }
        assert!(is_set(&bitmap, 9));
        assert!(is_set(&bitmap, 20));

        // Cross word boundary
        let mut bitmap = [!0u64; 4];
        clear_range(&mut bitmap, 60, 70);
        for i in 60..70 {
            assert!(!is_set(&bitmap, i));
        }
    }

    #[test]
    fn test_set_bits() {
        let mut bitmap = [0u64; 4];
        let indices = [5, 10, 15, 42, 100, 200, 255];

        set_bits(&mut bitmap, &indices);

        for &idx in &indices {
            assert!(is_set(&bitmap, idx));
        }
        assert!(!is_set(&bitmap, 6));
        assert!(!is_set(&bitmap, 99));
    }

    #[test]
    fn test_clear_bits() {
        let mut bitmap = [!0u64; 4];
        let indices = [5, 10, 15, 42, 100, 200, 255];

        clear_bits(&mut bitmap, &indices);

        for &idx in &indices {
            assert!(!is_set(&bitmap, idx));
        }
        assert!(is_set(&bitmap, 6));
        assert!(is_set(&bitmap, 99));
    }

    #[test]
    fn test_set_all() {
        let mut bitmap = [0u64; 4];
        set_all(&mut bitmap);

        for i in 0..=255 {
            assert!(is_set(&bitmap, i));
        }
    }

    #[test]
    fn test_clear_all() {
        let mut bitmap = [!0u64; 4];
        clear_all(&mut bitmap);

        for i in 0..=255 {
            assert!(!is_set(&bitmap, i));
        }
    }
}
