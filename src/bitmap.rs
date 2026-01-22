//! Low-level bitmap operations using CPU intrinsics.
//!
//! These functions provide efficient bit manipulation for 256-bit bitmaps
//! represented as arrays of 4 u64 words.

/// Set a bit in the bitmap at the given index.
///
/// # Arguments
/// * `bitmap` - Mutable reference to 4-word bitmap
/// * `idx` - Bit index (0-255)
///
/// # Performance
/// O(1) - direct array access and bitwise OR
#[inline]
pub fn set_bit(bitmap: &mut [u64; 4], idx: u8) {
    let word = idx as usize / 64;
    let bit = idx as usize % 64;
    bitmap[word] |= 1u64 << bit;
}

/// Clear a bit in the bitmap at the given index.
///
/// # Arguments
/// * `bitmap` - Mutable reference to 4-word bitmap
/// * `idx` - Bit index (0-255)
///
/// # Performance
/// O(1) - direct array access and bitwise AND
#[inline]
pub fn clear_bit(bitmap: &mut [u64; 4], idx: u8) {
    let word = idx as usize / 64;
    let bit = idx as usize % 64;
    bitmap[word] &= !(1u64 << bit);
}

/// Check if a bit is set in the bitmap.
///
/// # Arguments
/// * `bitmap` - Reference to 4-word bitmap
/// * `idx` - Bit index (0-255)
///
/// # Returns
/// `true` if bit is set, `false` otherwise
///
/// # Performance
/// O(1) - direct array access and bitwise AND
#[inline]
pub fn is_set(bitmap: &[u64; 4], idx: u8) -> bool {
    let word = idx as usize / 64;
    let bit = idx as usize % 64;
    bitmap[word] & (1u64 << bit) != 0
}

/// Count trailing zeros (find first set bit from right).
///
/// Uses CPU TZCNT instruction for O(1) performance.
///
/// # Arguments
/// * `word` - 64-bit word
///
/// # Returns
/// Number of trailing zeros (0-64)
///
/// # Performance
/// O(1) - single CPU instruction (TZCNT)
#[inline]
pub fn trailing_zeros(word: u64) -> u32 {
    word.trailing_zeros()
}

/// Count leading zeros (find first set bit from left).
///
/// Uses CPU LZCNT instruction for O(1) performance.
///
/// # Arguments
/// * `word` - 64-bit word
///
/// # Returns
/// Number of leading zeros (0-64)
///
/// # Performance
/// O(1) - single CPU instruction (LZCNT)
#[inline]
pub fn leading_zeros(word: u64) -> u32 {
    word.leading_zeros()
}

/// Count set bits in a word.
///
/// Uses CPU POPCNT instruction for O(1) performance.
///
/// # Arguments
/// * `word` - 64-bit word
///
/// # Returns
/// Number of set bits (0-64)
///
/// # Performance
/// O(1) - single CPU instruction (POPCNT)
#[inline]
pub fn popcount(word: u64) -> u32 {
    word.count_ones()
}

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

    #[test]
    fn test_set_bit() {
        let mut bitmap = [0u64; 4];
        set_bit(&mut bitmap, 0);
        assert_eq!(bitmap[0], 1);

        set_bit(&mut bitmap, 63);
        assert_eq!(bitmap[0], 1u64 | (1u64 << 63));

        set_bit(&mut bitmap, 64);
        assert_eq!(bitmap[1], 1);

        set_bit(&mut bitmap, 255);
        assert_eq!(bitmap[3], 1u64 << 63);
    }

    #[test]
    fn test_clear_bit() {
        let mut bitmap = [!0u64; 4];
        clear_bit(&mut bitmap, 0);
        assert_eq!(bitmap[0], !1u64);

        clear_bit(&mut bitmap, 255);
        assert_eq!(bitmap[3], !(1u64 << 63));
    }

    #[test]
    fn test_is_set() {
        let mut bitmap = [0u64; 4];
        assert!(!is_set(&bitmap, 0));

        set_bit(&mut bitmap, 42);
        assert!(is_set(&bitmap, 42));
        assert!(!is_set(&bitmap, 43));
    }

    #[test]
    fn test_trailing_zeros() {
        assert_eq!(trailing_zeros(0), 64);
        assert_eq!(trailing_zeros(1), 0);
        assert_eq!(trailing_zeros(2), 1);
        assert_eq!(trailing_zeros(4), 2);
        assert_eq!(trailing_zeros(1u64 << 63), 63);
    }

    #[test]
    fn test_leading_zeros() {
        assert_eq!(leading_zeros(0), 64);
        assert_eq!(leading_zeros(1), 63);
        assert_eq!(leading_zeros(1u64 << 63), 0);
    }

    #[test]
    fn test_popcount() {
        assert_eq!(popcount(0), 0);
        assert_eq!(popcount(1), 1);
        assert_eq!(popcount(3), 2);
        assert_eq!(popcount(!0u64), 64);
    }

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
