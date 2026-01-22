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
}
