//! CPU intrinsic operations for fast bit manipulation.
#![allow(dead_code)]

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
