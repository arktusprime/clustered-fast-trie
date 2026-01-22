//! Basic single-bit operations.

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
}
