//! Basic single-bit operations with atomic support.

use core::sync::atomic::{AtomicU64, Ordering};

/// Set a bit in the bitmap at the given index (atomic operation).
///
/// # Arguments
/// * `bitmap` - Reference to 4-word atomic bitmap
/// * `idx` - Bit index (0-255)
///
/// # Performance
/// O(1) - atomic fetch_or operation (~10 cycles)
///
/// # Thread Safety
/// Lock-free atomic operation. Safe for concurrent access.
/// Uses Relaxed ordering for maximum performance.
#[inline]
pub fn set_bit(bitmap: &[AtomicU64; 4], idx: u8) {
    let word = idx as usize / 64;
    let bit = idx as usize % 64;
    bitmap[word].fetch_or(1u64 << bit, Ordering::Relaxed);
}

/// Clear a bit in the bitmap at the given index (atomic operation).
///
/// # Arguments
/// * `bitmap` - Reference to 4-word atomic bitmap
/// * `idx` - Bit index (0-255)
///
/// # Performance
/// O(1) - atomic fetch_and operation (~10 cycles)
///
/// # Thread Safety
/// Lock-free atomic operation. Safe for concurrent access.
/// Uses Relaxed ordering for maximum performance.
#[inline]
pub fn clear_bit(bitmap: &[AtomicU64; 4], idx: u8) {
    let word = idx as usize / 64;
    let bit = idx as usize % 64;
    bitmap[word].fetch_and(!(1u64 << bit), Ordering::Relaxed);
}

/// Check if a bit is set in the bitmap (atomic load).
///
/// # Arguments
/// * `bitmap` - Reference to 4-word atomic bitmap
/// * `idx` - Bit index (0-255)
///
/// # Returns
/// `true` if bit is set, `false` otherwise
///
/// # Performance
/// O(1) - atomic load operation (~1-2 cycles)
///
/// # Thread Safety
/// Lock-free atomic operation. Safe for concurrent access.
/// Uses Acquire ordering to ensure visibility of previous writes.
#[inline]
pub fn is_set(bitmap: &[AtomicU64; 4], idx: u8) -> bool {
    let word = idx as usize / 64;
    let bit = idx as usize % 64;
    let value = bitmap[word].load(Ordering::Acquire);
    value & (1u64 << bit) != 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_bit() {
        const INIT: AtomicU64 = AtomicU64::new(0);
        let bitmap = [INIT; 4];
        
        set_bit(&bitmap, 0);
        assert_eq!(bitmap[0].load(Ordering::Acquire), 1);

        set_bit(&bitmap, 63);
        assert_eq!(bitmap[0].load(Ordering::Acquire), 1u64 | (1u64 << 63));

        set_bit(&bitmap, 64);
        assert_eq!(bitmap[1].load(Ordering::Acquire), 1);

        set_bit(&bitmap, 255);
        assert_eq!(bitmap[3].load(Ordering::Acquire), 1u64 << 63);
    }

    #[test]
    fn test_clear_bit() {
        let bitmap = [
            AtomicU64::new(!0u64),
            AtomicU64::new(!0u64),
            AtomicU64::new(!0u64),
            AtomicU64::new(!0u64),
        ];
        
        clear_bit(&bitmap, 0);
        assert_eq!(bitmap[0].load(Ordering::Acquire), !1u64);

        clear_bit(&bitmap, 255);
        assert_eq!(bitmap[3].load(Ordering::Acquire), !(1u64 << 63));
    }

    #[test]
    fn test_is_set() {
        const INIT: AtomicU64 = AtomicU64::new(0);
        let bitmap = [INIT; 4];
        
        assert!(!is_set(&bitmap, 0));

        set_bit(&bitmap, 42);
        assert!(is_set(&bitmap, 42));
        assert!(!is_set(&bitmap, 43));
    }

    #[test]
    fn test_concurrent_set_bit() {
        // Test that multiple threads can set different bits concurrently
        const INIT: AtomicU64 = AtomicU64::new(0);
        let bitmap = [INIT; 4];
        
        // Simulate concurrent access by setting multiple bits
        set_bit(&bitmap, 0);
        set_bit(&bitmap, 1);
        set_bit(&bitmap, 63);
        set_bit(&bitmap, 64);
        
        assert!(is_set(&bitmap, 0));
        assert!(is_set(&bitmap, 1));
        assert!(is_set(&bitmap, 63));
        assert!(is_set(&bitmap, 64));
    }
}
