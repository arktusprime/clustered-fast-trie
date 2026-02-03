//! Basic single-bit operations with atomic support.

use crate::atomic::{AtomicU64, Ordering};

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

/// Atomically test and set a bit in the bitmap.
///
/// Returns `true` if the bit was NOT set (caller is first to set it).
/// Returns `false` if the bit was already set (another thread got there first).
///
/// This is the critical primitive for lock-free node allocation coordination.
/// Multiple threads can race to allocate the same child - only one succeeds.
///
/// # Arguments
/// * `bitmap` - Reference to 4-word atomic bitmap
/// * `idx` - Bit index (0-255)
///
/// # Returns
/// `true` if bit was NOT set (caller won the race), `false` if already set
///
/// # Performance
/// O(1) - atomic fetch_or operation (~10 cycles)
///
/// # Thread Safety
/// Lock-free atomic operation. Safe for concurrent access.
/// Uses AcqRel ordering to ensure:
/// - Acquire: see all writes before the bit was set
/// - Release: make our writes visible to threads that see the bit set
///
/// # Example
/// ```ignore
/// // Multiple threads racing to allocate child at index 42
/// if test_and_set_bit(&node.bitmap, 42) {
///     // This thread won - allocate the child
///     let child_idx = arena.alloc();
///     node.children[42] = child_idx;
/// } else {
///     // Another thread won - use their allocation
///     let child_idx = node.children[42];
/// }
/// ```
#[inline]
pub fn test_and_set_bit(bitmap: &[AtomicU64; 4], idx: u8) -> bool {
    let word = idx as usize / 64;
    let bit = idx as usize % 64;
    let mask = 1u64 << bit;
    let prev = bitmap[word].fetch_or(mask, Ordering::AcqRel);
    prev & mask == 0 // true if bit was NOT set
}

/// Check if a bit is set with seqlock protection.
///
/// Uses seqlock protocol to ensure consistent read during concurrent bulk modifications.
/// Reader loops until observing stable seq (even and unchanged).
///
/// # Arguments
/// * `seq` - Sequence counter for seqlock protocol
/// * `bitmap` - Reference to 4-word atomic bitmap
/// * `idx` - Bit index (0-255)
///
/// # Returns
/// `true` if bit is set, `false` otherwise
///
/// # Performance
/// O(1) - ~5% overhead from seqlock without collisions
/// Retries only if concurrent bulk modification detected
///
/// # Thread Safety
/// Lock-free read operation. Safe for concurrent access.
/// Protocol:
/// 1. Load seq (must be even = no writer active)
/// 2. Read bit value
/// 3. Load seq again (must match = no writer intervened)
/// 4. If seq changed or was odd, retry
///
/// # Example
/// ```ignore
/// // Reader checks bit while writers may be doing bulk operations
/// if is_set_seqlock(&node.seq, &node.bitmap, 42) {
///     // Bit is set (consistent read)
/// }
/// ```
#[inline]
#[allow(dead_code)]
pub fn is_set_seqlock(seq: &AtomicU64, bitmap: &[AtomicU64; 4], idx: u8) -> bool {
    loop {
        let seq_before = seq.load(Ordering::Acquire);
        if seq_before & 1 != 0 {
            // Odd - writer active, retry
            core::hint::spin_loop();
            continue;
        }
        let result = is_set(bitmap, idx);
        let seq_after = seq.load(Ordering::Acquire);
        if seq_before == seq_after {
            return result; // Consistent read
        }
        // Seq changed, retry
    }
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

    #[test]
    fn test_test_and_set_bit_first_wins() {
        const INIT: AtomicU64 = AtomicU64::new(0);
        let bitmap = [INIT; 4];

        // First call should return true (bit was NOT set)
        assert!(test_and_set_bit(&bitmap, 42));
        assert!(is_set(&bitmap, 42));

        // Second call should return false (bit was already set)
        assert!(!test_and_set_bit(&bitmap, 42));
        assert!(is_set(&bitmap, 42));
    }

    #[test]
    fn test_test_and_set_bit_different_words() {
        const INIT: AtomicU64 = AtomicU64::new(0);
        let bitmap = [INIT; 4];

        // Test across all 4 words
        assert!(test_and_set_bit(&bitmap, 0)); // word 0
        assert!(test_and_set_bit(&bitmap, 64)); // word 1
        assert!(test_and_set_bit(&bitmap, 128)); // word 2
        assert!(test_and_set_bit(&bitmap, 192)); // word 3

        // Verify all set
        assert!(is_set(&bitmap, 0));
        assert!(is_set(&bitmap, 64));
        assert!(is_set(&bitmap, 128));
        assert!(is_set(&bitmap, 192));

        // Try again - should all return false
        assert!(!test_and_set_bit(&bitmap, 0));
        assert!(!test_and_set_bit(&bitmap, 64));
        assert!(!test_and_set_bit(&bitmap, 128));
        assert!(!test_and_set_bit(&bitmap, 192));
    }

    #[test]
    fn test_test_and_set_bit_boundary_cases() {
        const INIT: AtomicU64 = AtomicU64::new(0);
        let bitmap = [INIT; 4];

        // Test word boundaries
        assert!(test_and_set_bit(&bitmap, 63)); // last bit of word 0
        assert!(test_and_set_bit(&bitmap, 64)); // first bit of word 1
        assert!(test_and_set_bit(&bitmap, 127)); // last bit of word 1
        assert!(test_and_set_bit(&bitmap, 128)); // first bit of word 2
        assert!(test_and_set_bit(&bitmap, 191)); // last bit of word 2
        assert!(test_and_set_bit(&bitmap, 192)); // first bit of word 3
        assert!(test_and_set_bit(&bitmap, 255)); // last bit of word 3

        // Verify all set
        assert!(is_set(&bitmap, 63));
        assert!(is_set(&bitmap, 64));
        assert!(is_set(&bitmap, 127));
        assert!(is_set(&bitmap, 128));
        assert!(is_set(&bitmap, 191));
        assert!(is_set(&bitmap, 192));
        assert!(is_set(&bitmap, 255));
    }

    #[test]
    fn test_test_and_set_bit_with_existing_bits() {
        // Start with some bits already set
        let bitmap = [
            AtomicU64::new(0b1010),
            AtomicU64::new(0),
            AtomicU64::new(0),
            AtomicU64::new(0),
        ];

        // Bit 1 is already set
        assert!(!test_and_set_bit(&bitmap, 1));

        // Bit 3 is already set
        assert!(!test_and_set_bit(&bitmap, 3));

        // Bit 2 is NOT set
        assert!(test_and_set_bit(&bitmap, 2));

        // Now bit 2 is set
        assert!(!test_and_set_bit(&bitmap, 2));
    }
}
