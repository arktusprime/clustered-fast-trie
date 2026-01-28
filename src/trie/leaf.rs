//! Leaf node structure for storing actual keys.

use crate::atomic::{AtomicU64, Ordering};
use crate::constants::EMPTY;

/// Empty link sentinel for prev/next pointers (all bits set).
pub const EMPTY_LINK: u64 = u64::MAX;

/// Pack arena_idx and leaf_idx into a single u64.
///
/// # Arguments
/// * `arena_idx` - Arena index (u64, truncated to u32)
/// * `leaf_idx` - Leaf index within arena (u32)
///
/// # Returns
/// Packed u64: (arena_idx << 32) | leaf_idx
#[inline(always)]
pub fn pack_link(arena_idx: u64, leaf_idx: u32) -> u64 {
    ((arena_idx as u32 as u64) << 32) | (leaf_idx as u64)
}

/// Unpack arena_idx and leaf_idx from a packed u64.
///
/// # Arguments
/// * `packed` - Packed link value
///
/// # Returns
/// (arena_idx, leaf_idx) tuple
#[inline(always)]
pub fn unpack_link(packed: u64) -> (u64, u32) {
    let arena_idx = (packed >> 32) as u64;
    let leaf_idx = (packed & 0xFFFFFFFF) as u32;
    (arena_idx, leaf_idx)
}

/// Leaf node storing 256 keys via bitmap.
///
/// Each leaf represents a 256-key range identified by a prefix.
/// Keys within the range are stored as bits in the bitmap.
///
/// # Memory Layout
/// - `bitmap`: 32 bytes (256 bits = 4 × u64)
/// - `prefix`: 8 bytes (key prefix, last byte = 0)
/// - `next`: 8 bytes (packed arena_idx and leaf_idx)
/// - `prev`: 8 bytes (packed arena_idx and leaf_idx)
/// - Total: 56 bytes per leaf
///
/// # Linked List
/// Leaves are linked in sorted order by prefix for O(1) per-element iteration.
/// - `next`: points to leaf with next higher prefix
/// - `prev`: points to leaf with next lower prefix
/// - `EMPTY` (u32::MAX) indicates end of list
///
/// # Performance
/// - Key insert: O(1) via bitmap set
/// - Key check: O(1) via bitmap test
/// - Key remove: O(1) via bitmap clear
/// - Iteration: O(1) per key via linked list traversal
///
/// # Concurrency
/// - Bitmap uses AtomicU64 for lock-free operations
/// - Prefix is immutable after creation
/// - Next/prev updated atomically during list operations
#[derive(Debug)]
#[repr(C)]
pub struct Leaf {
    /// Bitmap indicating which keys exist (256 bits).
    ///
    /// Bit index corresponds to last byte of key.
    /// Bit set = key exists, bit clear = key absent.
    ///
    /// Uses AtomicU64 for lock-free multi-threading.
    pub bitmap: [AtomicU64; 4],

    /// Key prefix (all bytes except last).
    ///
    /// Identifies which 256-key range this leaf represents.
    /// Last byte is always 0 (masked out).
    ///
    /// Immutable after leaf creation.
    pub prefix: u64,

    /// Next leaf in sorted order (packed: arena_idx << 32 | leaf_idx).
    ///
    /// Points to leaf with next higher prefix, or `EMPTY_LINK` if this is last leaf.
    /// Upper 32 bits: arena_idx (u64 truncated to u32 for u128 keys)
    /// Lower 32 bits: leaf_idx within arena
    pub next: u64,

    /// Previous leaf in sorted order (packed: arena_idx << 32 | leaf_idx).
    ///
    /// Points to leaf with next lower prefix, or `EMPTY_LINK` if this is first leaf.
    /// Upper 32 bits: arena_idx (u64 truncated to u32 for u128 keys)
    /// Lower 32 bits: leaf_idx within arena
    pub prev: u64,
}

impl Leaf {
    /// Create a new empty leaf with given prefix.
    ///
    /// # Arguments
    /// * `prefix` - Key prefix (last byte should be 0)
    ///
    /// # Returns
    /// New leaf with empty bitmap and unlinked (next/prev = EMPTY_LINK)
    ///
    /// # Performance
    /// O(1) - simple initialization
    ///
    /// Hot path for arena allocation - always inlined.
    #[inline(always)]
    pub fn new(prefix: u64) -> Self {
        Leaf {
            bitmap: [
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
            ],
            prefix,
            next: EMPTY_LINK,
            prev: EMPTY_LINK,
        }
    }
}

impl Clone for Leaf {
    fn clone(&self) -> Self {
        Leaf {
            bitmap: [
                AtomicU64::new(self.bitmap[0].load(Ordering::Relaxed)),
                AtomicU64::new(self.bitmap[1].load(Ordering::Relaxed)),
                AtomicU64::new(self.bitmap[2].load(Ordering::Relaxed)),
                AtomicU64::new(self.bitmap[3].load(Ordering::Relaxed)),
            ],
            prefix: self.prefix,
            next: self.next,
            prev: self.prev,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_leaf() {
        let prefix = 0x12345600u64;
        let leaf = Leaf::new(prefix);

        // Bitmap should be empty
        assert_eq!(leaf.bitmap[0].load(Ordering::Relaxed), 0);
        assert_eq!(leaf.bitmap[1].load(Ordering::Relaxed), 0);
        assert_eq!(leaf.bitmap[2].load(Ordering::Relaxed), 0);
        assert_eq!(leaf.bitmap[3].load(Ordering::Relaxed), 0);

        // Prefix should match
        assert_eq!(leaf.prefix, prefix);

        // Should be unlinked
        assert_eq!(leaf.next, EMPTY_LINK);
        assert_eq!(leaf.prev, EMPTY_LINK);
    }

    #[test]
    fn test_leaf_size() {
        use core::mem::size_of;

        // Verify expected memory layout
        assert_eq!(size_of::<Leaf>(), 56); // 32 + 8 + 8 + 8 = 56 bytes
        assert_eq!(size_of::<[AtomicU64; 4]>(), 32);
        assert_eq!(size_of::<u64>(), 8);
    }

    #[test]
    fn test_clone() {
        let mut leaf = Leaf::new(0x12345600);
        leaf.bitmap[0].store(0xFF, Ordering::Relaxed);
        leaf.next = pack_link(1, 42);
        leaf.prev = pack_link(0, 10);

        let cloned = leaf.clone();

        assert_eq!(cloned.bitmap[0].load(Ordering::Relaxed), 0xFF);
        assert_eq!(cloned.prefix, 0x12345600);
        assert_eq!(cloned.next, pack_link(1, 42));
        assert_eq!(cloned.prev, pack_link(0, 10));
    }

    #[test]
    fn test_pack_unpack_link() {
        // Test with various arena_idx and leaf_idx values
        let test_cases = [
            (0u64, 0u32),
            (1, 42),
            (0xFFFFFFFF, 0xFFFFFFFF),
            (0x12345678, 0xABCDEF01),
        ];

        for (arena_idx, leaf_idx) in test_cases {
            let packed = pack_link(arena_idx, leaf_idx);
            let (unpacked_arena, unpacked_leaf) = unpack_link(packed);

            // Note: arena_idx is truncated to u32
            assert_eq!(unpacked_arena, arena_idx & 0xFFFFFFFF);
            assert_eq!(unpacked_leaf, leaf_idx);
        }
    }

    #[test]
    fn test_empty_link() {
        assert_eq!(EMPTY_LINK, u64::MAX);

        // Verify EMPTY_LINK unpacks to max values
        let (arena_idx, leaf_idx) = unpack_link(EMPTY_LINK);
        assert_eq!(arena_idx, 0xFFFFFFFF);
        assert_eq!(leaf_idx, 0xFFFFFFFF);
    }
}
