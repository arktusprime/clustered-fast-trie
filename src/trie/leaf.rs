//! Leaf node structure for storing actual keys.

use crate::atomic::{AtomicU64, Ordering};
use crate::constants::EMPTY;

/// Leaf node storing 256 keys via bitmap.
///
/// Each leaf represents a 256-key range identified by a prefix.
/// Keys within the range are stored as bits in the bitmap.
///
/// # Memory Layout
/// - `bitmap`: 32 bytes (256 bits = 4 × u64)
/// - `prefix`: 8 bytes (key prefix, last byte = 0)
/// - `next`: 4 bytes (arena index of next leaf)
/// - `prev`: 4 bytes (arena index of previous leaf)
/// - Total: 48 bytes per leaf
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

    /// Arena index of next leaf in sorted order.
    ///
    /// Points to leaf with next higher prefix, or `EMPTY` if this is last leaf.
    pub next: u32,

    /// Arena index of previous leaf in sorted order.
    ///
    /// Points to leaf with next lower prefix, or `EMPTY` if this is first leaf.
    pub prev: u32,
}

impl Leaf {
    /// Create a new empty leaf with given prefix.
    ///
    /// # Arguments
    /// * `prefix` - Key prefix (last byte should be 0)
    ///
    /// # Returns
    /// New leaf with empty bitmap and unlinked (next/prev = EMPTY)
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
            next: EMPTY,
            prev: EMPTY,
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
        assert_eq!(leaf.next, EMPTY);
        assert_eq!(leaf.prev, EMPTY);
    }

    #[test]
    fn test_leaf_size() {
        use core::mem::size_of;

        // Verify expected memory layout
        assert_eq!(size_of::<Leaf>(), 48); // 32 + 8 + 4 + 4 = 48 bytes
        assert_eq!(size_of::<[AtomicU64; 4]>(), 32);
        assert_eq!(size_of::<u64>(), 8);
        assert_eq!(size_of::<u32>(), 4);
    }

    #[test]
    fn test_clone() {
        let mut leaf = Leaf::new(0x12345600);
        leaf.bitmap[0].store(0xFF, Ordering::Relaxed);
        leaf.next = 42;
        leaf.prev = 10;

        let cloned = leaf.clone();

        assert_eq!(cloned.bitmap[0].load(Ordering::Relaxed), 0xFF);
        assert_eq!(cloned.prefix, 0x12345600);
        assert_eq!(cloned.next, 42);
        assert_eq!(cloned.prev, 10);
    }
}
