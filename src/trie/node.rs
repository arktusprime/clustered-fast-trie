//! Internal node structure for 256-way branching trie.

use crate::atomic::{AtomicU64, Ordering};
use crate::constants::EMPTY;

/// Internal node with 256-way branching.
///
/// Uses direct indexing for O(1) child access and bitmap for O(1) existence checks.
///
/// # Memory Layout
/// - `bitmap`: 32 bytes (256 bits = 4 × u64) - cache line 0
/// - `_pad`: 32 bytes padding - completes cache line 0
/// - `children`: 1024 bytes (256 × u32) - cache lines 1-16
/// - Total: 1088 bytes per node (17 cache lines)
///
/// # Cache Optimization
/// - Aligned to 64-byte cache line boundary
/// - Seq and bitmap in dedicated cache line for fast access
/// - Children array starts at cache line boundary
///
/// # Performance
/// - Child access: O(1) via direct indexing
/// - Existence check: O(1) via bitmap
/// - Min/max child: O(1) via bitmap intrinsics (TZCNT/LZCNT)
///
/// # Concurrency
/// - Seq counter for seqlock protocol (bulk operations)
/// - Even = stable, odd = writer active
#[repr(C, align(64))]
pub struct Node {
    /// Sequence counter for seqlock protocol.
    ///
    /// Used for bulk operations to ensure readers see consistent state.
    /// Even value = stable, odd value = writer active.
    pub seq: AtomicU64,

    /// Bitmap indicating which children exist (256 bits).
    ///
    /// Each bit corresponds to a child index (0-255).
    /// Bit set = child exists, bit clear = no child.
    ///
    /// Uses AtomicU64 for lock-free multi-threading.
    pub bitmap: [AtomicU64; 4],

    /// Padding to align children to cache line boundary.
    ///
    /// Ensures seq + bitmap occupy exactly one cache line (64 bytes).
    pub(crate) _pad: [u64; 3],

    /// Direct-indexed children array.
    ///
    /// `children[i]` contains arena index of child at byte value `i`,
    /// or `EMPTY` (u32::MAX) if no child exists.
    pub children: [u32; 256],
}

impl Node {
    /// Create a new empty node.
    ///
    /// All children are initialized to `EMPTY` (u32::MAX).
    ///
    /// # Performance
    /// O(1) - uses array initialization
    ///
    /// Hot path for arena allocation - always inlined.
    #[inline(always)]
    pub fn new() -> Self {
        Node {
            seq: AtomicU64::new(0),
            bitmap: [
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
            ],
            _pad: [0; 3],
            children: [EMPTY; 256],
        }
    }
}

impl Default for Node {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for Node {
    fn clone(&self) -> Self {
        Node {
            seq: AtomicU64::new(self.seq.load(Ordering::Relaxed)),
            bitmap: [
                AtomicU64::new(self.bitmap[0].load(Ordering::Relaxed)),
                AtomicU64::new(self.bitmap[1].load(Ordering::Relaxed)),
                AtomicU64::new(self.bitmap[2].load(Ordering::Relaxed)),
                AtomicU64::new(self.bitmap[3].load(Ordering::Relaxed)),
            ],
            _pad: self._pad,
            children: self.children,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_node() {
        let node = Node::new();

        // Seq should be 0 (even = stable)
        assert_eq!(node.seq.load(Ordering::Relaxed), 0);

        // Bitmap should be empty
        assert_eq!(node.bitmap[0].load(Ordering::Relaxed), 0);
        assert_eq!(node.bitmap[1].load(Ordering::Relaxed), 0);
        assert_eq!(node.bitmap[2].load(Ordering::Relaxed), 0);
        assert_eq!(node.bitmap[3].load(Ordering::Relaxed), 0);

        // All children should be EMPTY
        for i in 0..256 {
            assert_eq!(node.children[i], EMPTY);
        }
    }

    #[test]
    fn test_node_size() {
        use core::mem::{align_of, size_of};

        // Verify expected memory layout with cache line alignment
        assert_eq!(size_of::<Node>(), 1088); // 8 + 32 + 24 + 1024 = 1088 bytes
        assert_eq!(align_of::<Node>(), 64); // Aligned to cache line
        assert_eq!(size_of::<AtomicU64>(), 8);
        assert_eq!(size_of::<[AtomicU64; 4]>(), 32);
        assert_eq!(size_of::<[u64; 3]>(), 24);
        assert_eq!(size_of::<[u32; 256]>(), 1024);
    }

    #[test]
    fn test_default() {
        let node = Node::default();
        assert_eq!(node.seq.load(Ordering::Relaxed), 0);
        assert_eq!(node.bitmap[0].load(Ordering::Relaxed), 0);
        assert_eq!(node.children[0], EMPTY);
    }
}
