//! Internal node structure for 256-way branching trie.

use crate::atomic::{AtomicU64, Ordering};
use crate::constants::EMPTY;

/// Internal node with 256-way branching.
///
/// Uses direct indexing for O(1) child access and bitmap for O(1) existence checks.
///
/// # Memory Layout
/// - `seq`: 8 bytes - sequence counter for seqlock
/// - `bitmap`: 32 bytes (256 bits = 4 × u64) - child existence bitmap
/// - `parent_idx`: 4 bytes - index of parent node (for cleanup)
/// - `child_arena_idx`: 4 bytes - physical index in Trie.arenas Vec
/// - `children`: 1024 bytes (256 × u32) - child indices
/// - Total: 1072 bytes per node
///
/// # Cache Optimization
/// - Aligned to 64-byte cache line boundary
/// - Seq and bitmap in dedicated cache line for fast access
/// - Children array uses sequential access patterns
///
/// # Arena Architecture
/// - `child_arena_idx` stores physical index into `Trie.arenas: Vec<ChildArenas>`
/// - O(1) arena access via direct array indexing
/// - Set only at split levels (u64 level 4, u128 levels 4 and 12)
/// - Zero for nodes that don't have child arenas
///
/// # Performance
/// - Child access: O(1) via direct indexing
/// - Existence check: O(1) via bitmap
/// - Min/max child: O(1) via bitmap intrinsics (TZCNT/LZCNT)
/// - Parent access: O(1) via parent_idx (for cleanup)
/// - Arena access: O(1) via Trie.arenas[child_arena_idx]
///
/// # Concurrency
/// - Seq counter for seqlock protocol (bulk operations)
/// - Even = stable, odd = writer active
#[derive(Debug)]
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

    /// Index of parent node in arena.
    ///
    /// Used for O(1) traversal up the tree during cleanup operations.
    /// Root node (index 0) has parent_idx = 0 (points to itself).
    pub parent_idx: u32,

    /// Physical arena index for children of this node (u32 index into Trie.arenas).
    ///
    /// - 0: children in same arena as parent (no split)
    /// - N: children in arena at Trie.arenas[N] (split level)
    ///
    /// Set only at split levels:
    /// - u32: always 0 (no splits)
    /// - u64: set at level 4
    /// - u128: set at levels 4 and 12
    pub child_arena_idx: u32,

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
    /// Parent index is initialized to 0 (will be set during insertion).
    /// Child arena index is initialized to 0 (same arena as parent).
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
            parent_idx: 0,
            child_arena_idx: 0,
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
            parent_idx: self.parent_idx,
            child_arena_idx: self.child_arena_idx,
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
        // 8 (seq) + 32 (bitmap) + 4 (parent_idx) + 4 (child_arena_idx) + 1024 (children) = 1072 bytes
        assert_eq!(size_of::<Node>(), 1088); // Rounded to next cache line multiple (17 × 64)
        assert_eq!(align_of::<Node>(), 64); // Aligned to cache line
        assert_eq!(size_of::<AtomicU64>(), 8);
        assert_eq!(size_of::<[AtomicU64; 4]>(), 32);
        assert_eq!(size_of::<u32>(), 4);
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
