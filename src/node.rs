//! Internal node structure for 256-way branching trie.

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
/// - Bitmap in dedicated cache line for fast access
/// - Children array starts at cache line boundary
///
/// # Performance
/// - Child access: O(1) via direct indexing
/// - Existence check: O(1) via bitmap
/// - Min/max child: O(1) via bitmap intrinsics (TZCNT/LZCNT)
#[repr(C, align(64))]
#[derive(Clone)]
pub struct Node {
    /// Bitmap indicating which children exist (256 bits).
    ///
    /// Each bit corresponds to a child index (0-255).
    /// Bit set = child exists, bit clear = no child.
    ///
    /// Placed first for cache locality (checked before children access).
    pub bitmap: [u64; 4],

    /// Padding to align children to cache line boundary.
    ///
    /// Ensures bitmap occupies exactly one cache line (64 bytes).
    _pad: [u64; 4],

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
    #[inline]
    pub fn new() -> Self {
        Node {
            bitmap: [0; 4],
            _pad: [0; 4],
            children: [EMPTY; 256],
        }
    }
}

impl Default for Node {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_node() {
        let node = Node::new();

        // Bitmap should be empty
        assert_eq!(node.bitmap, [0; 4]);

        // All children should be EMPTY
        for i in 0..256 {
            assert_eq!(node.children[i], EMPTY);
        }
    }

    #[test]
    fn test_node_size() {
        use core::mem::{align_of, size_of};

        // Verify expected memory layout with cache line alignment
        assert_eq!(size_of::<Node>(), 1088); // 32 + 32 + 1024 = 1088 bytes
        assert_eq!(align_of::<Node>(), 64); // Aligned to cache line
        assert_eq!(size_of::<[u64; 4]>(), 32);
        assert_eq!(size_of::<[u32; 256]>(), 1024);
    }

    #[test]
    fn test_default() {
        let node = Node::default();
        assert_eq!(node.bitmap, [0; 4]);
        assert_eq!(node.children[0], EMPTY);
    }
}
