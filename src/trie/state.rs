//! Node state checking operations.

use crate::bitmap;
use crate::trie::Node;

impl Node {
    /// Check if node has no children.
    ///
    /// # Returns
    /// `true` if node is empty (no children), `false` otherwise
    ///
    /// # Performance
    /// O(1) - uses OR reduction across bitmap words (SIMD-friendly)
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        bitmap::is_empty(&self.bitmap)
    }

    /// Check if node has all 256 children.
    ///
    /// # Returns
    /// `true` if node is full (all children exist), `false` otherwise
    ///
    /// # Performance
    /// O(1) - uses AND reduction across bitmap words (SIMD-friendly)
    #[inline(always)]
    pub fn is_full(&self) -> bool {
        bitmap::is_full(&self.bitmap)
    }

    /// Get number of children in this node.
    ///
    /// # Returns
    /// Count of existing children (0-256)
    ///
    /// # Performance
    /// O(1) - uses POPCNT instruction
    #[inline(always)]
    pub fn child_count(&self) -> u32 {
        bitmap::count_bits(&self.bitmap)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_empty_new_node() {
        let node = Node::new();
        assert!(node.is_empty());
    }

    #[test]
    fn test_is_empty_with_children() {
        let mut node = Node::new();

        // Add a child
        node.set_child(42, 100);
        assert!(!node.is_empty());

        // Add more children
        node.set_child(0, 10);
        node.set_child(255, 999);
        assert!(!node.is_empty());
    }

    #[test]
    fn test_is_empty_after_clear() {
        let mut node = Node::new();

        // Add and remove single child
        node.set_child(42, 100);
        assert!(!node.is_empty());

        node.clear_child(42);
        assert!(node.is_empty());
    }

    #[test]
    fn test_is_empty_after_clear_all() {
        let mut node = Node::new();

        // Add multiple children
        for i in 0..10 {
            node.set_child(i * 25, i as u32 * 100);
        }
        assert!(!node.is_empty());

        // Clear all
        for i in 0..10 {
            node.clear_child(i * 25);
        }
        assert!(node.is_empty());
    }

    #[test]
    fn test_is_full_new_node() {
        let node = Node::new();
        assert!(!node.is_full());
    }

    #[test]
    fn test_is_full_partial() {
        let mut node = Node::new();

        // Add some children
        for i in 0..10 {
            node.set_child(i * 25, i as u32 * 100);
        }
        assert!(!node.is_full());

        // Add more but not all
        for i in 10..100 {
            node.set_child(i, i as u32);
        }
        assert!(!node.is_full());
    }

    #[test]
    fn test_is_full_complete() {
        let mut node = Node::new();

        // Add all 256 children
        for i in 0..256 {
            node.set_child(i as u8, i as u32);
        }
        assert!(node.is_full());
        assert!(!node.is_empty());
    }

    #[test]
    fn test_is_full_after_clear() {
        let mut node = Node::new();

        // Fill completely
        for i in 0..256 {
            node.set_child(i as u8, i as u32);
        }
        assert!(node.is_full());

        // Clear one child
        node.clear_child(42);
        assert!(!node.is_full());
        assert!(!node.is_empty());
    }

    #[test]
    fn test_child_count_new_node() {
        let node = Node::new();
        assert_eq!(node.child_count(), 0);
    }

    #[test]
    fn test_child_count_partial() {
        let mut node = Node::new();

        // Add some children (non-overlapping indices)
        for i in 0..10 {
            node.set_child(i * 25, i as u32 * 100);
        }
        assert_eq!(node.child_count(), 10);

        // Add more (different indices)
        for i in 1..11 {
            // 1, 2, 3, ..., 10 (avoid 0 which might overlap)
            node.set_child(i, i as u32);
        }
        assert_eq!(node.child_count(), 20); // 10 + 10 = 20
    }

    #[test]
    fn test_child_count_full() {
        let mut node = Node::new();

        // Add all 256 children
        for i in 0..256 {
            node.set_child(i as u8, i as u32);
        }
        assert_eq!(node.child_count(), 256);
    }

    #[test]
    fn test_child_count_after_clear() {
        let mut node = Node::new();

        // Add children
        for i in 0..20 {
            node.set_child(i * 10, i as u32);
        }
        assert_eq!(node.child_count(), 20);

        // Clear some
        for i in 0..5 {
            node.clear_child(i * 10);
        }
        assert_eq!(node.child_count(), 15);

        // Clear all remaining
        for i in 5..20 {
            node.clear_child(i * 10);
        }
        assert_eq!(node.child_count(), 0);
    }
}
