//! Basic child operations for Node.

use crate::bitmap;
use crate::constants::EMPTY;
use crate::trie::Node;

impl Node {
    /// Check if child exists at given byte index.
    ///
    /// # Arguments
    /// * `byte` - Byte value (0-255) to check
    ///
    /// # Returns
    /// `true` if child exists, `false` otherwise
    ///
    /// # Performance
    /// O(1) - single bitmap check via bitwise AND
    #[inline(always)]
    pub fn has_child(&self, byte: u8) -> bool {
        bitmap::is_set(&self.bitmap, byte)
    }

    /// Get child index at given byte.
    ///
    /// # Arguments
    /// * `byte` - Byte value (0-255)
    ///
    /// # Returns
    /// Arena index of child, or `EMPTY` (u32::MAX) if no child exists
    ///
    /// # Performance
    /// O(1) - direct array indexing
    ///
    /// # Note
    /// Does not check if child exists. Use `has_child()` first if needed.
    #[inline(always)]
    pub fn get_child(&self, byte: u8) -> u32 {
        self.children[byte as usize]
    }

    /// Set child at given byte index.
    ///
    /// Updates both bitmap and children array.
    ///
    /// # Arguments
    /// * `byte` - Byte value (0-255)
    /// * `child_idx` - Arena index of child node
    ///
    /// # Performance
    /// O(1) - bitmap update + array write
    #[inline(always)]
    pub fn set_child(&mut self, byte: u8, child_idx: u32) {
        bitmap::set_bit(&mut self.bitmap, byte);
        self.children[byte as usize] = child_idx;
    }

    /// Clear child at given byte index.
    ///
    /// Updates both bitmap and children array.
    ///
    /// # Arguments
    /// * `byte` - Byte value (0-255)
    ///
    /// # Performance
    /// O(1) - bitmap update + array write
    #[inline]
    pub fn clear_child(&mut self, byte: u8) {
        bitmap::clear_bit(&mut self.bitmap, byte);
        self.children[byte as usize] = EMPTY;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_has_child() {
        let mut node = Node::new();

        // Initially no children
        assert!(!node.has_child(0));
        assert!(!node.has_child(42));
        assert!(!node.has_child(255));

        // Set a child
        node.set_child(42, 100);
        assert!(node.has_child(42));
        assert!(!node.has_child(41));
        assert!(!node.has_child(43));
    }

    #[test]
    fn test_get_child() {
        let mut node = Node::new();

        // Initially all EMPTY
        assert_eq!(node.get_child(0), EMPTY);
        assert_eq!(node.get_child(42), EMPTY);

        // Set children
        node.set_child(0, 10);
        node.set_child(42, 100);
        node.set_child(255, 999);

        assert_eq!(node.get_child(0), 10);
        assert_eq!(node.get_child(42), 100);
        assert_eq!(node.get_child(255), 999);
        assert_eq!(node.get_child(1), EMPTY);
    }

    #[test]
    fn test_set_child() {
        let mut node = Node::new();

        node.set_child(42, 100);

        // Bitmap updated
        assert!(node.has_child(42));

        // Children array updated
        assert_eq!(node.get_child(42), 100);

        // Can overwrite
        node.set_child(42, 200);
        assert_eq!(node.get_child(42), 200);
    }

    #[test]
    fn test_clear_child() {
        let mut node = Node::new();

        // Set then clear
        node.set_child(42, 100);
        assert!(node.has_child(42));
        assert_eq!(node.get_child(42), 100);

        node.clear_child(42);
        assert!(!node.has_child(42));
        assert_eq!(node.get_child(42), EMPTY);
    }

    #[test]
    fn test_multiple_children() {
        let mut node = Node::new();

        // Set multiple children
        for i in 0..10 {
            node.set_child(i * 25, i as u32 * 100);
        }

        // Verify all set
        for i in 0..10 {
            let byte = i * 25;
            assert!(node.has_child(byte));
            assert_eq!(node.get_child(byte), i as u32 * 100);
        }

        // Clear some
        node.clear_child(50);
        node.clear_child(100);

        assert!(!node.has_child(50));
        assert!(!node.has_child(100));
        assert!(node.has_child(0));
        assert!(node.has_child(25));
    }
}
