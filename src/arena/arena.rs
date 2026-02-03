//! Arena structure for Node and Leaf storage

extern crate alloc;
use alloc::vec::Vec;

use crate::trie::{Leaf, Node};

/// Generic arena for storing trie elements (Node or Leaf).
///
/// Provides contiguous memory allocation with O(1) access by index.
/// Uses Vec for dynamic growth and cache-friendly layout.
///
/// # Type Parameters
/// * `T` - Element type (Node or Leaf)
///
/// # Memory Layout
/// - Elements stored contiguously in Vec
/// - Index-based access (u32 indices)
/// - Grows dynamically as needed
///
/// # Performance
/// - Allocation: O(1) amortized
/// - Access: O(1) by index
/// - Cache-friendly: sequential layout
#[derive(Debug)]
pub struct Arena<T> {
    /// Storage for elements.
    ///
    /// Index in this Vec is the arena index used for references.
    elements: Vec<T>,
}

#[allow(dead_code)]
impl<T> Arena<T> {
    /// Create a new empty arena.
    ///
    /// # Returns
    /// Empty arena with no allocated elements
    ///
    /// # Performance
    /// O(1) - creates empty Vec
    #[inline(always)]
    pub fn new() -> Self {
        Arena {
            elements: Vec::new(),
        }
    }

    /// Create arena with pre-allocated capacity.
    ///
    /// # Arguments
    /// * `capacity` - Number of elements to pre-allocate
    ///
    /// # Returns
    /// Empty arena with reserved capacity
    ///
    /// # Performance
    /// O(capacity) - allocates memory upfront
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Arena {
            elements: Vec::with_capacity(capacity),
        }
    }

    /// Get element by index.
    ///
    /// # Arguments
    /// * `index` - Arena index (u32)
    ///
    /// # Returns
    /// Reference to element at index
    ///
    /// # Panics
    /// Panics if index is out of bounds
    ///
    /// # Performance
    /// O(1) - direct Vec indexing
    #[inline(always)]
    pub fn get(&self, index: u32) -> &T {
        &self.elements[index as usize]
    }

    /// Get mutable element by index.
    ///
    /// # Arguments
    /// * `index` - Arena index (u32)
    ///
    /// # Returns
    /// Mutable reference to element at index
    ///
    /// # Panics
    /// Panics if index is out of bounds
    ///
    /// # Performance
    /// O(1) - direct Vec indexing
    #[inline(always)]
    pub fn get_mut(&mut self, index: u32) -> &mut T {
        &mut self.elements[index as usize]
    }

    /// Get number of allocated elements.
    ///
    /// # Returns
    /// Number of elements in arena
    ///
    /// # Performance
    /// O(1) - Vec length
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.elements.len()
    }

    /// Check if arena is empty.
    ///
    /// # Returns
    /// true if no elements allocated
    ///
    /// # Performance
    /// O(1) - Vec is_empty
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }
}

impl Arena<Node> {
    /// Allocate a new node.
    ///
    /// # Returns
    /// Arena index of the newly allocated node
    ///
    /// # Performance
    /// O(1) amortized - Vec push
    ///
    /// Hot path - always inlined.
    #[inline(always)]
    pub fn alloc(&mut self) -> u32 {
        let index = self.elements.len() as u32;
        self.elements.push(Node::new());
        index
    }
}

impl Arena<Leaf> {
    /// Allocate a new leaf with given prefix.
    ///
    /// # Arguments
    /// * `prefix` - Key prefix for this leaf (last byte should be 0)
    ///
    /// # Returns
    /// Arena index of the newly allocated leaf
    ///
    /// # Performance
    /// O(1) amortized - Vec push
    ///
    /// Hot path - always inlined.
    #[inline(always)]
    pub fn alloc(&mut self, prefix: u64) -> u32 {
        let index = self.elements.len() as u32;
        self.elements.push(Leaf::new(prefix));
        index
    }
}

impl<T> Default for Arena<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Type alias for Node arena.
///
/// Used for storing internal trie nodes at all levels.
#[allow(dead_code)]
pub type NodeArena = Arena<Node>;

/// Type alias for Leaf arena.
///
/// Used for storing leaf nodes (256-bit bitmaps).
#[allow(dead_code)]
pub type LeafArena = Arena<Leaf>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_new() {
        let arena: Arena<Node> = Arena::new();
        assert_eq!(arena.len(), 0);
        assert!(arena.is_empty());
    }

    #[test]
    fn test_arena_with_capacity() {
        let arena: Arena<Node> = Arena::with_capacity(100);
        assert_eq!(arena.len(), 0);
        assert!(arena.is_empty());
    }

    #[test]
    fn test_node_arena_alloc() {
        let mut arena = Arena::<Node>::new();

        let idx0 = arena.alloc();
        assert_eq!(idx0, 0);
        assert_eq!(arena.len(), 1);

        let idx1 = arena.alloc();
        assert_eq!(idx1, 1);
        assert_eq!(arena.len(), 2);
    }

    #[test]
    fn test_leaf_arena_alloc() {
        let mut arena = Arena::<Leaf>::new();

        let idx0 = arena.alloc(0x12345600);
        assert_eq!(idx0, 0);
        assert_eq!(arena.len(), 1);
        assert_eq!(arena.get(idx0).prefix, 0x12345600);

        let idx1 = arena.alloc(0xABCDEF00);
        assert_eq!(idx1, 1);
        assert_eq!(arena.len(), 2);
        assert_eq!(arena.get(idx1).prefix, 0xABCDEF00);
    }

    #[test]
    fn test_arena_get() {
        use crate::atomic::Ordering;

        let mut arena = Arena::<Node>::new();
        let idx = arena.alloc();

        let node = arena.get(idx);
        assert_eq!(node.seq.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_arena_get_mut() {
        use crate::constants::EMPTY;

        let mut arena = Arena::<Node>::new();
        let idx = arena.alloc();

        let node = arena.get_mut(idx);
        node.children[0] = 42;

        assert_eq!(arena.get(idx).children[0], 42);
        assert_eq!(arena.get(idx).children[1], EMPTY);
    }

    #[test]
    fn test_default() {
        let arena: Arena<Node> = Arena::default();
        assert!(arena.is_empty());
    }
}
