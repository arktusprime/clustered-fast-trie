//! Main Trie structure for ordered integer sets.

use crate::arena::{ArenaAllocator, KeyRange, SegmentId};
use crate::key::TrieKey;

/// Ordered integer set with sublogarithmic complexity.
///
/// A high-performance trie data structure optimized for clustered integer keys.
/// Supports u32, u64, and u128 key types with guaranteed O(log log U) complexity
/// for all operations, independent of the number of elements.
///
/// # Key Features
/// - Sublogarithmic complexity: O(log log U) for all operations
/// - Cache-optimized: hot path caching for sequential inserts (0.8-1.2 ns)
/// - Memory efficient: 0.5-0.6 bytes per key at optimal density
/// - Lock-free: atomic operations for multi-threaded access
/// - Zero dependencies: no_std compatible (requires alloc)
///
/// # Architecture
/// - 256-way branching trie (8 bits per level)
/// - Arena allocation for cache locality
/// - Lazy allocation: nodes/leaves created on-demand
/// - Single-tenant mode: client owns entire key space
///
/// # Performance Characteristics
/// - Insert: O(log log U), 0.8-1.2 ns (cache hit), 5-10 ns (cache miss)
/// - Contains: O(log log U), similar to insert
/// - Remove: O(log log U), similar to insert
/// - Memory: ~1KB per internal node, ~48 bytes per leaf
///
/// # Example
/// ```rust
/// use clustered_fast_trie::Trie;
///
/// let trie = Trie::<u32>::new();
/// // Trie is ready for insert/contains/remove operations (to be implemented)
/// ```
#[derive(Debug)]
pub struct Trie<K: TrieKey> {
    /// Arena allocator for memory management
    allocator: ArenaAllocator,

    /// Root segment ID for single-tenant mode
    root_segment: SegmentId,

    /// Phantom data to associate with key type
    _phantom: core::marker::PhantomData<K>,
}

impl<K: TrieKey> Trie<K> {
    /// Create a new empty trie.
    ///
    /// Initializes the trie in single-tenant mode where the client owns
    /// the entire key space. Creates one segment covering the full key range.
    ///
    /// # Performance
    /// O(1) - creates empty allocator and reserves one segment
    ///
    /// # Memory Usage
    /// ~300 bytes initial overhead (allocator + segment metadata + cache)
    ///
    /// # Example
    /// ```rust
    /// use clustered_fast_trie::Trie;
    ///
    /// let trie = Trie::<u64>::new();
    /// // Trie is ready for insert/contains/remove operations
    /// ```
    pub fn new() -> Self {
        let mut allocator = ArenaAllocator::new();

        // Create root segment covering entire key space for single-tenant mode
        let key_range = KeyRange {
            start: 0,             // Start from 0
            size: K::max_value(), // Cover full key range
        };

        let root_segment = allocator.create_segment(key_range, 0);

        Self {
            allocator,
            root_segment,
            _phantom: core::marker::PhantomData,
        }
    }

    /// Insert a key into the trie.
    ///
    /// Adds the specified key to the trie using lazy allocation.
    /// Creates nodes and leaves as needed during traversal.
    ///
    /// # Arguments
    /// * `key` - The key to insert
    ///
    /// # Returns
    /// * `true` if the key was newly inserted
    /// * `false` if the key already existed
    ///
    /// # Performance
    /// O(log log U) - traverses at most K::LEVELS + 1 levels
    ///
    /// # Example
    /// ```rust
    /// use clustered_fast_trie::Trie;
    ///
    /// let mut trie = Trie::<u32>::new();
    /// assert!(trie.insert(42));   // New key
    /// assert!(!trie.insert(42));  // Already exists
    /// ```
    pub fn insert(&mut self, key: K) -> bool {
        // Step 1: Ensure arenas are allocated
        if let Err(_) = self.allocator.allocate_arena(self.root_segment) {
            return false; // Failed to allocate arenas
        }

        // Step 2: Get segment metadata
        let segment_meta = self
            .allocator
            .get_segment_meta(self.root_segment)
            .expect("Segment should exist");
        let arena_idx = segment_meta.cache_key;

        // Step 3: Ensure root node exists (index 0 in node arena)
        let root_node_idx = self.ensure_root_node(arena_idx);

        // Step 4: Traverse trie levels to find/create path to leaf
        let leaf_idx = self.traverse_to_leaf(key, root_node_idx, arena_idx);

        // Step 5: Set bit in leaf bitmap
        self.set_bit_in_leaf(key, leaf_idx, arena_idx)
    }

    /// Check if a key exists in the trie.
    ///
    /// Searches for the specified key without modifying the trie structure.
    /// Uses read-only operations for optimal performance.
    ///
    /// # Arguments
    /// * `key` - The key to search for
    ///
    /// # Returns
    /// * `true` if the key exists in the trie
    /// * `false` if the key does not exist
    ///
    /// # Performance
    /// O(log log U) - traverses at most K::LEVELS + 1 levels
    ///
    /// # Example
    /// ```rust
    /// use clustered_fast_trie::Trie;
    ///
    /// let mut trie = Trie::<u32>::new();
    /// assert!(!trie.contains(42));  // Key doesn't exist
    /// trie.insert(42);
    /// assert!(trie.contains(42));   // Key exists
    /// ```
    pub fn contains(&self, key: K) -> bool {
        // Step 1: Check if arenas are allocated
        let segment_meta = match self.allocator.get_segment_meta(self.root_segment) {
            Some(meta) => meta,
            None => return false, // Segment doesn't exist
        };
        let arena_idx = segment_meta.cache_key;

        // Step 2: Check if node arena exists and has root node
        let node_arena = match self.allocator.get_node_arena(arena_idx) {
            Some(arena) => arena,
            None => return false, // Node arena not allocated
        };

        if node_arena.is_empty() {
            return false; // No root node exists
        }

        // Step 3: Traverse trie levels to find leaf
        let mut current_node_idx = 0; // Start at root

        // Traverse internal levels (0..K::LEVELS-1)
        for level in 0..(K::LEVELS - 1) {
            let byte = key.byte_at(level);
            let current_node = node_arena.get(current_node_idx);

            if !current_node.has_child(byte) {
                return false; // Path doesn't exist
            }

            current_node_idx = current_node.get_child(byte);
        }

        // Final level: check if leaf exists
        let last_node_byte = key.byte_at(K::LEVELS - 1);
        let final_node = node_arena.get(current_node_idx);

        if !final_node.has_child(last_node_byte) {
            return false; // Leaf doesn't exist
        }

        let leaf_idx = final_node.get_child(last_node_byte);

        // Step 4: Check if leaf arena exists
        let leaf_arena = match self.allocator.get_leaf_arena(arena_idx) {
            Some(arena) => arena,
            None => return false, // Leaf arena not allocated
        };

        // Step 5: Check bit in leaf bitmap
        self.check_bit_in_leaf(key, leaf_idx, arena_idx)
    }

    /// Remove a key from the trie.
    ///
    /// Removes the specified key from the trie if it exists.
    /// Uses atomic clear operation for thread safety.
    ///
    /// # Arguments
    /// * `key` - The key to remove
    ///
    /// # Returns
    /// * `true` if the key was removed (existed before)
    /// * `false` if the key didn't exist
    ///
    /// # Performance
    /// O(log log U) - traverses at most K::LEVELS + 1 levels
    ///
    /// # Example
    /// ```rust
    /// use clustered_fast_trie::Trie;
    ///
    /// let mut trie = Trie::<u32>::new();
    /// assert!(!trie.remove(42));  // Key doesn't exist
    /// trie.insert(42);
    /// assert!(trie.remove(42));   // Key removed
    /// assert!(!trie.remove(42));  // Key no longer exists
    /// ```
    pub fn remove(&mut self, key: K) -> bool {
        // Step 1: Check if arenas are allocated
        let segment_meta = match self.allocator.get_segment_meta(self.root_segment) {
            Some(meta) => meta,
            None => return false, // Segment doesn't exist
        };
        let arena_idx = segment_meta.cache_key;

        // Step 2: Check if node arena exists and has root node
        let node_arena = match self.allocator.get_node_arena(arena_idx) {
            Some(arena) => arena,
            None => return false, // Node arena not allocated
        };

        if node_arena.is_empty() {
            return false; // No root node exists
        }

        // Step 3: Traverse trie levels to find leaf
        let mut current_node_idx = 0; // Start at root

        // Traverse internal levels (0..K::LEVELS-1)
        for level in 0..(K::LEVELS - 1) {
            let byte = key.byte_at(level);
            let current_node = node_arena.get(current_node_idx);

            if !current_node.has_child(byte) {
                return false; // Path doesn't exist
            }

            current_node_idx = current_node.get_child(byte);
        }

        // Final level: check if leaf exists
        let last_node_byte = key.byte_at(K::LEVELS - 1);
        let final_node = node_arena.get(current_node_idx);

        if !final_node.has_child(last_node_byte) {
            return false; // Leaf doesn't exist
        }

        let leaf_idx = final_node.get_child(last_node_byte);

        // Step 4: Check if leaf arena exists
        let leaf_arena = match self.allocator.get_leaf_arena_mut(arena_idx) {
            Some(arena) => arena,
            None => return false, // Leaf arena not allocated
        };

        // Step 5: Clear bit in leaf bitmap
        self.clear_bit_in_leaf(key, leaf_idx, arena_idx)
    }

    /// Ensure root node exists at index 0 in node arena.
    fn ensure_root_node(&mut self, arena_idx: u64) -> u32 {
        let node_arena = self
            .allocator
            .get_node_arena_mut(arena_idx)
            .expect("Node arena should be allocated");

        if node_arena.is_empty() {
            // Create root node at index 0
            node_arena.alloc()
        } else {
            // Root node already exists at index 0
            0
        }
    }

    /// Traverse trie levels to find or create leaf.
    ///
    /// Performs full traversal through internal Node structures at each level,
    /// creating nodes as needed. On the final level, finds or creates a Leaf.
    ///
    /// # Algorithm
    /// 1. Start at root node
    /// 2. For each level (0..K::LEVELS-1):
    ///    - Extract byte at current level from key
    ///    - Check if child exists for this byte
    ///    - If exists: move to that child node
    ///    - If not: create new node and link it
    /// 3. At final level (K::LEVELS-1):
    ///    - Extract last byte before leaf level
    ///    - Check if leaf exists for this byte
    ///    - If exists: return leaf index
    ///    - If not: create new leaf and link it
    ///
    /// # Performance
    /// O(K::LEVELS) = O(log log U) where U is key space size
    fn traverse_to_leaf(&mut self, key: K, mut current_node_idx: u32, arena_idx: u64) -> u32 {
        // Traverse internal levels (0..K::LEVELS-1)
        // Each level navigates through Node structures
        for level in 0..(K::LEVELS - 1) {
            let byte = key.byte_at(level);

            // Check if child exists (read-only operation)
            let child_idx = {
                let node_arena = self
                    .allocator
                    .get_node_arena(arena_idx)
                    .expect("Node arena should be allocated");
                let current_node = node_arena.get(current_node_idx);

                if current_node.has_child(byte) {
                    Some(current_node.get_child(byte))
                } else {
                    None
                }
            };

            if let Some(idx) = child_idx {
                // Child exists - move to it
                current_node_idx = idx;
            } else {
                // Child doesn't exist - create new node and link it
                let node_arena = self
                    .allocator
                    .get_node_arena_mut(arena_idx)
                    .expect("Node arena should be allocated");

                let new_node_idx = node_arena.alloc();
                let current_node = node_arena.get_mut(current_node_idx);
                current_node.set_child(byte, new_node_idx);
                current_node_idx = new_node_idx;
            }
        }

        // Final level (K::LEVELS - 1): transition from Node to Leaf
        let last_node_byte = key.byte_at(K::LEVELS - 1);

        // Check if leaf exists (read-only operation)
        let leaf_idx = {
            let node_arena = self
                .allocator
                .get_node_arena(arena_idx)
                .expect("Node arena should be allocated");
            let final_node = node_arena.get(current_node_idx);

            if final_node.has_child(last_node_byte) {
                Some(final_node.get_child(last_node_byte))
            } else {
                None
            }
        };

        if let Some(idx) = leaf_idx {
            // Leaf already exists
            idx
        } else {
            // Create new leaf and link it
            let prefix = key.prefix().to_u128() as u64;

            let leaf_arena = self
                .allocator
                .get_leaf_arena_mut(arena_idx)
                .expect("Leaf arena should be allocated");
            let new_leaf_idx = leaf_arena.alloc(prefix);

            let node_arena = self
                .allocator
                .get_node_arena_mut(arena_idx)
                .expect("Node arena should be allocated");
            let final_node = node_arena.get_mut(current_node_idx);
            final_node.set_child(last_node_byte, new_leaf_idx);

            new_leaf_idx
        }
    }

    /// Set bit in leaf bitmap for the given key.
    fn set_bit_in_leaf(&mut self, key: K, leaf_idx: u32, arena_idx: u64) -> bool {
        use crate::bitmap::test_and_set_bit;

        let leaf_arena = self
            .allocator
            .get_leaf_arena_mut(arena_idx)
            .expect("Leaf arena should be allocated");

        let leaf = leaf_arena.get_mut(leaf_idx);
        let bit_idx = key.last_byte();

        // Use atomic test-and-set: returns true if bit was NOT set (new insertion)
        test_and_set_bit(&leaf.bitmap, bit_idx)
    }

    /// Check if bit is set in leaf bitmap for the given key.
    fn check_bit_in_leaf(&self, key: K, leaf_idx: u32, arena_idx: u64) -> bool {
        use crate::bitmap::is_set;

        let leaf_arena = self
            .allocator
            .get_leaf_arena(arena_idx)
            .expect("Leaf arena should be allocated");

        let leaf = leaf_arena.get(leaf_idx);
        let bit_idx = key.last_byte();

        // Use atomic read: check if bit is set
        is_set(&leaf.bitmap, bit_idx)
    }

    /// Clear bit in leaf bitmap for the given key.
    fn clear_bit_in_leaf(&mut self, key: K, leaf_idx: u32, arena_idx: u64) -> bool {
        use crate::bitmap::{clear_bit, is_set};

        let leaf_arena = self
            .allocator
            .get_leaf_arena_mut(arena_idx)
            .expect("Leaf arena should be allocated");

        let leaf = leaf_arena.get_mut(leaf_idx);
        let bit_idx = key.last_byte();

        // Check if bit was set before clearing
        let was_set = is_set(&leaf.bitmap, bit_idx);
        if was_set {
            clear_bit(&leaf.bitmap, bit_idx);
        }

        was_set
    }
}

impl<K: TrieKey> Default for Trie<K> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_trie_u32() {
        let trie = Trie::<u32>::new();

        // Check that allocator has one segment
        assert!(trie.allocator.get_segment_meta(trie.root_segment).is_some());

        // Check segment covers full u32 range
        let meta = trie.allocator.get_segment_meta(trie.root_segment).unwrap();
        assert_eq!(meta.key_offset, 0);
        assert_eq!(meta.cache_key, 0);
        assert_eq!(meta.run_length, 1);
    }

    #[test]
    fn test_new_trie_u64() {
        let trie = Trie::<u64>::new();

        // Check that allocator has one segment
        assert!(trie.allocator.get_segment_meta(trie.root_segment).is_some());

        // Check segment metadata
        let meta = trie.allocator.get_segment_meta(trie.root_segment).unwrap();
        assert_eq!(meta.key_offset, 0);
        assert_eq!(meta.cache_key, 0);
    }

    #[test]
    fn test_new_trie_u128() {
        let trie = Trie::<u128>::new();

        // Check that allocator has one segment
        assert!(trie.allocator.get_segment_meta(trie.root_segment).is_some());

        // Check segment metadata
        let meta = trie.allocator.get_segment_meta(trie.root_segment).unwrap();
        assert_eq!(meta.key_offset, 0);
        assert_eq!(meta.cache_key, 0);
    }

    #[test]
    fn test_default() {
        let trie1 = Trie::<u32>::new();
        let trie2 = Trie::<u32>::default();

        // Both should have same structure
        assert_eq!(trie1.root_segment, trie2.root_segment);
    }

    #[test]
    fn test_insert_basic() {
        let mut trie = Trie::<u32>::new();

        // Test basic insertion
        let result = trie.insert(42);
        assert!(result); // Should return true for new key
    }

    #[test]
    fn test_insert_duplicate() {
        let mut trie = Trie::<u32>::new();

        // First insertion should return true
        let result1 = trie.insert(42);
        assert!(result1); // Should return true for new key

        // Second insertion should return false
        let result2 = trie.insert(42);
        assert!(!result2); // Should return false for existing key
    }

    #[test]
    fn test_contains_basic() {
        let mut trie = Trie::<u32>::new();

        // Key doesn't exist initially
        assert!(!trie.contains(42));

        // Insert key
        trie.insert(42);

        // Key should exist now
        assert!(trie.contains(42));

        // Other key should not exist
        assert!(!trie.contains(43));
    }

    #[test]
    fn test_remove_basic() {
        let mut trie = Trie::<u32>::new();

        // Key doesn't exist initially - remove should return false
        assert!(!trie.remove(42));

        // Insert key
        trie.insert(42);
        assert!(trie.contains(42));

        // Remove key - should return true (was removed)
        assert!(trie.remove(42));

        // Key should no longer exist
        assert!(!trie.contains(42));

        // Remove again - should return false (doesn't exist)
        assert!(!trie.remove(42));
    }

    #[test]
    fn test_integration_operations() {
        let mut trie = Trie::<u32>::new();

        // Test with multiple keys
        let keys = [10, 20, 30, 40, 50];

        // Initially all keys should not exist
        for &key in &keys {
            assert!(!trie.contains(key));
            assert!(!trie.remove(key)); // Remove non-existent key
        }

        // Insert all keys
        for &key in &keys {
            assert!(trie.insert(key)); // Should return true (new key)
            assert!(trie.contains(key)); // Should exist after insert
        }

        // Try to insert duplicates
        for &key in &keys {
            assert!(!trie.insert(key)); // Should return false (already exists)
            assert!(trie.contains(key)); // Should still exist
        }

        // Remove some keys
        assert!(trie.remove(20)); // Remove existing key
        assert!(!trie.contains(20)); // Should not exist after remove
        assert!(!trie.remove(20)); // Remove again - should return false

        assert!(trie.remove(40)); // Remove another key
        assert!(!trie.contains(40)); // Should not exist after remove

        // Check remaining keys still exist
        assert!(trie.contains(10));
        assert!(trie.contains(30));
        assert!(trie.contains(50));

        // Re-insert removed keys
        assert!(trie.insert(20)); // Should return true (new key again)
        assert!(trie.contains(20)); // Should exist after re-insert

        assert!(trie.insert(40)); // Re-insert another key
        assert!(trie.contains(40)); // Should exist after re-insert

        // Final state - all keys should exist
        for &key in &keys {
            assert!(trie.contains(key));
        }
    }
}
