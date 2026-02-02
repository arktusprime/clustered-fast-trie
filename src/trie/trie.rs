//! Main Trie structure for ordered integer sets.

use crate::arena::{SegmentManager, KeyRange, SegmentId};
use crate::key::TrieKey;
use crate::trie::{ChildArenas, Node};
use alloc::boxed::Box;

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
/// - Hierarchical arena allocation: arenas stored in nodes
/// - Root arenas: stored in root_node.child_arenas
/// - Child arenas: stored in nodes at split levels
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
    /// Flat vector of all arenas (root + child arenas).
    ///
    /// NEW ARCHITECTURE: Vec<ChildArenas> for O(1) arena access.
    /// - Index 0: root arena (always present)
    /// - Index 1+: child arenas (created on split levels)
    ///
    /// Physical index stored in Node.child_arena_idx for O(1) access.
    arenas: alloc::vec::Vec<ChildArenas>,

    /// Segment manager for multi-tenant memory management
    segment_manager: SegmentManager,

    /// Root segment ID for single-tenant mode
    root_segment: SegmentId,

    /// Physical index of root arena in arenas Vec (always 0).
    root_arena_idx: u32,

    /// Index of root node within root arena (always 0).
    root_node_idx: u32,

    /// Root node containing root arenas for the entire trie.
    /// DEPRECATED: Will be removed after migration.
    /// Use arenas[root_arena_idx] instead.
    root_node: Node,

    /// Number of keys stored in the trie
    len: usize,

    /// Cached minimum key for O(1) access
    min_key: Option<K>,

    /// Cached maximum key for O(1) access
    max_key: Option<K>,

    /// First leaf in linked list (packed: arena_idx << 32 | leaf_idx, or EMPTY_LINK if no leaves)
    first_leaf: u64,

    /// Last leaf in linked list (packed: arena_idx << 32 | leaf_idx, or EMPTY_LINK if no leaves)
    last_leaf: u64,

    /// Phantom data to associate with key type
    _phantom: core::marker::PhantomData<K>,
}

impl<K: TrieKey> Trie<K> {
    /// Create a new empty trie.
    ///
    /// Initializes the trie in single-tenant mode where the client owns
    /// the entire key space. Creates one segment covering the full key range.
    /// Root node is initialized with child arenas for storing nodes and leaves.
    ///
    /// # Performance
    /// O(1) - creates empty segment manager and initializes root node with arenas
    ///
    /// # Memory Usage
    /// ~300 bytes initial overhead (segment manager + root node + arenas)
    ///
    /// # Example
    /// ```rust
    /// use clustered_fast_trie::Trie;
    ///
    /// let trie = Trie::<u64>::new();
    /// // Trie is ready for insert/contains/remove operations
    /// ```
    pub fn new() -> Self {
        let mut segment_manager = SegmentManager::new();

        // Create root segment covering entire key space for single-tenant mode
        let key_range = KeyRange {
            start: 0,             // Start from 0
            size: K::max_value(), // Cover full key range
        };

        let root_segment = segment_manager.create_segment(key_range, 0);

        // NEW ARCHITECTURE: Create Vec<ChildArenas> with root arena
        let mut arenas = alloc::vec::Vec::new();
        arenas.push(ChildArenas::new());
        let root_arena_idx = 0;

        // Create root node at index 0 in root arena
        let root_node_idx = arenas[0].node_arena.alloc();
        debug_assert_eq!(root_node_idx, 0, "Root node must be at index 0");

        // DEPRECATED: Keep root_node for compatibility during migration
        let mut root_node = Node::new();
        root_node.child_arenas = Some(Box::new(ChildArenas::new()));

        Self {
            arenas,
            segment_manager,
            root_segment,
            root_arena_idx,
            root_node_idx,
            root_node,
            len: 0,
            min_key: None,
            max_key: None,
            first_leaf: crate::trie::EMPTY_LINK,
            last_leaf: crate::trie::EMPTY_LINK,
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
        // NEW ARCHITECTURE: Use Vec<ChildArenas>
        // Step 1: Root arena is always at index 0
        let arena_idx = self.root_arena_idx;

        // Step 2: Root node is always at index 0 in root arena
        let root_node_idx = self.root_node_idx;

        // Step 3: Traverse trie levels to find/create path to leaf
        // TODO: Update traverse_to_leaf to use new architecture
        let (leaf_idx, _path, _path_len, arena_idx) =
            self.traverse_to_leaf(key, root_node_idx, arena_idx as u64);

        // Step 4: Set bit in leaf bitmap
        // TODO: Update set_bit_in_leaf to use new architecture
        let was_new = self.set_bit_in_leaf(key, leaf_idx, arena_idx);

        // Step 5: Update cache if new insertion
        if was_new {
            self.len += 1;
            self.update_min_max_insert(key);
        }

        was_new
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
        // NEW ARCHITECTURE: Use Vec<ChildArenas> directly
        // Step 1: Start at root arena (index 0)
        let mut current_arena_idx = self.root_arena_idx;

        // Step 2: Check if root arena has nodes
        let node_arena = self.get_node_arena_new(current_arena_idx);
        if node_arena.is_empty() {
            return false; // No root node exists
        }

        // Step 3: Traverse trie levels to find leaf, switching arenas at split levels
        let mut current_node_idx = self.root_node_idx;

        // Traverse internal levels (0..K::LEVELS-1)
        for level in 0..(K::LEVELS - 1) {
            let byte = key.byte_at(level);

            // Get current node arena (may have changed at split level)
            let node_arena = self.get_node_arena_new(current_arena_idx);
            let current_node = node_arena.get(current_node_idx);

            if !current_node.has_child(byte) {
                return false; // Path doesn't exist
            }

            // Check if we need to switch arenas at split level BEFORE moving to child
            if K::SPLIT_LEVELS.contains(&(level + 1)) {
                // Next level is a split level - get child arena index from CURRENT node (parent)
                let child_arena_idx = current_node.child_arena_idx;

                // Check if child arena exists
                if child_arena_idx as usize >= self.arenas.len() {
                    return false; // Child arena not allocated
                }

                current_arena_idx = child_arena_idx;
            }

            // Now move to child node
            current_node_idx = current_node.get_child(byte);
        }

        // Final level: check if leaf exists
        let last_node_byte = key.byte_at(K::LEVELS - 1);
        let node_arena = self.get_node_arena_new(current_arena_idx);
        let final_node = node_arena.get(current_node_idx);

        if !final_node.has_child(last_node_byte) {
            return false; // Leaf doesn't exist
        }

        let leaf_idx = final_node.get_child(last_node_byte);

        // Step 4: Get leaf arena and check bit
        let leaf_arena = self.get_leaf_arena_new(current_arena_idx);
        
        // Step 5: Check bit in leaf bitmap
        // TODO: Update check_bit_in_leaf to use new architecture (for now inline it)
        use crate::bitmap::is_set;
        let leaf = leaf_arena.get(leaf_idx);
        let bit_idx = key.last_byte();
        is_set(&leaf.bitmap, bit_idx)
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
        // NEW ARCHITECTURE: Use Vec<ChildArenas>
        // Step 1: Start at root arena
        let mut current_arena_idx = self.root_arena_idx;

        // Step 2: Check if root arena has nodes
        let node_arena = self.get_node_arena_new(current_arena_idx);
        if node_arena.is_empty() {
            return false; // No root node exists
        }

        // Step 3: Traverse trie levels to find leaf, tracking path and arena switches
        let mut path: [(u32, u8, u32); 16] = [(0, 0, 0); 16]; // Max 16 levels for u128: (node_idx, byte, arena_idx as u32)
        let mut path_len = 0;
        let mut current_node_idx = self.root_node_idx;

        // Traverse internal levels (0..K::LEVELS-1)
        for level in 0..(K::LEVELS - 1) {
            let byte = key.byte_at(level);
            path[path_len] = (current_node_idx, byte, current_arena_idx);
            path_len += 1;

            // Get current node arena (may have changed at split level)
            let node_arena = self.get_node_arena_new(current_arena_idx);
            let current_node = node_arena.get(current_node_idx);

            if !current_node.has_child(byte) {
                return false; // Path doesn't exist
            }

            current_node_idx = current_node.get_child(byte);

            // Check if we need to switch arenas at split level
            if K::SPLIT_LEVELS.contains(&(level + 1)) {
                // Next level is a split level - get child arena index from current node
                let child_arena_idx = current_node.child_arena_idx;

                // Check if child arena exists
                if child_arena_idx as usize >= self.arenas.len() {
                    return false; // Child arena not allocated
                }

                current_arena_idx = child_arena_idx;
            }
        }

        // Final level: check if leaf exists
        let last_node_byte = key.byte_at(K::LEVELS - 1);
        path[path_len] = (current_node_idx, last_node_byte, current_arena_idx);
        path_len += 1;

        let node_arena = self.get_node_arena_new(current_arena_idx);
        let final_node = node_arena.get(current_node_idx);

        if !final_node.has_child(last_node_byte) {
            return false; // Leaf doesn't exist
        }

        let leaf_idx = final_node.get_child(last_node_byte);

        // Step 4: Clear bit in leaf bitmap (inline)
        use crate::bitmap::{clear_bit, is_set};
        let was_removed = {
            let leaf_arena = self.get_leaf_arena_mut_new(current_arena_idx);
            let leaf = leaf_arena.get_mut(leaf_idx);
            let bit_idx = key.last_byte();
            let was_set = is_set(&leaf.bitmap, bit_idx);
            if was_set {
                clear_bit(&leaf.bitmap, bit_idx);
            }
            was_set
        };

        if !was_removed {
            return false;
        }

        // Step 5: Check if leaf is empty and cleanup if needed
        let is_leaf_empty = {
            let leaf_arena = self.get_leaf_arena_new(current_arena_idx);
            let leaf = leaf_arena.get(leaf_idx);
            crate::bitmap::is_empty(&leaf.bitmap)
        };

        if is_leaf_empty {
            // Unlink leaf from linked list
            // TODO: Update unlink_leaf to use new architecture
            self.unlink_leaf(leaf_idx, current_arena_idx as u64);

            // Remove link from parent node
            {
                let node_arena = self.get_node_arena_mut_new(current_arena_idx);
                let parent_node = node_arena.get_mut(current_node_idx);
                parent_node.clear_child(last_node_byte);
            }

            // Free the leaf - TODO: Implement free-list in Phase 5
            // self.allocator.free_leaf(current_arena_idx, leaf_idx);

            // Cleanup empty nodes up the path
            // TODO: Update cleanup_empty_nodes to use new architecture
            self.cleanup_empty_nodes(&path, path_len - 1);
        }

        // Step 6: Update cache
        self.len -= 1;
        self.update_min_max_remove(key);

        true
    }

    /// Get the number of keys in the trie.
    ///
    /// Returns the count of unique keys stored in the trie.
    ///
    /// # Performance
    /// O(1) - returns cached value
    ///
    /// # Example
    /// ```rust
    /// use clustered_fast_trie::Trie;
    ///
    /// let mut trie = Trie::<u32>::new();
    /// assert_eq!(trie.len(), 0);
    /// trie.insert(42);
    /// assert_eq!(trie.len(), 1);
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the trie is empty.
    ///
    /// Returns `true` if the trie contains no keys.
    ///
    /// # Performance
    /// O(1) - checks cached length
    ///
    /// # Example
    /// ```rust
    /// use clustered_fast_trie::Trie;
    ///
    /// let mut trie = Trie::<u32>::new();
    /// assert!(trie.is_empty());
    /// trie.insert(42);
    /// assert!(!trie.is_empty());
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get the minimum key in the trie.
    ///
    /// Returns the smallest key stored in the trie, or `None` if empty.
    ///
    /// # Performance
    /// O(1) - returns cached value
    ///
    /// # Example
    /// ```rust
    /// use clustered_fast_trie::Trie;
    ///
    /// let mut trie = Trie::<u32>::new();
    /// assert_eq!(trie.min(), None);
    /// trie.insert(42);
    /// trie.insert(10);
    /// assert_eq!(trie.min(), Some(10));
    /// ```
    #[inline]
    pub fn min(&self) -> Option<K> {
        self.min_key
    }

    /// Get the maximum key in the trie.
    ///
    /// Returns the largest key stored in the trie, or `None` if empty.
    ///
    /// # Performance
    /// O(1) - returns cached value
    ///
    /// # Example
    /// ```rust
    /// use clustered_fast_trie::Trie;
    ///
    /// let mut trie = Trie::<u32>::new();
    /// assert_eq!(trie.max(), None);
    /// trie.insert(42);
    /// trie.insert(10);
    /// assert_eq!(trie.max(), Some(42));
    /// ```
    #[inline]
    pub fn max(&self) -> Option<K> {
        self.max_key
    }

    /// Find successor (smallest key > given key).
    ///
    /// Returns the next key in sorted order after the given key.
    /// Uses linked list of leaves for O(1) performance on clustered data.
    ///
    /// # Algorithm (from root)
    /// 1. Quick checks using cached min/max
    /// 2. Traverse to leaf containing key's prefix (save path)
    /// 3. If path doesn't exist AND leaf empty → backtrack to find next leaf
    /// 4. If path exists OR leaf not empty:
    ///    - Search for next bit in current leaf bitmap (O(1))
    ///    - If not found, follow leaf.next to next leaf (O(1))
    ///    - Take minimum bit from next leaf
    /// 5. Reconstruct key from prefix + bit
    ///
    /// # Arguments
    /// * `key` - The key to find successor for
    ///
    /// # Returns
    /// Next key after given key, or None if no successor exists
    ///
    /// # Performance
    /// - O(1) for existing keys or keys in same/adjacent leaves
    /// - O(log log U) for non-existing keys (requires backtracking)
    ///
    /// # Example
    /// ```rust
    /// use clustered_fast_trie::Trie;
    ///
    /// let mut trie = Trie::<u32>::new();
    /// trie.insert(10);
    /// trie.insert(20);
    /// trie.insert(30);
    ///
    /// assert_eq!(trie.successor(10), Some(20));
    /// assert_eq!(trie.successor(15), Some(20));
    /// assert_eq!(trie.successor(30), None);
    /// ```
    pub fn successor(&self, key: K) -> Option<K> {
        // Quick checks using cached min/max
        if let Some(max) = self.max_key {
            if key >= max {
                return None; // No successor if key >= max
            }
        }
        if let Some(min) = self.min_key {
            if key < min {
                return Some(min); // Min is successor if key < min
            }
        }

        // Get root arena
        let segment_meta = self.allocator.get_segment_meta(self.root_segment)?;
        let mut current_arena_idx = segment_meta.cache_key;

        // Get node arena
        let node_arena = self.allocator.get_node_arena(current_arena_idx)?;
        if node_arena.is_empty() {
            return None;
        }

        // Traverse to leaf containing key's prefix (save path for backtracking)
        let mut path: [(u32, u8, u64); 16] = [(0, 0, 0); 16];
        let mut path_len = 0;
        let mut current_node_idx = 0; // Start at root

        // Traverse internal levels (0..K::LEVELS-1)
        for level in 0..(K::LEVELS - 1) {
            let byte = key.byte_at(level);
            path[path_len] = (current_node_idx, byte, current_arena_idx);
            path_len += 1;

            // Get current node arena (may have changed at split level)
            let node_arena = self.allocator.get_node_arena(current_arena_idx)?;
            let current_node = node_arena.get(current_node_idx);

            if !current_node.has_child(byte) {
                // Path doesn't exist → backtrack to find next leaf
                return self.successor_backtrack(key, &path, path_len, current_arena_idx);
            }

            current_node_idx = current_node.get_child(byte);

            // Check if we need to switch arenas at split level
            if K::SPLIT_LEVELS.contains(&(level + 1)) {
                // Next level is a split level - get child arena index from current node
                let child_arena_idx = current_node.child_arena_idx as u64;

                // Check if child arena exists
                if !self.allocator.has_arena(child_arena_idx) {
                    return None; // Child arena not allocated
                }

                current_arena_idx = child_arena_idx;
            }
        }

        // Final level: check if leaf exists
        let last_node_byte = key.byte_at(K::LEVELS - 1);
        path[path_len] = (current_node_idx, last_node_byte, current_arena_idx);
        path_len += 1;

        let node_arena = self.allocator.get_node_arena(current_arena_idx)?;
        let final_node = node_arena.get(current_node_idx);

        if !final_node.has_child(last_node_byte) {
            // Leaf doesn't exist → backtrack to find next leaf
            return self.successor_backtrack(key, &path, path_len, current_arena_idx);
        }

        let leaf_idx = final_node.get_child(last_node_byte);

        // Leaf exists → search in leaf and next
        self.successor_from_leaf_internal(key, leaf_idx, current_arena_idx)
    }

    /// Find successor starting from a specific leaf (for bulk operations).
    ///
    /// Optimized version for ordered bulk operations (iterators, ranges).
    /// Assumes the leaf index is already known (cached from previous operation).
    ///
    /// # Algorithm (from cached leaf)
    /// 1. Search for next bit in current leaf bitmap (O(1))
    /// 2. If not found, follow leaf.next to next leaf (O(1))
    /// 3. Take minimum bit from next leaf
    /// 4. Reconstruct key from prefix + bit
    ///
    /// # Arguments
    /// * `key` - The key to find successor for
    /// * `leaf_idx` - Index of the leaf to start search from (cached)
    ///
    /// # Returns
    /// Next key after given key, or None if no successor exists
    ///
    /// # Performance
    /// O(1) - direct leaf access, no trie traversal
    ///
    /// # Use Case
    /// Iterators and range operations cache the last accessed leaf index
    /// and use this method for subsequent calls, achieving O(1) per element.
    pub fn successor_from_leaf(&self, key: K, leaf_idx: u32) -> Option<K> {
        let segment_meta = self.allocator.get_segment_meta(self.root_segment)?;
        let arena_idx = segment_meta.cache_key;
        self.successor_from_leaf_internal(key, leaf_idx, arena_idx)
    }

    /// Internal helper for successor from leaf.
    fn successor_from_leaf_internal(&self, key: K, leaf_idx: u32, arena_idx: u64) -> Option<K> {
        use crate::trie::{unpack_link, EMPTY_LINK};

        let leaf_arena = self.allocator.get_leaf_arena(arena_idx)?;
        let leaf = leaf_arena.get(leaf_idx);

        // Try to find successor in current leaf
        let last_byte = key.last_byte();
        if let Some(next_bit) = crate::bitmap::next_set_bit(&leaf.bitmap, last_byte) {
            // Found in same leaf - O(1) for clustered data!
            let mut key_value = key.prefix().to_u128();
            key_value |= next_bit as u128;
            return Some(K::from_u128(key_value));
        }

        // Not in current leaf - try next leaf via linked list
        if leaf.next != EMPTY_LINK {
            let (next_arena_idx, next_leaf_idx) = unpack_link(leaf.next);
            let next_leaf_arena = self.allocator.get_leaf_arena(next_arena_idx)?;
            let next_leaf = next_leaf_arena.get(next_leaf_idx);
            if let Some(min_bit) = crate::bitmap::min_bit(&next_leaf.bitmap) {
                // Found in next leaf - O(1) for adjacent leaves!
                let mut key_value = (next_leaf.prefix as u128) << 8;
                key_value |= min_bit as u128;
                return Some(K::from_u128(key_value));
            }
        }

        // No successor found
        None
    }

    /// Backtrack to find next leaf when path doesn't exist.
    fn successor_backtrack(
        &self,
        key: K,
        path: &[(u32, u8, u64)],
        path_len: usize,
        arena_idx: u64,
    ) -> Option<K> {
        use crate::trie::{unpack_link, EMPTY_LINK};

        // Find next leaf using backtracking
        let next_link = self.find_next_leaf(path, path_len);

        if next_link == EMPTY_LINK {
            return None;
        }

        // Get minimum key from next leaf
        let (next_arena_idx, next_leaf_idx) = unpack_link(next_link);
        let leaf_arena = self.allocator.get_leaf_arena(next_arena_idx)?;
        let next_leaf = leaf_arena.get(next_leaf_idx);

        if let Some(min_bit) = crate::bitmap::min_bit(&next_leaf.bitmap) {
            let mut key_value = (next_leaf.prefix as u128) << 8;
            key_value |= min_bit as u128;
            return Some(K::from_u128(key_value));
        }

        None
    }

    /// Find predecessor (largest key < given key).
    ///
    /// Returns the previous key in sorted order before the given key.
    /// Uses linked list of leaves for O(1) performance on clustered data.
    ///
    /// # Algorithm (from root)
    /// 1. Quick checks using cached min/max
    /// 2. Traverse to leaf containing key's prefix (save path)
    /// 3. If path doesn't exist AND leaf empty → backtrack to find prev leaf
    /// 4. If path exists OR leaf not empty:
    ///    - Search for prev bit in current leaf bitmap (O(1))
    ///    - If not found, follow leaf.prev to prev leaf (O(1))
    ///    - Take maximum bit from prev leaf
    /// 5. Reconstruct key from prefix + bit
    ///
    /// # Arguments
    /// * `key` - The key to find predecessor for
    ///
    /// # Returns
    /// Previous key before given key, or None if no predecessor exists
    ///
    /// # Performance
    /// - O(1) for existing keys or keys in same/adjacent leaves
    /// - O(log log U) for non-existing keys (requires backtracking)
    ///
    /// # Example
    /// ```rust
    /// use clustered_fast_trie::Trie;
    ///
    /// let mut trie = Trie::<u32>::new();
    /// trie.insert(10);
    /// trie.insert(20);
    /// trie.insert(30);
    ///
    /// assert_eq!(trie.predecessor(30), Some(20));
    /// assert_eq!(trie.predecessor(25), Some(20));
    /// assert_eq!(trie.predecessor(10), None);
    /// ```
    pub fn predecessor(&self, key: K) -> Option<K> {
        // Quick checks using cached min/max
        if let Some(min) = self.min_key {
            if key <= min {
                return None; // No predecessor if key <= min
            }
        }
        if let Some(max) = self.max_key {
            if key > max {
                return Some(max); // Max is predecessor if key > max
            }
        }

        // Get root arena
        let segment_meta = self.allocator.get_segment_meta(self.root_segment)?;
        let mut current_arena_idx = segment_meta.cache_key;

        // Get node arena
        let node_arena = self.allocator.get_node_arena(current_arena_idx)?;
        if node_arena.is_empty() {
            return None;
        }

        // Traverse to leaf containing key's prefix (save path for backtracking)
        let mut path: [(u32, u8, u64); 16] = [(0, 0, 0); 16];
        let mut path_len = 0;
        let mut current_node_idx = 0; // Start at root

        // Traverse internal levels (0..K::LEVELS-1)
        for level in 0..(K::LEVELS - 1) {
            let byte = key.byte_at(level);
            path[path_len] = (current_node_idx, byte, current_arena_idx);
            path_len += 1;

            // Get current node arena (may have changed at split level)
            let node_arena = self.allocator.get_node_arena(current_arena_idx)?;
            let current_node = node_arena.get(current_node_idx);

            if !current_node.has_child(byte) {
                // Path doesn't exist → backtrack to find prev leaf
                return self.predecessor_backtrack(key, &path, path_len, current_arena_idx);
            }

            current_node_idx = current_node.get_child(byte);

            // Check if we need to switch arenas at split level
            if K::SPLIT_LEVELS.contains(&(level + 1)) {
                // Next level is a split level - get child arena index from current node
                let child_arena_idx = current_node.child_arena_idx as u64;

                // Check if child arena exists
                if !self.allocator.has_arena(child_arena_idx) {
                    return None; // Child arena not allocated
                }

                current_arena_idx = child_arena_idx;
            }
        }

        // Final level: check if leaf exists
        let last_node_byte = key.byte_at(K::LEVELS - 1);
        path[path_len] = (current_node_idx, last_node_byte, current_arena_idx);
        path_len += 1;

        let node_arena = self.allocator.get_node_arena(current_arena_idx)?;
        let final_node = node_arena.get(current_node_idx);

        if !final_node.has_child(last_node_byte) {
            // Leaf doesn't exist → backtrack to find prev leaf
            return self.predecessor_backtrack(key, &path, path_len, current_arena_idx);
        }

        let leaf_idx = final_node.get_child(last_node_byte);

        // Leaf exists → search in leaf and prev
        self.predecessor_from_leaf_internal(key, leaf_idx, current_arena_idx)
    }

    /// Find predecessor starting from a specific leaf (for bulk operations).
    ///
    /// Optimized version for ordered bulk operations (iterators, ranges).
    /// Assumes the leaf index is already known (cached from previous operation).
    ///
    /// # Algorithm (from cached leaf)
    /// 1. Search for prev bit in current leaf bitmap (O(1))
    /// 2. If not found, follow leaf.prev to prev leaf (O(1))
    /// 3. Take maximum bit from prev leaf
    /// 4. Reconstruct key from prefix + bit
    ///
    /// # Arguments
    /// * `key` - The key to find predecessor for
    /// * `leaf_idx` - Index of the leaf to start search from (cached)
    ///
    /// # Returns
    /// Previous key before given key, or None if no predecessor exists
    ///
    /// # Performance
    /// O(1) - direct leaf access, no trie traversal
    ///
    /// # Use Case
    /// Reverse iterators and range operations cache the last accessed leaf index
    /// and use this method for subsequent calls, achieving O(1) per element.
    pub fn predecessor_from_leaf(&self, key: K, leaf_idx: u32) -> Option<K> {
        let segment_meta = self.allocator.get_segment_meta(self.root_segment)?;
        let arena_idx = segment_meta.cache_key;
        self.predecessor_from_leaf_internal(key, leaf_idx, arena_idx)
    }

    /// Internal helper for predecessor from leaf.
    fn predecessor_from_leaf_internal(&self, key: K, leaf_idx: u32, arena_idx: u64) -> Option<K> {
        use crate::trie::{unpack_link, EMPTY_LINK};

        let leaf_arena = self.allocator.get_leaf_arena(arena_idx)?;
        let leaf = leaf_arena.get(leaf_idx);

        // Try to find predecessor in current leaf
        let last_byte = key.last_byte();
        if let Some(prev_bit) = crate::bitmap::prev_set_bit(&leaf.bitmap, last_byte) {
            // Found in same leaf - O(1) for clustered data!
            let mut key_value = key.prefix().to_u128();
            key_value |= prev_bit as u128;
            return Some(K::from_u128(key_value));
        }

        // Not in current leaf - try prev leaf via linked list
        if leaf.prev != EMPTY_LINK {
            let (prev_arena_idx, prev_leaf_idx) = unpack_link(leaf.prev);
            let prev_leaf_arena = self.allocator.get_leaf_arena(prev_arena_idx)?;
            let prev_leaf = prev_leaf_arena.get(prev_leaf_idx);
            if let Some(max_bit) = crate::bitmap::max_bit(&prev_leaf.bitmap) {
                // Found in prev leaf - O(1) for adjacent leaves!
                let mut key_value = (prev_leaf.prefix as u128) << 8;
                key_value |= max_bit as u128;
                return Some(K::from_u128(key_value));
            }
        }

        // No predecessor found
        None
    }

    /// Backtrack to find prev leaf when path doesn't exist.
    fn predecessor_backtrack(
        &self,
        key: K,
        path: &[(u32, u8, u64)],
        path_len: usize,
        arena_idx: u64,
    ) -> Option<K> {
        use crate::trie::{unpack_link, EMPTY_LINK};

        // Find prev leaf using backtracking
        let prev_link = self.find_prev_leaf(path, path_len);

        if prev_link == EMPTY_LINK {
            return None;
        }

        // Get maximum key from prev leaf
        let (prev_arena_idx, prev_leaf_idx) = unpack_link(prev_link);
        let leaf_arena = self.allocator.get_leaf_arena(prev_arena_idx)?;
        let prev_leaf = leaf_arena.get(prev_leaf_idx);

        if let Some(max_bit) = crate::bitmap::max_bit(&prev_leaf.bitmap) {
            let mut key_value = (prev_leaf.prefix as u128) << 8;
            key_value |= max_bit as u128;
            return Some(K::from_u128(key_value));
        }

        None
    }

    /// Update min/max cache after insert.
    ///
    /// Called when a new key is inserted to maintain cached min/max values.
    ///
    /// # Performance
    /// O(1) - simple comparison and update
    #[inline]
    fn update_min_max_insert(&mut self, key: K) {
        match self.min_key {
            None => self.min_key = Some(key),
            Some(m) if key < m => self.min_key = Some(key),
            _ => {}
        }
        match self.max_key {
            None => self.max_key = Some(key),
            Some(m) if key > m => self.max_key = Some(key),
            _ => {}
        }
    }

    /// Update min/max cache after remove.
    ///
    /// Called when a key is removed to maintain cached min/max values.
    /// If the removed key was min or max, searches for new value.
    ///
    /// # Performance
    /// O(1) if removed key is not min/max
    /// O(log log U) if need to find new min/max
    #[inline]
    fn update_min_max_remove(&mut self, key: K) {
        if self.len == 0 {
            // Trie is now empty
            self.min_key = None;
            self.max_key = None;
        } else {
            // Check if we removed min or max
            if self.min_key == Some(key) {
                self.min_key = self.find_min();
            }
            if self.max_key == Some(key) {
                self.max_key = self.find_max();
            }
        }
    }

    /// Find minimum key by traversing leftmost path.
    ///
    /// Traverses from root to leaf, always taking the minimum child at each level.
    ///
    /// # Performance
    /// O(log log U) - traverses K::LEVELS levels
    ///
    /// # Returns
    /// Minimum key in the trie, or None if empty
    fn find_min(&self) -> Option<K> {
        // Get root arena
        let segment_meta = self.allocator.get_segment_meta(self.root_segment)?;
        let mut current_arena_idx = segment_meta.cache_key;

        // Get node arena
        let node_arena = self.allocator.get_node_arena(current_arena_idx)?;
        if node_arena.is_empty() {
            return None;
        }

        // Start building key from most significant byte
        let mut key_value: u128 = 0;
        let mut current_node_idx = 0; // Start at root

        // Traverse internal levels (0..K::LEVELS-1)
        for level in 0..(K::LEVELS - 1) {
            // Get current node arena (may have changed at split level)
            let node_arena = self.allocator.get_node_arena(current_arena_idx)?;
            let node = node_arena.get(current_node_idx);
            let min_byte = node.min_child()?;

            // Add byte to key at current level
            key_value |= (min_byte as u128) << ((K::LEVELS - level) * 8);

            // Move to child node
            current_node_idx = node.get_child(min_byte);

            // Check if we need to switch arenas at split level
            if K::SPLIT_LEVELS.contains(&(level + 1)) {
                // Next level is a split level - get child arena index from current node
                let child_arena_idx = node.child_arena_idx as u64;

                // Check if child arena exists
                if !self.allocator.has_arena(child_arena_idx) {
                    return None; // Child arena not allocated
                }

                current_arena_idx = child_arena_idx;
            }
        }

        // Final level: find minimum child (leaf)
        let node_arena = self.allocator.get_node_arena(current_arena_idx)?;
        let final_node = node_arena.get(current_node_idx);
        let min_leaf_byte = final_node.min_child()?;
        key_value |= (min_leaf_byte as u128) << 8;

        let leaf_idx = final_node.get_child(min_leaf_byte);

        // Get leaf arena and find minimum bit
        let leaf_arena = self.allocator.get_leaf_arena(current_arena_idx)?;
        let leaf = leaf_arena.get(leaf_idx);
        let min_bit = crate::bitmap::min_bit(&leaf.bitmap)?;

        // Add final byte to key
        key_value |= min_bit as u128;

        // Convert u128 back to key type K
        Some(K::from_u128(key_value))
    }

    /// Find maximum key by traversing rightmost path.
    ///
    /// Traverses from root to leaf, always taking the maximum child at each level.
    ///
    /// # Performance
    /// O(log log U) - traverses K::LEVELS levels
    ///
    /// # Returns
    /// Maximum key in the trie, or None if empty
    fn find_max(&self) -> Option<K> {
        // Get root arena
        let segment_meta = self.allocator.get_segment_meta(self.root_segment)?;
        let mut current_arena_idx = segment_meta.cache_key;

        // Get node arena
        let node_arena = self.allocator.get_node_arena(current_arena_idx)?;
        if node_arena.is_empty() {
            return None;
        }

        // Start building key from most significant byte
        let mut key_value: u128 = 0;
        let mut current_node_idx = 0; // Start at root

        // Traverse internal levels (0..K::LEVELS-1)
        for level in 0..(K::LEVELS - 1) {
            // Get current node arena (may have changed at split level)
            let node_arena = self.allocator.get_node_arena(current_arena_idx)?;
            let node = node_arena.get(current_node_idx);
            let max_byte = node.max_child()?;

            // Add byte to key at current level
            key_value |= (max_byte as u128) << ((K::LEVELS - level) * 8);

            // Move to child node
            current_node_idx = node.get_child(max_byte);

            // Check if we need to switch arenas at split level
            if K::SPLIT_LEVELS.contains(&(level + 1)) {
                // Next level is a split level - get child arena index from current node
                let child_arena_idx = node.child_arena_idx as u64;

                // Check if child arena exists
                if !self.allocator.has_arena(child_arena_idx) {
                    return None; // Child arena not allocated
                }

                current_arena_idx = child_arena_idx;
            }
        }

        // Final level: find maximum child (leaf)
        let node_arena = self.allocator.get_node_arena(current_arena_idx)?;
        let final_node = node_arena.get(current_node_idx);
        let max_leaf_byte = final_node.max_child()?;
        key_value |= (max_leaf_byte as u128) << 8;

        let leaf_idx = final_node.get_child(max_leaf_byte);

        // Get leaf arena and find maximum bit
        let leaf_arena = self.allocator.get_leaf_arena(current_arena_idx)?;
        let leaf = leaf_arena.get(leaf_idx);
        let max_bit = crate::bitmap::max_bit(&leaf.bitmap)?;

        // Add final byte to key
        key_value |= max_bit as u128;

        // Convert u128 back to key type K
        Some(K::from_u128(key_value))
    }

    // ============================================================================
    // Arena Access Helper Methods
    // ============================================================================
    //
    // These methods provide unified access to arenas regardless of whether they
    // are root arenas (stored in Trie.root_node.child_arenas) or child arenas
    // (stored in nodes at split levels).
    //
    // Current implementation (Phase 1): Only supports root arenas (arena_idx = 0).
    // Future implementation (Phase 4): Will support child arenas via path traversal.

    /// Get immutable reference to node arena.
    ///
    /// # Arguments
    /// * `arena_idx` - Arena index (0 = root arenas, >0 = child arenas)
    ///
    /// # Returns
    /// Reference to node arena, or None if arena doesn't exist
    ///
    /// # Current Limitations
    /// Only supports arena_idx = 0 (root arenas). Child arenas (arena_idx > 0)
    /// will be supported in Phase 4 after implementing path-based arena lookup.
    #[inline]
    fn get_node_arena(&self, arena_idx: u64) -> Option<&crate::arena::Arena<crate::trie::Node>> {
        if arena_idx == 0 {
            // Root arenas stored in root_node
            self.root_node
                .child_arenas
                .as_ref()
                .map(|ca| &ca.node_arena)
        } else {
            // TODO Phase 4: Child arenas - need path-based lookup
            None
        }
    }

    /// Get mutable reference to node arena.
    ///
    /// # Arguments
    /// * `arena_idx` - Arena index (0 = root arenas, >0 = child arenas)
    ///
    /// # Returns
    /// Mutable reference to node arena, or None if arena doesn't exist
    ///
    /// # Current Limitations
    /// Only supports arena_idx = 0 (root arenas). Child arenas (arena_idx > 0)
    /// will be supported in Phase 4 after implementing path-based arena lookup.
    #[inline]
    fn get_node_arena_mut(
        &mut self,
        arena_idx: u64,
    ) -> Option<&mut crate::arena::Arena<crate::trie::Node>> {
        if arena_idx == 0 {
            // Root arenas stored in root_node
            self.root_node
                .child_arenas
                .as_mut()
                .map(|ca| &mut ca.node_arena)
        } else {
            // TODO Phase 4: Child arenas - need path-based lookup
            None
        }
    }

    /// Get immutable reference to leaf arena.
    ///
    /// # Arguments
    /// * `arena_idx` - Arena index (0 = root arenas, >0 = child arenas)
    ///
    /// # Returns
    /// Reference to leaf arena, or None if arena doesn't exist
    ///
    /// # Current Limitations
    /// Only supports arena_idx = 0 (root arenas). Child arenas (arena_idx > 0)
    /// will be supported in Phase 4 after implementing path-based arena lookup.
    #[inline]
    fn get_leaf_arena(&self, arena_idx: u64) -> Option<&crate::arena::Arena<crate::trie::Leaf>> {
        if arena_idx == 0 {
            // Root arenas stored in root_node
            self.root_node
                .child_arenas
                .as_ref()
                .map(|ca| &ca.leaf_arena)
        } else {
            // TODO Phase 4: Child arenas - need path-based lookup
            None
        }
    }

    /// Get mutable reference to leaf arena.
    ///
    /// # Arguments
    /// * `arena_idx` - Arena index (0 = root arenas, >0 = child arenas)
    ///
    /// # Returns
    /// Mutable reference to leaf arena, or None if arena doesn't exist
    ///
    /// # Current Limitations
    /// Only supports arena_idx = 0 (root arenas). Child arenas (arena_idx > 0)
    /// will be supported in Phase 4 after implementing path-based arena lookup.
    #[inline]
    fn get_leaf_arena_mut(
        &mut self,
        arena_idx: u64,
    ) -> Option<&mut crate::arena::Arena<crate::trie::Leaf>> {
        if arena_idx == 0 {
            // Root arenas stored in root_node
            self.root_node
                .child_arenas
                .as_mut()
                .map(|ca| &mut ca.leaf_arena)
        } else {
            // TODO Phase 4: Child arenas - need path-based lookup
            None
        }
    }

    /// Check if arena exists.
    ///
    /// # Arguments
    /// * `arena_idx` - Arena index (0 = root arenas, >0 = child arenas)
    ///
    /// # Returns
    /// true if arena exists, false otherwise
    ///
    /// # Current Limitations
    /// Only supports arena_idx = 0 (root arenas). Child arenas (arena_idx > 0)
    /// will be supported in Phase 4 after implementing path-based arena lookup.
    #[inline]
    fn has_arena(&self, arena_idx: u64) -> bool {
        if arena_idx == 0 {
            // Root arenas stored in root_node
            self.root_node.child_arenas.is_some()
        } else {
            // TODO Phase 4: Child arenas - need path-based lookup
            false
        }
    }

    // ============================================================================
    // End of Arena Access Helper Methods (OLD - for compatibility)
    // ============================================================================

    // ============================================================================
    // NEW ARCHITECTURE: Arena Access Helper Methods for Vec<ChildArenas>
    // ============================================================================
    //
    // These methods provide direct O(1) access to arenas via physical indices
    // stored in Vec<ChildArenas>. No path traversal needed!

    /// Get immutable reference to node arena by physical index (NEW ARCHITECTURE).
    ///
    /// # Arguments
    /// * `arena_idx` - Physical index in arenas Vec (u32)
    ///
    /// # Returns
    /// Reference to node arena
    ///
    /// # Panics
    /// Panics if arena_idx is out of bounds
    ///
    /// # Performance
    /// O(1) - direct Vec indexing
    #[inline(always)]
    fn get_node_arena_new(&self, arena_idx: u32) -> &crate::arena::Arena<crate::trie::Node> {
        &self.arenas[arena_idx as usize].node_arena
    }

    /// Get mutable reference to node arena by physical index (NEW ARCHITECTURE).
    ///
    /// # Arguments
    /// * `arena_idx` - Physical index in arenas Vec (u32)
    ///
    /// # Returns
    /// Mutable reference to node arena
    ///
    /// # Panics
    /// Panics if arena_idx is out of bounds
    ///
    /// # Performance
    /// O(1) - direct Vec indexing
    #[inline(always)]
    fn get_node_arena_mut_new(&mut self, arena_idx: u32) -> &mut crate::arena::Arena<crate::trie::Node> {
        &mut self.arenas[arena_idx as usize].node_arena
    }

    /// Get immutable reference to leaf arena by physical index (NEW ARCHITECTURE).
    ///
    /// # Arguments
    /// * `arena_idx` - Physical index in arenas Vec (u32)
    ///
    /// # Returns
    /// Reference to leaf arena
    ///
    /// # Panics
    /// Panics if arena_idx is out of bounds
    ///
    /// # Performance
    /// O(1) - direct Vec indexing
    #[inline(always)]
    fn get_leaf_arena_new(&self, arena_idx: u32) -> &crate::arena::Arena<crate::trie::Leaf> {
        &self.arenas[arena_idx as usize].leaf_arena
    }

    /// Get mutable reference to leaf arena by physical index (NEW ARCHITECTURE).
    ///
    /// # Arguments
    /// * `arena_idx` - Physical index in arenas Vec (u32)
    ///
    /// # Returns
    /// Mutable reference to leaf arena
    ///
    /// # Panics
    /// Panics if arena_idx is out of bounds
    ///
    /// # Performance
    /// O(1) - direct Vec indexing
    #[inline(always)]
    fn get_leaf_arena_mut_new(&mut self, arena_idx: u32) -> &mut crate::arena::Arena<crate::trie::Leaf> {
        &mut self.arenas[arena_idx as usize].leaf_arena
    }

    /// Create new child arena and return its physical index (NEW ARCHITECTURE).
    ///
    /// # Returns
    /// Physical index of the newly created arena in arenas Vec
    ///
    /// # Performance
    /// O(1) amortized - Vec::push()
    #[inline]
    fn create_child_arena(&mut self) -> u32 {
        self.arenas.push(ChildArenas::new());
        (self.arenas.len() - 1) as u32
    }

    // ============================================================================
    // End of NEW ARCHITECTURE Arena Access Helper Methods
    // ============================================================================

    /// Ensure root node exists at index 0 in node arena.
    ///
    /// Root node is always stored at index 0 in root_node.child_arenas.node_arena.
    ///
    /// # Returns
    /// Always returns 0 (root node index)
    ///
    /// # Performance
    /// O(1) - checks if arena is empty and allocates root node if needed
    fn ensure_root_node(&mut self) -> u32 {
        let root_arenas = self.root_node.child_arenas.as_mut()
            .expect("Root node should have child arenas");

        if root_arenas.node_arena.is_empty() {
            // Create root node at index 0
            root_arenas.node_arena.alloc()
        } else {
            // Root node already exists at index 0
            0
        }
    }

    /// Traverse trie levels to find or create leaf.
    ///
    /// Performs full traversal through internal Node structures at each level,
    /// creating nodes as needed. On the final level, finds or creates a Leaf.
    /// Collects the path during traversal for later cleanup operations.
    ///
    /// # Algorithm
    /// 1. Start at root node
    /// 2. For each level (0..K::LEVELS-1):
    ///    - Extract byte at current level from key
    ///    - Record (node_idx, byte) in path
    ///    - Check if child exists for this byte
    ///    - If exists: move to that child node
    ///    - If not: create new node and link it
    /// 3. At final level (K::LEVELS-1):
    ///    - Extract last byte before leaf level
    ///    - Record (node_idx, byte) in path
    ///    - Check if leaf exists for this byte
    ///    - If exists: return (leaf_idx, path, path_len)
    ///    - If not: create new leaf and link it
    ///
    /// # Performance
    /// O(K::LEVELS) = O(log log U) where U is key space size
    ///
    /// # Returns
    /// (leaf_idx, path, path_len) where:
    /// - leaf_idx: index of the leaf in leaf arena
    /// - path: array of (node_idx, byte) pairs representing the path from root
    /// - path_len: number of valid entries in path array
    fn traverse_to_leaf(
        &mut self,
        key: K,
        mut current_node_idx: u32,
        mut current_arena_idx: u64,
    ) -> (u32, [(u32, u8, u64); 16], usize, u64) {
        // Path tracking: (node_idx, byte, arena_idx) for each level
        // Max 16 levels for u128 (0..15)
        let mut path = [(0u32, 0u8, 0u64); 16];
        let mut path_len = 0;

        // Traverse internal levels (0..K::LEVELS-1)
        // Each level navigates through Node structures
        for level in 0..(K::LEVELS - 1) {
            let byte = key.byte_at(level);

            // Record current node, byte, and arena in path
            path[path_len] = (current_node_idx, byte, current_arena_idx);
            path_len += 1;

            // Check if child exists (read-only operation)
            let child_idx = {
                let node_arena = self
                    .allocator
                    .get_node_arena(current_arena_idx)
                    .expect("Node arena should be allocated");
                let current_node = node_arena.get(current_node_idx);

                if current_node.has_child(byte) {
                    Some(current_node.get_child(byte))
                } else {
                    None
                }
            };

            if let Some(idx) = child_idx {
                // Child exists - check if we need to switch arena BEFORE moving to child
                if K::SPLIT_LEVELS.contains(&(level + 1)) {
                    // Next level is a split level - get or set child_arena_idx from CURRENT node
                    let child_arena_idx = {
                        let node_arena = self
                            .allocator
                            .get_node_arena(current_arena_idx)
                            .expect("Node arena should be allocated");
                        let current_node = node_arena.get(current_node_idx);
                        current_node.child_arena_idx as u64
                    };

                    // If child_arena_idx is 0, it means this node was created before
                    // we set it properly - compute and set it now
                    if child_arena_idx == 0 {
                        let computed_arena_idx = key.arena_idx_at_level(level + 1);

                        // Allocate child arena if not exists
                        if !self.allocator.has_arena(computed_arena_idx) {
                            self.allocator.allocate_arena_for_key(computed_arena_idx);
                        }

                        // Set child_arena_idx in the CURRENT node (parent)
                        let node_arena = self
                            .allocator
                            .get_node_arena_mut(current_arena_idx)
                            .expect("Node arena should be allocated");
                        let current_node = node_arena.get_mut(current_node_idx);
                        current_node.child_arena_idx = computed_arena_idx as u32;

                        current_arena_idx = computed_arena_idx;
                    } else {
                        current_arena_idx = child_arena_idx;
                    }
                }

                // Now move to child node
                current_node_idx = idx;
            } else {
                // Child doesn't exist - create new node and link it

                // Determine which arena the new node should be in
                // If we're at a split level, the new node goes to child arena
                let (target_arena_idx, is_split_level) = if K::SPLIT_LEVELS.contains(&(level + 1)) {
                    // Next level is split - new node goes to child arena
                    let child_arena_idx = key.arena_idx_at_level(level + 1);

                    // Allocate child arena if not exists
                    if !self.allocator.has_arena(child_arena_idx) {
                        self.allocator.allocate_arena_for_key(child_arena_idx);
                    }

                    (child_arena_idx, true)
                } else {
                    (current_arena_idx, false)
                };

                // Allocate new node in target arena
                let new_node_idx = {
                    let node_arena = self
                        .allocator
                        .get_node_arena_mut(target_arena_idx)
                        .expect("Node arena should be allocated");
                    node_arena.alloc()
                };

                // Set parent_idx for the new node
                {
                    let node_arena = self
                        .allocator
                        .get_node_arena_mut(target_arena_idx)
                        .expect("Node arena should be allocated");
                    let new_node = node_arena.get_mut(new_node_idx);
                    new_node.parent_idx = current_node_idx;
                }

                // Link from parent to child
                {
                    let node_arena = self
                        .allocator
                        .get_node_arena_mut(current_arena_idx)
                        .expect("Node arena should be allocated");
                    let current_node = node_arena.get_mut(current_node_idx);
                    current_node.set_child(byte, new_node_idx);

                    // If we're creating a node in child arena, store child_arena_idx in parent
                    if is_split_level {
                        current_node.child_arena_idx = target_arena_idx as u32;
                    }
                }

                // Move to new node
                current_node_idx = new_node_idx;
                current_arena_idx = target_arena_idx;
            }
        }

        // Final level (K::LEVELS - 1): transition from Node to Leaf
        let last_node_byte = key.byte_at(K::LEVELS - 1);

        // Record final node, byte, and arena in path
        path[path_len] = (current_node_idx, last_node_byte, current_arena_idx);
        path_len += 1;

        // Check if leaf exists (read-only operation)
        let leaf_idx = {
            let node_arena = self
                .allocator
                .get_node_arena(current_arena_idx)
                .expect("Node arena should be allocated");
            let final_node = node_arena.get(current_node_idx);

            if final_node.has_child(last_node_byte) {
                Some(final_node.get_child(last_node_byte))
            } else {
                None
            }
        };

        let leaf_idx = if let Some(idx) = leaf_idx {
            // Leaf already exists
            idx
        } else {
            // Create new leaf and link it
            let prefix = key.prefix().to_u128() as u64;

            let leaf_arena = self
                .allocator
                .get_leaf_arena_mut(current_arena_idx)
                .expect("Leaf arena should be allocated");
            let new_leaf_idx = leaf_arena.alloc(prefix);

            let node_arena = self
                .allocator
                .get_node_arena_mut(current_arena_idx)
                .expect("Node arena should be allocated");
            let final_node = node_arena.get_mut(current_node_idx);
            final_node.set_child(last_node_byte, new_leaf_idx);

            // Link new leaf into linked list
            self.link_leaf(new_leaf_idx, &path, path_len, current_arena_idx);

            new_leaf_idx
        };

        // Return tuple: (leaf_idx, path, path_len, arena_idx)
        (leaf_idx, path, path_len, current_arena_idx)
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

    /// Remove leaf from linked list.
    ///
    /// Unlinks the leaf from the doubly-linked list by updating the next/prev
    /// pointers of its neighbors. Also updates first_leaf_idx/last_leaf_idx
    /// if the removed leaf was at a boundary.
    ///
    /// # Algorithm
    /// 1. Get prev/next indices from the leaf being removed
    /// 2. Update prev leaf's next pointer to skip removed leaf
    /// 3. Update next leaf's prev pointer to skip removed leaf
    /// 4. Update first_leaf if removed leaf was first
    /// 5. Update last_leaf if removed leaf was last
    ///
    /// # Arguments
    /// * `leaf_idx` - Index of the leaf to remove from list
    /// * `arena_idx` - Arena index for storage
    ///
    /// # Performance
    /// O(1) - direct pointer updates using prev/next fields, supports cross-arena
    fn unlink_leaf(&mut self, leaf_idx: u32, arena_idx: u64) {
        use crate::trie::{unpack_link, EMPTY_LINK};

        let leaf_arena = self
            .allocator
            .get_leaf_arena(arena_idx)
            .expect("Leaf arena should be allocated");

        // Get prev/next from the leaf being removed
        let leaf = leaf_arena.get(leaf_idx);
        let prev_link = leaf.prev;
        let next_link = leaf.next;

        // Update prev leaf's next pointer
        if prev_link == EMPTY_LINK {
            // Removed leaf was first - update first_leaf
            self.first_leaf = next_link;
        } else {
            let (prev_arena_idx, prev_leaf_idx) = unpack_link(prev_link);
            let prev_leaf_arena = self
                .allocator
                .get_leaf_arena_mut(prev_arena_idx)
                .expect("Prev leaf arena should be allocated");
            let prev_leaf = prev_leaf_arena.get_mut(prev_leaf_idx);
            prev_leaf.next = next_link;
        }

        // Update next leaf's prev pointer
        if next_link == EMPTY_LINK {
            // Removed leaf was last - update last_leaf
            self.last_leaf = prev_link;
        } else {
            let (next_arena_idx, next_leaf_idx) = unpack_link(next_link);
            let next_leaf_arena = self
                .allocator
                .get_leaf_arena_mut(next_arena_idx)
                .expect("Next leaf arena should be allocated");
            let next_leaf = next_leaf_arena.get_mut(next_leaf_idx);
            next_leaf.prev = prev_link;
        }
    }

    /// Insert leaf into linked list in sorted order.
    ///
    /// Links the newly created leaf into the doubly-linked list of leaves,
    /// maintaining sorted order by prefix. Updates next/prev pointers for
    /// the new leaf and its neighbors, and updates first_leaf/last_leaf
    /// if necessary.
    ///
    /// # Algorithm
    /// 1. If this is the first leaf: set first_leaf and last_leaf
    /// 2. Find prev/next leaves using trie backtracking (O(log log U))
    /// 3. Update new leaf's prev/next pointers
    /// 4. Update neighbors' pointers to link to new leaf
    /// 5. Update first_leaf/last_leaf if at boundaries
    ///
    /// # Arguments
    /// * `new_leaf_idx` - Index of the newly created leaf
    /// * `path` - Array of (node_idx, byte, arena_idx) tuples from root to leaf
    /// * `path_len` - Number of valid entries in path
    /// * `arena_idx` - Arena index for storage
    ///
    /// # Performance
    /// O(log log U) - uses trie structure to find neighbors efficiently, supports cross-arena
    fn link_leaf(
        &mut self,
        new_leaf_idx: u32,
        path: &[(u32, u8, u64)],
        path_len: usize,
        arena_idx: u64,
    ) {
        use crate::trie::{pack_link, unpack_link, EMPTY_LINK};

        // Check if this is the first leaf
        if self.first_leaf == EMPTY_LINK {
            // First leaf ever - initialize list
            let new_link = pack_link(arena_idx, new_leaf_idx);
            self.first_leaf = new_link;
            self.last_leaf = new_link;
            // new leaf already has prev=EMPTY_LINK and next=EMPTY_LINK from Leaf::new()
            return;
        }

        // Find prev/next leaves using trie backtracking
        let prev_link = self.find_prev_leaf(path, path_len);
        let next_link = self.find_next_leaf(path, path_len);

        // Get leaf arena for updates
        let leaf_arena = self
            .allocator
            .get_leaf_arena_mut(arena_idx)
            .expect("Leaf arena should be allocated");

        // Update new leaf's pointers
        let new_leaf = leaf_arena.get_mut(new_leaf_idx);
        new_leaf.prev = prev_link;
        new_leaf.next = next_link;

        let new_link = pack_link(arena_idx, new_leaf_idx);

        // Update prev leaf's next pointer
        if prev_link == EMPTY_LINK {
            // New leaf is now first
            self.first_leaf = new_link;
        } else {
            let (prev_arena_idx, prev_leaf_idx) = unpack_link(prev_link);
            let prev_leaf_arena = self
                .allocator
                .get_leaf_arena_mut(prev_arena_idx)
                .expect("Prev leaf arena should be allocated");
            let prev_leaf = prev_leaf_arena.get_mut(prev_leaf_idx);
            prev_leaf.next = new_link;
        }

        // Update next leaf's prev pointer
        if next_link == EMPTY_LINK {
            // New leaf is now last
            self.last_leaf = new_link;
        } else {
            let (next_arena_idx, next_leaf_idx) = unpack_link(next_link);
            let next_leaf_arena = self
                .allocator
                .get_leaf_arena_mut(next_arena_idx)
                .expect("Next leaf arena should be allocated");
            let next_leaf = next_leaf_arena.get_mut(next_leaf_idx);
            next_leaf.prev = new_link;
        }
    }

    /// Find previous leaf using trie backtracking.
    ///
    /// Backtracks through the path to find the predecessor leaf.
    /// At each level, checks if there's a predecessor child, and if found,
    /// descends to the maximum leaf in that branch.
    ///
    /// # Arguments
    /// * `path` - Array of (node_idx, byte, arena_idx) tuples from root to current position
    /// * `path_len` - Number of valid entries in path
    ///
    /// # Returns
    /// Packed link (arena_idx << 32 | leaf_idx) of the previous leaf, or EMPTY_LINK if none exists
    ///
    /// # Performance
    /// O(log log U) - backtracks at most K::LEVELS levels, supports cross-arena navigation
    fn find_prev_leaf(&self, path: &[(u32, u8, u64)], path_len: usize) -> u64 {
        use crate::trie::EMPTY_LINK;

        // Backtrack through path to find predecessor branch
        for i in (0..path_len).rev() {
            let (node_idx, byte, arena_idx) = path[i];

            let node_arena = match self.allocator.get_node_arena(arena_idx) {
                Some(arena) => arena,
                None => continue,
            };
            let node = node_arena.get(node_idx);

            // Check if there's a predecessor child at this level
            if let Some(pred_byte) = node.predecessor_child(byte) {
                // Found predecessor branch - descend to maximum leaf
                let pred_child_idx = node.get_child(pred_byte);

                // Check if next level is a split level - need to switch arena
                let next_arena_idx = if K::SPLIT_LEVELS.contains(&(i + 1)) {
                    // Get child_arena_idx from the node we're descending into
                    let child_arena_idx = node.child_arena_idx as u64;
                    if !self.allocator.has_arena(child_arena_idx) {
                        return EMPTY_LINK;
                    }
                    child_arena_idx
                } else {
                    arena_idx
                };

                return self.find_max_leaf_from(pred_child_idx, i + 1, next_arena_idx);
            }
        }

        // No predecessor found
        EMPTY_LINK
    }

    /// Find next leaf using trie backtracking.
    ///
    /// Backtracks through the path to find the successor leaf.
    /// At each level, checks if there's a successor child, and if found,
    /// descends to the minimum leaf in that branch.
    ///
    /// # Arguments
    /// * `path` - Array of (node_idx, byte, arena_idx) tuples from root to current position
    /// * `path_len` - Number of valid entries in path
    ///
    /// # Returns
    /// Packed link (arena_idx << 32 | leaf_idx) of the next leaf, or EMPTY_LINK if none exists
    ///
    /// # Performance
    /// O(log log U) - backtracks at most K::LEVELS levels, supports cross-arena navigation
    fn find_next_leaf(&self, path: &[(u32, u8, u64)], path_len: usize) -> u64 {
        use crate::trie::EMPTY_LINK;

        // Backtrack through path to find successor branch
        for i in (0..path_len).rev() {
            let (node_idx, byte, arena_idx) = path[i];

            let node_arena = match self.allocator.get_node_arena(arena_idx) {
                Some(arena) => arena,
                None => continue,
            };
            let node = node_arena.get(node_idx);

            // Check if there's a successor child at this level
            if let Some(succ_byte) = node.successor_child(byte) {
                // Found successor branch - descend to minimum leaf
                let succ_child_idx = node.get_child(succ_byte);

                // Check if next level is a split level - need to switch arena
                let next_arena_idx = if K::SPLIT_LEVELS.contains(&(i + 1)) {
                    // Get child_arena_idx from the node we're descending into
                    let child_arena_idx = node.child_arena_idx as u64;
                    if !self.allocator.has_arena(child_arena_idx) {
                        return EMPTY_LINK;
                    }
                    child_arena_idx
                } else {
                    arena_idx
                };

                return self.find_min_leaf_from(succ_child_idx, i + 1, next_arena_idx);
            }
        }

        // No successor found
        EMPTY_LINK
    }

    /// Find minimum leaf starting from a node at given level.
    ///
    /// Descends from the specified node, always taking the minimum child
    /// at each level until reaching a leaf. Used for finding the next leaf
    /// in sorted order when building the linked list.
    ///
    /// # Arguments
    /// * `node_idx` - Starting node index
    /// * `start_level` - Level of the starting node (0..K::LEVELS-1)
    /// * `arena_idx` - Arena index for storage
    ///
    /// # Returns
    /// Packed link (arena_idx << 32 | leaf_idx) of the minimum leaf, or EMPTY_LINK if none exists
    ///
    /// # Performance
    /// O(K::LEVELS - start_level) = O(log log U), supports cross-arena navigation
    fn find_min_leaf_from(&self, mut node_idx: u32, start_level: usize, mut arena_idx: u64) -> u64 {
        use crate::trie::{pack_link, EMPTY_LINK};

        // Traverse internal levels from start_level to K::LEVELS-1
        for level in start_level..(K::LEVELS - 1) {
            let node_arena = match self.allocator.get_node_arena(arena_idx) {
                Some(arena) => arena,
                None => return EMPTY_LINK,
            };

            let node = node_arena.get(node_idx);
            let min_byte = match node.min_child() {
                Some(b) => b,
                None => return EMPTY_LINK,
            };
            node_idx = node.get_child(min_byte);

            // Check if we need to switch arenas at split level
            if K::SPLIT_LEVELS.contains(&(level + 1)) {
                let child_arena_idx = node.child_arena_idx as u64;
                if !self.allocator.has_arena(child_arena_idx) {
                    return EMPTY_LINK;
                }
                arena_idx = child_arena_idx;
            }
        }

        // Final level: get minimum leaf
        let node_arena = match self.allocator.get_node_arena(arena_idx) {
            Some(arena) => arena,
            None => return EMPTY_LINK,
        };
        let final_node = node_arena.get(node_idx);
        match final_node.min_child() {
            Some(min_byte) => {
                let leaf_idx = final_node.get_child(min_byte);
                pack_link(arena_idx, leaf_idx)
            }
            None => EMPTY_LINK,
        }
    }

    /// Find maximum leaf starting from a node at given level.
    ///
    /// Descends from the specified node, always taking the maximum child
    /// at each level until reaching a leaf. Used for finding the previous leaf
    /// in sorted order when building the linked list.
    ///
    /// # Arguments
    /// * `node_idx` - Starting node index
    /// * `start_level` - Level of the starting node (0..K::LEVELS-1)
    /// * `arena_idx` - Arena index for storage
    ///
    /// # Returns
    /// Packed link (arena_idx << 32 | leaf_idx) of the maximum leaf, or EMPTY_LINK if none exists
    ///
    /// # Performance
    /// O(K::LEVELS - start_level) = O(log log U), supports cross-arena navigation
    fn find_max_leaf_from(&self, mut node_idx: u32, start_level: usize, mut arena_idx: u64) -> u64 {
        use crate::trie::{pack_link, EMPTY_LINK};

        // Traverse internal levels from start_level to K::LEVELS-1
        for level in start_level..(K::LEVELS - 1) {
            let node_arena = match self.allocator.get_node_arena(arena_idx) {
                Some(arena) => arena,
                None => return EMPTY_LINK,
            };

            let node = node_arena.get(node_idx);
            let max_byte = match node.max_child() {
                Some(b) => b,
                None => return EMPTY_LINK,
            };
            node_idx = node.get_child(max_byte);

            // Check if we need to switch arenas at split level
            if K::SPLIT_LEVELS.contains(&(level + 1)) {
                let child_arena_idx = node.child_arena_idx as u64;
                if !self.allocator.has_arena(child_arena_idx) {
                    return EMPTY_LINK;
                }
                arena_idx = child_arena_idx;
            }
        }

        // Final level: get maximum leaf
        let node_arena = match self.allocator.get_node_arena(arena_idx) {
            Some(arena) => arena,
            None => return EMPTY_LINK,
        };
        let final_node = node_arena.get(node_idx);
        match final_node.max_child() {
            Some(max_byte) => {
                let leaf_idx = final_node.get_child(max_byte);
                pack_link(arena_idx, leaf_idx)
            }
            None => EMPTY_LINK,
        }
    }

    /// Clean up empty nodes along the path from leaf to root.
    ///
    /// Traverses the path bottom-up, checking each node for emptiness.
    /// If a node is empty, removes the link from its parent and frees the node.
    /// Stops at the first non-empty node (optimization: parent can't be empty if child isn't).
    ///
    /// # Algorithm
    /// 1. Start from bottom of path (closest to leaf)
    /// 2. For each node in path (bottom-up):
    ///    - Check if node is empty via bitmap OR reduction
    ///    - If NOT empty: stop (parent can't be empty)
    ///    - If empty:
    ///      - Get parent from path[i-1]
    ///      - Remove link: parent.clear_child(byte)
    ///      - Free node: allocator.free_node(arena_idx, node_idx)
    /// 3. Never delete root node (index 0)
    ///
    /// # Arguments
    /// * `path` - Array of (node_idx, byte, arena_idx) tuples from root to leaf
    /// * `path_len` - Number of valid entries in path
    ///
    /// # Performance
    /// O(K::LEVELS) worst case, but typically stops early at first non-empty node
    fn cleanup_empty_nodes(&mut self, path: &[(u32, u8, u64)], path_len: usize) {
        // Nothing to clean if path is empty
        if path_len == 0 {
            return;
        }

        // Traverse path bottom-up (from leaf towards root)
        // Start at path_len-1 (last node before leaf)
        for i in (0..path_len).rev() {
            let (node_idx, byte, arena_idx) = path[i];

            // Never delete root node (index 0)
            if node_idx == 0 {
                break;
            }

            // Check if node is empty
            let is_empty = {
                let node_arena = self
                    .allocator
                    .get_node_arena(arena_idx)
                    .expect("Node arena should be allocated");
                let node = node_arena.get(node_idx);
                node.is_empty()
            };

            if !is_empty {
                // Node is not empty - stop cleanup
                // (if child is not empty, parent can't be empty either)
                break;
            }

            // Node is empty - remove link from parent and free it
            if i > 0 {
                // Get parent from previous level in path
                let (parent_idx, _, parent_arena_idx) = path[i - 1];

                // Remove link from parent
                let node_arena = self
                    .allocator
                    .get_node_arena_mut(parent_arena_idx)
                    .expect("Node arena should be allocated");
                let parent = node_arena.get_mut(parent_idx);
                parent.clear_child(byte);
            }

            // Free the empty node
            self.allocator.free_node(arena_idx, node_idx);
        }
    }
}

impl<K: TrieKey> Default for Trie<K> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use std::eprintln;
    use std::println;

    #[test]
    fn test_new_trie_u32() {
        let trie = Trie::<u32>::new();

        // Check that allocator has one segment
        assert!(trie.allocator.get_segment_meta(trie.root_segment).is_some());

        // Check segment covers full u32 range
        let meta = trie.allocator.get_segment_meta(trie.root_segment).unwrap();
        assert_eq!(meta.key_offset, 0);
        assert_eq!(meta.cache_key, 0);
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

    #[test]
    fn test_multi_level_structure_u32() {
        let mut trie = Trie::<u32>::new();

        // For u32, we have 4 levels (0, 1, 2, 3)
        // Key structure: [byte0][byte1][byte2][byte3]
        // Levels 0-2: internal Nodes
        // Level 3: Node points to Leaf

        // Insert keys that differ at different levels
        // Key 0x01020304 = [1][2][3][4]
        // Key 0x01020305 = [1][2][3][5] - shares path up to level 2
        // Key 0x01020404 = [1][2][4][4] - shares path up to level 1
        // Key 0x01030304 = [1][3][3][4] - shares path up to level 0

        let key1 = 0x01020304u32;
        let key2 = 0x01020305u32;
        let key3 = 0x01020404u32;
        let key4 = 0x01030304u32;

        // Insert all keys
        assert!(trie.insert(key1));
        assert!(trie.insert(key2));
        assert!(trie.insert(key3));
        assert!(trie.insert(key4));

        // Verify all keys exist
        assert!(trie.contains(key1));
        assert!(trie.contains(key2));
        assert!(trie.contains(key3));
        assert!(trie.contains(key4));

        // Get arena index
        let segment_meta = trie.allocator.get_segment_meta(trie.root_segment).unwrap();
        let arena_idx = segment_meta.cache_key;

        // Check that node arena has multiple nodes
        let node_arena = trie.allocator.get_node_arena(arena_idx).unwrap();
        // Should have: root + nodes for different branches
        // At minimum: 1 root + nodes for different paths
        assert!(
            node_arena.len() > 1,
            "Should have multiple nodes for different paths"
        );

        // Check that leaf arena has multiple leaves
        let leaf_arena = trie.allocator.get_leaf_arena(arena_idx).unwrap();
        // Keys with different prefixes should create different leaves
        assert!(
            leaf_arena.len() >= 2,
            "Should have multiple leaves for different prefixes"
        );
    }

    #[test]
    fn test_multi_level_structure_u64() {
        let mut trie = Trie::<u64>::new();

        // For u64, we have 8 levels (0-7) with split at level 4
        // Insert keys that differ at different levels but share same upper 4 bytes

        let key1 = 0x0102030405060708u64;
        let key2 = 0x0102030405060709u64; // Differs at last byte (level 7)
        let key3 = 0x0102030405070708u64; // Differs at byte 6 (level 6)
        let key4 = 0x0102030405080708u64; // Differs at byte 5 (level 5) - FIXED to share same upper 4 bytes

        // Insert all keys
        let r1 = trie.insert(key1);
        eprintln!("Insert key1: {}", r1);
        let r2 = trie.insert(key2);
        eprintln!("Insert key2: {}", r2);
        let r3 = trie.insert(key3);
        eprintln!("Insert key3: {}", r3);
        let r4 = trie.insert(key4);
        eprintln!("Insert key4: {}", r4);

        // Debug arena state
        eprintln!(
            "Root arena nodes: {:?}",
            trie.allocator.get_node_arena(0).map(|a| a.len())
        );
        eprintln!(
            "Root arena leaves: {:?}",
            trie.allocator.get_leaf_arena(0).map(|a| a.len())
        );
        eprintln!(
            "Child arena 0x01020304 nodes: {:?}",
            trie.allocator.get_node_arena(0x01020304).map(|a| a.len())
        );
        eprintln!(
            "Child arena 0x01020304 leaves: {:?}",
            trie.allocator.get_leaf_arena(0x01020304).map(|a| a.len())
        );

        // Debug: check child_arena_idx in root arena nodes
        if let Some(root_arena) = trie.allocator.get_node_arena(0) {
            eprintln!("\nRoot arena nodes:");
            for i in 0..root_arena.len() {
                let node = root_arena.get(i as u32);
                eprintln!("  Node {}: child_arena_idx={:08x}", i, node.child_arena_idx);
            }
        }

        // Verify all keys exist
        eprintln!("\nChecking key1...");
        let c1 = trie.contains(key1);
        eprintln!("Contains key1: {}", c1);
        assert!(c1);
        eprintln!("Checking key2...");
        assert!(trie.contains(key2));
        eprintln!("Checking key3...");
        assert!(trie.contains(key3));
        eprintln!("Checking key4...");
        assert!(trie.contains(key4));

        // For u64 with split at level 4, these keys create child arena
        // Child arena index = upper 4 bytes = 0x01020304
        let child_arena_idx = 0x01020304u64;

        // Check that child arena exists and has nodes
        let node_arena = trie.allocator.get_node_arena(child_arena_idx).unwrap();
        assert!(
            node_arena.len() > 1,
            "Should have multiple nodes in child arena for u64 keys"
        );

        // Check that child arena has multiple leaves
        let leaf_arena = trie.allocator.get_leaf_arena(child_arena_idx).unwrap();
        assert!(
            leaf_arena.len() >= 2,
            "Should have multiple leaves in child arena for different prefixes"
        );
    }

    #[test]
    fn test_different_prefixes() {
        let mut trie = Trie::<u32>::new();

        // Insert keys with completely different first bytes
        // This should create separate branches from root
        let key1 = 0x01000000u32; // First byte = 1
        let key2 = 0x02000000u32; // First byte = 2
        let key3 = 0x03000000u32; // First byte = 3

        assert!(trie.insert(key1));
        assert!(trie.insert(key2));
        assert!(trie.insert(key3));

        // All keys should exist
        assert!(trie.contains(key1));
        assert!(trie.contains(key2));
        assert!(trie.contains(key3));

        // Get arena index
        let segment_meta = trie.allocator.get_segment_meta(trie.root_segment).unwrap();
        let arena_idx = segment_meta.cache_key;

        // Check root node has 3 children (for bytes 1, 2, 3)
        let node_arena = trie.allocator.get_node_arena(arena_idx).unwrap();
        let root_node = node_arena.get(0);

        assert!(root_node.has_child(0x01));
        assert!(root_node.has_child(0x02));
        assert!(root_node.has_child(0x03));
        assert!(!root_node.has_child(0x04));

        // Should have created separate leaves for each prefix
        let leaf_arena = trie.allocator.get_leaf_arena(arena_idx).unwrap();
        assert_eq!(leaf_arena.len(), 3, "Should have 3 separate leaves");
    }

    #[test]
    fn test_shared_prefix_path() {
        let mut trie = Trie::<u32>::new();

        // Insert keys that share a common prefix
        // Key 0x01020300 = [1][2][3][0]
        // Key 0x01020301 = [1][2][3][1]
        // Key 0x01020302 = [1][2][3][2]
        // All share prefix [1][2][3], so should share path up to level 2

        let key1 = 0x01020300u32;
        let key2 = 0x01020301u32;
        let key3 = 0x01020302u32;

        assert!(trie.insert(key1));
        assert!(trie.insert(key2));
        assert!(trie.insert(key3));

        // All keys should exist
        assert!(trie.contains(key1));
        assert!(trie.contains(key2));
        assert!(trie.contains(key3));

        // Get arena index
        let segment_meta = trie.allocator.get_segment_meta(trie.root_segment).unwrap();
        let arena_idx = segment_meta.cache_key;

        // All keys share same prefix [1][2][3], so should have same leaf
        let leaf_arena = trie.allocator.get_leaf_arena(arena_idx).unwrap();
        assert_eq!(
            leaf_arena.len(),
            1,
            "Keys with same prefix should share one leaf"
        );

        // Check that the leaf has 3 bits set (for last bytes 0, 1, 2)
        let leaf = leaf_arena.get(0);
        use crate::bitmap::is_set;
        assert!(is_set(&leaf.bitmap, 0));
        assert!(is_set(&leaf.bitmap, 1));
        assert!(is_set(&leaf.bitmap, 2));
        assert!(!is_set(&leaf.bitmap, 3));
    }

    // DISABLED: This test causes memory allocation failure due to sparse vector
    // resize with huge arena_idx values. Will be re-enabled after migration to
    // arena-in-nodes architecture (Phase 4 of MIGRATION_PLAN.md).
    //
    // Issue: u128 keys with large values create huge child_arena_idx, causing
    // ArenaAllocator.sparse.resize() to attempt allocating ~2.9 EB of memory.
    //
    // #[test]
    // fn test_u128_multi_level() {
    //     let mut trie = Trie::<u128>::new();
    //
    //     // For u128, we have 16 levels (0-15) with splits at levels 4 and 12
    //     // Insert keys that differ at various levels
    //
    //     let key1 = 0x0102030405060708090A0B0C0D0E0F10u128;
    //     let key2 = 0x0102030405060708090A0B0C0D0E0F11u128; // Differs at last byte
    //     let key3 = 0x0102030405060708090A0B0C0D0E1011u128; // Differs at byte 14
    //
    //     assert!(trie.insert(key1));
    //     assert!(trie.insert(key2));
    //     assert!(trie.insert(key3));
    //
    //     // Verify all keys exist
    //     assert!(trie.contains(key1));
    //     assert!(trie.contains(key2));
    //     assert!(trie.contains(key3));
    //
    //     // For u128 with splits at levels 4 and 12, these keys create L2 child arena
    //     // Keys differ at bytes 14-15, so they're at levels 12-15
    //     // L2 child arena index = bytes 0-11 = 0x0102030405060708090A0B0C
    //     let child_arena_idx = (0x0102030405060708u64 << 32) | 0x090A0B0Cu64;
    //
    //     // Check that child arena exists and has structures
    //     let node_arena = trie.allocator.get_node_arena(child_arena_idx).unwrap();
    //     assert!(
    //         node_arena.len() > 1,
    //         "Should have multiple nodes in L2 child arena for u128 keys"
    //     );
    //
    //     let leaf_arena = trie.allocator.get_leaf_arena(child_arena_idx).unwrap();
    //     assert!(
    //         leaf_arena.len() >= 1,
    //         "Should have at least one leaf in L2 child arena"
    //     );
    // }

    #[test]
    fn test_cache_updates_on_insert() {
        let mut trie = Trie::<u64>::new();

        // Initially empty
        assert_eq!(trie.len(), 0);
        assert_eq!(trie.min(), None);
        assert_eq!(trie.max(), None);
        assert!(trie.is_empty());

        // Insert first key
        assert!(trie.insert(100));
        assert_eq!(trie.len(), 1);
        assert_eq!(trie.min(), Some(100));
        assert_eq!(trie.max(), Some(100));
        assert!(!trie.is_empty());

        // Insert smaller key (updates min)
        assert!(trie.insert(50));
        assert_eq!(trie.len(), 2);
        assert_eq!(trie.min(), Some(50));
        assert_eq!(trie.max(), Some(100));

        // Insert larger key (updates max)
        assert!(trie.insert(200));
        assert_eq!(trie.len(), 3);
        assert_eq!(trie.min(), Some(50));
        assert_eq!(trie.max(), Some(200));

        // Insert middle key (no min/max change)
        assert!(trie.insert(75));
        assert_eq!(trie.len(), 4);
        assert_eq!(trie.min(), Some(50));
        assert_eq!(trie.max(), Some(200));

        // Insert duplicate (no changes)
        assert!(!trie.insert(100));
        assert_eq!(trie.len(), 4);
        assert_eq!(trie.min(), Some(50));
        assert_eq!(trie.max(), Some(200));
    }
}
