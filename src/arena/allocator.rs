//! Main arena allocator implementation

use crate::arena::{Arena, KeyRange, SegmentCache, SegmentId, SegmentMeta};
use crate::trie::{Leaf, Node};
use alloc::vec::Vec;

/// Arena allocator for multi-tenant memory management.
///
/// Manages segments, arenas, and per-segment caches for optimal performance.
/// Supports both single-tenant (direct access) and multi-tenant (segmented) modes.
///
/// # Architecture
/// - segments: Vec<Option<SegmentMeta>> - segment metadata indexed by perm_key
/// - segment_caches: Vec<Option<SegmentCache>> - per-segment hot path caches
/// - sparse: Vec<u64> - sparse-to-dense mapping (arena_idx -> cache_idx)
/// - dense: Vec<u64> - dense-to-sparse mapping (cache_idx -> arena_idx)
/// - node_arenas: Vec<Arena<Node>> - internal node storage (dense array)
/// - leaf_arenas: Vec<Arena<Leaf>> - leaf node storage (dense array)
/// - node_free_lists: Vec<Vec<u32>> - free node indices for reuse
/// - leaf_free_lists: Vec<Vec<u32>> - free leaf indices for reuse
///
/// # Memory Layout
/// - Lazy allocation: segments, arenas, nodes/leaves created on-demand
/// - Sparse arena allocation: arenas created only for used key ranges
/// - Per-segment isolation: each segment has own arena range
/// - Cache locality: related data stored in same arena
/// - Free list reuse: deleted nodes/leaves are recycled via free lists
///
/// # Sparse/Dense Mapping
/// - sparse[arena_idx] = cache_idx (O(1) lookup from key to dense array)
/// - dense[cache_idx] = arena_idx (O(1) reverse lookup for swap-remove)
/// - Validation: sparse[arena_idx] < dense.len() && dense[sparse[arena_idx]] == arena_idx
#[derive(Debug)]
pub struct ArenaAllocator {
    /// Segment metadata indexed by permanent key (perm_key).
    /// None = segment not created, Some = segment exists.
    segments: Vec<Option<SegmentMeta>>,

    /// Per-segment caches for hot path optimization.
    /// Parallel to segments Vec, same indexing by perm_key.
    segment_caches: Vec<Option<SegmentCache>>,

    /// Sparse-to-dense mapping for arena allocation.
    /// sparse[arena_idx] = cache_idx (position in dense arrays).
    /// u64::MAX = arena not allocated.
    sparse: Vec<u64>,

    /// Dense-to-sparse mapping for arena deallocation.
    /// dense[cache_idx] = arena_idx (original arena index).
    /// Used for O(1) swap-remove updates.
    dense: Vec<u64>,

    /// Internal node arenas for trie structure (dense array).
    /// Indexed by cache_idx from sparse mapping.
    node_arenas: Vec<Arena<Node>>,

    /// Leaf node arenas for bitmap storage (dense array).
    /// Indexed by cache_idx from sparse mapping.
    leaf_arenas: Vec<Arena<Leaf>>,

    /// Free lists for node reuse (parallel to node_arenas).
    /// Each Vec<u32> contains indices of freed nodes in corresponding arena.
    /// Indexed by cache_idx from sparse mapping.
    node_free_lists: Vec<Vec<u32>>,

    /// Free lists for leaf reuse (parallel to leaf_arenas).
    /// Each Vec<u32> contains indices of freed leaves in corresponding arena.
    /// Indexed by cache_idx from sparse mapping.
    leaf_free_lists: Vec<Vec<u32>>,
}

impl ArenaAllocator {
    /// Create a new empty arena allocator.
    ///
    /// Initializes all storage vectors as empty. Segments, arenas, caches,
    /// and free lists are allocated lazily on first use for optimal memory efficiency.
    ///
    /// # Performance
    /// O(1) - creates empty vectors with zero allocations
    ///
    /// # Memory Usage
    /// ~192 bytes (8 empty Vec headers × 24 bytes each)
    pub fn new() -> Self {
        Self {
            segments: Vec::new(),
            segment_caches: Vec::new(),
            sparse: Vec::new(),
            dense: Vec::new(),
            node_arenas: Vec::new(),
            leaf_arenas: Vec::new(),
            node_free_lists: Vec::new(),
            leaf_free_lists: Vec::new(),
        }
    }

    /// Create a new segment with specified key range.
    ///
    /// Allocates segment metadata and cache, but arenas are created lazily
    /// on first key insertion for optimal memory efficiency.
    ///
    /// # Arguments
    /// * `key_range` - Key range specification (start and size)
    /// * `numa_node` - NUMA node for memory allocation (0 for default)
    ///
    /// # Returns
    /// SegmentId (permanent key) for client use
    ///
    /// # Performance
    /// O(1) - allocates metadata only, no arena allocation
    ///
    /// # Memory Usage
    /// ~200 bytes per segment (SegmentMeta + SegmentCache + Vec overhead)
    pub fn create_segment(&mut self, key_range: KeyRange, numa_node: u8) -> SegmentId {
        // Calculate cache_key (will be used as arena_idx later)
        // For now, use segment count as base arena index
        let cache_key = self.segments.len() as u64;

        // Calculate run_length (number of arenas needed for this key range)
        // For now, allocate 1 arena - will expand as needed
        let run_length = 1;

        // Create segment metadata
        let segment_meta = SegmentMeta::new(
            cache_key,
            run_length,
            key_range.start, // key_offset = start of range
            numa_node,
        );

        // Create empty segment cache
        let segment_cache = SegmentCache::new();

        // Find next available segment ID (perm_key)
        let segment_id = self.segments.len() as u32;

        // Store segment metadata and cache
        self.segments.push(Some(segment_meta));
        self.segment_caches.push(Some(segment_cache));

        // Note: Arenas are NOT pre-allocated - they will be created lazily
        // via allocate_arena_for_key() when first key is inserted

        segment_id
    }

    /// Get segment metadata by segment ID.
    ///
    /// Returns reference to segment metadata for the given permanent key.
    /// Used internally for two-level addressing (perm_key → cache_key).
    ///
    /// # Arguments
    /// * `segment_id` - Permanent segment identifier (perm_key)
    ///
    /// # Returns
    /// Option<&SegmentMeta> - Some if segment exists, None if not found
    ///
    /// # Performance
    /// O(1) - direct Vec indexing
    pub fn get_segment_meta(&self, segment_id: SegmentId) -> Option<&SegmentMeta> {
        self.segments
            .get(segment_id as usize)
            .and_then(|opt| opt.as_ref())
    }

    /// Get mutable segment metadata by segment ID.
    ///
    /// Returns mutable reference to segment metadata for updates during
    /// defragmentation or arena expansion.
    ///
    /// # Arguments
    /// * `segment_id` - Permanent segment identifier (perm_key)
    ///
    /// # Returns
    /// Option<&mut SegmentMeta> - Some if segment exists, None if not found
    ///
    /// # Performance
    /// O(1) - direct Vec indexing
    pub fn get_segment_meta_mut(&mut self, segment_id: SegmentId) -> Option<&mut SegmentMeta> {
        self.segments
            .get_mut(segment_id as usize)
            .and_then(|opt| opt.as_mut())
    }

    /// Get segment cache by segment ID.
    ///
    /// Returns reference to segment cache for hot path optimization.
    /// Used for cache hit checks and path retrieval.
    ///
    /// # Arguments
    /// * `segment_id` - Permanent segment identifier (perm_key)
    ///
    /// # Returns
    /// Option<&SegmentCache> - Some if segment exists, None if not found
    ///
    /// # Performance
    /// O(1) - direct Vec indexing
    pub fn get_segment_cache(&self, segment_id: SegmentId) -> Option<&SegmentCache> {
        self.segment_caches
            .get(segment_id as usize)
            .and_then(|opt| opt.as_ref())
    }

    /// Get mutable segment cache by segment ID.
    ///
    /// Returns mutable reference to segment cache for updates after
    /// cache misses or path changes.
    ///
    /// # Arguments
    /// * `segment_id` - Permanent segment identifier (perm_key)
    ///
    /// # Returns
    /// Option<&mut SegmentCache> - Some if segment exists, None if not found
    ///
    /// # Performance
    /// O(1) - direct Vec indexing
    pub fn get_segment_cache_mut(&mut self, segment_id: SegmentId) -> Option<&mut SegmentCache> {
        self.segment_caches
            .get_mut(segment_id as usize)
            .and_then(|opt| opt.as_mut())
    }

    /// Get node arena by arena index.
    ///
    /// Returns reference to node arena for trie traversal and operations.
    /// Uses sparse/dense mapping for O(1) access.
    ///
    /// # Arguments
    /// * `arena_idx` - Arena index from key (via TrieKey::arena_idx())
    ///
    /// # Returns
    /// Option<&Arena<Node>> - Some if arena exists, None if not allocated
    ///
    /// # Performance
    /// O(1) - sparse lookup + dense array access
    pub fn get_node_arena(&self, arena_idx: u64) -> Option<&Arena<Node>> {
        self.get_cache_idx(arena_idx)
            .map(|cache_idx| &self.node_arenas[cache_idx])
    }

    /// Get mutable node arena by arena index.
    ///
    /// Returns mutable reference to node arena for modifications.
    /// Uses sparse/dense mapping for O(1) access.
    ///
    /// # Arguments
    /// * `arena_idx` - Arena index from key (via TrieKey::arena_idx())
    ///
    /// # Returns
    /// Option<&mut Arena<Node>> - Some if arena exists, None if not allocated
    ///
    /// # Performance
    /// O(1) - sparse lookup + dense array access
    pub fn get_node_arena_mut(&mut self, arena_idx: u64) -> Option<&mut Arena<Node>> {
        self.get_cache_idx(arena_idx)
            .map(|cache_idx| &mut self.node_arenas[cache_idx])
    }

    /// Get leaf arena by arena index.
    ///
    /// Returns reference to leaf arena for bitmap operations.
    /// Uses sparse/dense mapping for O(1) access.
    ///
    /// # Arguments
    /// * `arena_idx` - Arena index from key (via TrieKey::arena_idx())
    ///
    /// # Returns
    /// Option<&Arena<Leaf>> - Some if arena exists, None if not allocated
    ///
    /// # Performance
    /// O(1) - sparse lookup + dense array access
    pub fn get_leaf_arena(&self, arena_idx: u64) -> Option<&Arena<Leaf>> {
        self.get_cache_idx(arena_idx)
            .map(|cache_idx| &self.leaf_arenas[cache_idx])
    }

    /// Get mutable leaf arena by arena index.
    ///
    /// Returns mutable reference to leaf arena for modifications.
    /// Uses sparse/dense mapping for O(1) access.
    ///
    /// # Arguments
    /// * `arena_idx` - Arena index from key (via TrieKey::arena_idx())
    ///
    /// # Returns
    /// Option<&mut Arena<Leaf>> - Some if arena exists, None if not allocated
    ///
    /// # Performance
    /// O(1) - sparse lookup + dense array access
    pub fn get_leaf_arena_mut(&mut self, arena_idx: u64) -> Option<&mut Arena<Leaf>> {
        self.get_cache_idx(arena_idx)
            .map(|cache_idx| &mut self.leaf_arenas[cache_idx])
    }

    /// Check if arena exists for given arena index.
    ///
    /// Uses sparse/dense mapping for O(1) validation.
    ///
    /// # Arguments
    /// * `arena_idx` - Arena index from key (via TrieKey::arena_idx())
    ///
    /// # Returns
    /// bool - true if arena is allocated, false otherwise
    ///
    /// # Performance
    /// O(1) - two Vec lookups with bounds checks
    #[inline]
    fn has_arena(&self, arena_idx: u64) -> bool {
        let sparse_idx = arena_idx as usize;

        // Check if arena_idx is within sparse bounds
        if sparse_idx >= self.sparse.len() {
            return false;
        }

        let cache_idx = self.sparse[sparse_idx];

        // Validate: cache_idx must be valid and point back to arena_idx
        cache_idx != u64::MAX
            && (cache_idx as usize) < self.dense.len()
            && self.dense[cache_idx as usize] == arena_idx
    }

    /// Get cache index for arena index.
    ///
    /// Translates sparse arena_idx to dense cache_idx for array access.
    ///
    /// # Arguments
    /// * `arena_idx` - Arena index from key (via TrieKey::arena_idx())
    ///
    /// # Returns
    /// Option<usize> - Some(cache_idx) if arena exists, None otherwise
    ///
    /// # Performance
    /// O(1) - single Vec lookup with validation
    #[inline]
    fn get_cache_idx(&self, arena_idx: u64) -> Option<usize> {
        if self.has_arena(arena_idx) {
            Some(self.sparse[arena_idx as usize] as usize)
        } else {
            None
        }
    }

    /// Allocate arena for given arena index if not exists.
    ///
    /// Creates new arena in dense arrays and updates sparse/dense mappings.
    /// Idempotent - safe to call multiple times for same arena_idx.
    ///
    /// # Arguments
    /// * `arena_idx` - Arena index from key (via TrieKey::arena_idx())
    ///
    /// # Returns
    /// usize - cache_idx for accessing dense arrays
    ///
    /// # Performance
    /// O(1) - Vec push and sparse resize if needed
    ///
    /// # Memory
    /// - Sparse grows to arena_idx + 1 (lazy, only used indices)
    /// - Dense grows by 1 (compact, no gaps)
    /// - Arenas grow by 1 each (node + leaf)
    fn allocate_arena_for_key(&mut self, arena_idx: u64) -> usize {
        // Check if already allocated
        if let Some(cache_idx) = self.get_cache_idx(arena_idx) {
            return cache_idx;
        }

        // Allocate new arena at end of dense arrays
        let cache_idx = self.dense.len();

        // Expand sparse if needed
        let sparse_idx = arena_idx as usize;
        if sparse_idx >= self.sparse.len() {
            self.sparse.resize(sparse_idx + 1, u64::MAX);
        }

        // Update mappings
        self.sparse[sparse_idx] = cache_idx as u64;
        self.dense.push(arena_idx);

        // Allocate arenas
        self.node_arenas.push(Arena::new());
        self.leaf_arenas.push(Arena::new());
        self.node_free_lists.push(Vec::new());
        self.leaf_free_lists.push(Vec::new());

        cache_idx
    }

    /// Deallocate arena for given arena index.
    ///
    /// Removes arena from dense arrays using swap-remove and updates mappings.
    /// Should be called when last key in arena range is removed.
    ///
    /// # Arguments
    /// * `arena_idx` - Arena index to deallocate
    ///
    /// # Performance
    /// O(1) - swap-remove with mapping updates
    ///
    /// # Panics
    /// Panics if arena_idx is not allocated
    fn deallocate_arena(&mut self, arena_idx: u64) {
        let cache_idx = self
            .get_cache_idx(arena_idx)
            .expect("Arena must be allocated to deallocate");

        let last_idx = self.dense.len() - 1;

        // If not last element, swap with last
        if cache_idx != last_idx {
            // Swap in all parallel arrays
            self.dense.swap(cache_idx, last_idx);
            self.node_arenas.swap(cache_idx, last_idx);
            self.leaf_arenas.swap(cache_idx, last_idx);
            self.node_free_lists.swap(cache_idx, last_idx);
            self.leaf_free_lists.swap(cache_idx, last_idx);

            // Update sparse mapping for moved arena
            let moved_arena_idx = self.dense[cache_idx];
            self.sparse[moved_arena_idx as usize] = cache_idx as u64;
        }

        // Remove last element
        self.dense.pop();
        self.node_arenas.pop();
        self.leaf_arenas.pop();
        self.node_free_lists.pop();
        self.leaf_free_lists.pop();

        // Mark as deallocated in sparse
        self.sparse[arena_idx as usize] = u64::MAX;
    }

    /// Allocate node and leaf arenas for a segment if not already allocated.
    ///
    /// Implements lazy allocation - arenas are created only when first needed.
    /// This method ensures both node and leaf arenas exist for the given segment.
    ///
    /// # Arguments
    /// * `segment_id` - Permanent segment identifier (perm_key)
    ///
    /// # Returns
    /// Result<(), &'static str> - Ok if successful, Err with message if failed
    ///
    /// # Performance
    /// O(1) - sparse/dense mapping with arena creation
    ///
    /// # Memory Usage
    /// Allocates empty arenas (~24 bytes each) that grow as nodes/leaves are added
    pub fn allocate_arena(&mut self, segment_id: SegmentId) -> Result<(), &'static str> {
        // Get segment metadata to find arena index
        let segment_meta = self
            .get_segment_meta(segment_id)
            .ok_or("Segment not found")?;

        let arena_idx = segment_meta.cache_key;

        // Allocate arena using sparse/dense mapping
        self.allocate_arena_for_key(arena_idx);

        Ok(())
    }

    /// Allocate a node, reusing from free list if available.
    ///
    /// Attempts to reuse a freed node from the free list first.
    /// If no freed nodes are available, allocates a new node from the arena.
    ///
    /// # Arguments
    /// * `arena_idx` - Arena index from key (via TrieKey::arena_idx())
    ///
    /// # Returns
    /// u32 - Index of allocated node in the arena
    ///
    /// # Performance
    /// - Free list hit: O(1) - pop from Vec (~2-3 cycles)
    /// - Free list miss: O(1) - arena allocation (~5-10 cycles)
    ///
    /// # Panics
    /// Panics if arena_idx is not allocated
    #[inline]
    pub fn alloc_node(&mut self, arena_idx: u64) -> u32 {
        let cache_idx = self
            .get_cache_idx(arena_idx)
            .expect("Arena must be allocated before alloc_node");

        // Try to reuse from free list first
        if let Some(idx) = self.node_free_lists[cache_idx].pop() {
            return idx;
        }

        // No free nodes - allocate new one
        self.node_arenas[cache_idx].alloc()
    }

    /// Free a node, adding it to the free list for reuse.
    ///
    /// Marks the node as freed by adding its index to the free list.
    /// The node memory is not cleared - it will be overwritten on reuse.
    ///
    /// # Arguments
    /// * `arena_idx` - Arena index from key (via TrieKey::arena_idx())
    /// * `node_idx` - Index of node to free
    ///
    /// # Performance
    /// O(1) - push to Vec (~2-3 cycles)
    ///
    /// # Note
    /// The caller is responsible for ensuring the node is actually empty
    /// and no longer referenced before freeing it.
    #[inline]
    pub fn free_node(&mut self, arena_idx: u64, node_idx: u32) {
        let cache_idx = self
            .get_cache_idx(arena_idx)
            .expect("Arena must be allocated before free_node");
        self.node_free_lists[cache_idx].push(node_idx);
    }

    /// Allocate a leaf, reusing from free list if available.
    ///
    /// Attempts to reuse a freed leaf from the free list first.
    /// If no freed leaves are available, allocates a new leaf from the arena.
    ///
    /// # Arguments
    /// * `arena_idx` - Arena index from key (via TrieKey::arena_idx())
    /// * `prefix` - 56-bit prefix for the leaf (upper bits of keys)
    ///
    /// # Returns
    /// u32 - Index of allocated leaf in the arena
    ///
    /// # Performance
    /// - Free list hit: O(1) - pop from Vec (~2-3 cycles)
    /// - Free list miss: O(1) - arena allocation (~5-10 cycles)
    ///
    /// # Panics
    /// Panics if arena_idx is not allocated
    #[inline]
    pub fn alloc_leaf(&mut self, arena_idx: u64, prefix: u64) -> u32 {
        let cache_idx = self
            .get_cache_idx(arena_idx)
            .expect("Arena must be allocated before alloc_leaf");

        // Try to reuse from free list first
        if let Some(idx) = self.leaf_free_lists[cache_idx].pop() {
            // Reinitialize the leaf with new prefix
            let leaf = self.leaf_arenas[cache_idx].get_mut(idx);
            *leaf = crate::trie::Leaf::new(prefix);
            return idx;
        }

        // No free leaves - allocate new one
        self.leaf_arenas[cache_idx].alloc(prefix)
    }

    /// Free a leaf, adding it to the free list for reuse.
    ///
    /// Marks the leaf as freed by adding its index to the free list.
    /// The leaf memory is not cleared - it will be reinitialized on reuse.
    ///
    /// # Arguments
    /// * `arena_idx` - Arena index from key (via TrieKey::arena_idx())
    /// * `leaf_idx` - Index of leaf to free
    ///
    /// # Performance
    /// O(1) - push to Vec (~2-3 cycles)
    ///
    /// # Note
    /// The caller is responsible for ensuring the leaf is actually empty
    /// and no longer referenced before freeing it.
    #[inline]
    pub fn free_leaf(&mut self, arena_idx: u64, leaf_idx: u32) {
        let cache_idx = self
            .get_cache_idx(arena_idx)
            .expect("Arena must be allocated before free_leaf");
        self.leaf_free_lists[cache_idx].push(leaf_idx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_allocator() {
        let allocator = ArenaAllocator::new();

        assert_eq!(allocator.segments.len(), 0);
        assert_eq!(allocator.segment_caches.len(), 0);
        assert_eq!(allocator.node_arenas.len(), 0);
        assert_eq!(allocator.leaf_arenas.len(), 0);
    }

    #[test]
    fn test_create_segment() {
        let mut allocator = ArenaAllocator::new();

        let key_range = KeyRange {
            start: 1000,
            size: 500000,
        };

        let segment_id = allocator.create_segment(key_range, 0);

        // Check segment ID
        assert_eq!(segment_id, 0);

        // Check vectors grew
        assert_eq!(allocator.segments.len(), 1);
        assert_eq!(allocator.segment_caches.len(), 1);
        assert_eq!(allocator.node_arenas.len(), 1);
        assert_eq!(allocator.leaf_arenas.len(), 1);

        // Check segment metadata
        let segment_meta = allocator.segments[0].as_ref().unwrap();
        assert_eq!(segment_meta.cache_key, 0);
        assert_eq!(segment_meta.run_length, 1);
        assert_eq!(segment_meta.key_offset, 1000);
        assert_eq!(segment_meta.numa_node, 0);

        // Check segment cache exists
        assert!(allocator.segment_caches[0].is_some());

        // Check arenas are reserved but not allocated (lazy allocation)
        assert!(allocator.node_arenas[0].is_none());
        assert!(allocator.leaf_arenas[0].is_none());
    }

    #[test]
    fn test_create_multiple_segments() {
        let mut allocator = ArenaAllocator::new();

        let segment1 = allocator.create_segment(
            KeyRange {
                start: 0,
                size: 1000,
            },
            0,
        );
        let segment2 = allocator.create_segment(
            KeyRange {
                start: 2000,
                size: 3000,
            },
            1,
        );

        assert_eq!(segment1, 0);
        assert_eq!(segment2, 1);

        assert_eq!(allocator.segments.len(), 2);
        assert_eq!(allocator.segment_caches.len(), 2);
        assert_eq!(allocator.node_arenas.len(), 2);
        assert_eq!(allocator.leaf_arenas.len(), 2);

        // Check first segment
        let meta1 = allocator.segments[0].as_ref().unwrap();
        assert_eq!(meta1.cache_key, 0);
        assert_eq!(meta1.key_offset, 0);
        assert_eq!(meta1.numa_node, 0);

        // Check second segment
        let meta2 = allocator.segments[1].as_ref().unwrap();
        assert_eq!(meta2.cache_key, 1);
        assert_eq!(meta2.key_offset, 2000);
        assert_eq!(meta2.numa_node, 1);
    }

    #[test]
    fn test_get_segment_meta() {
        let mut allocator = ArenaAllocator::new();

        // Test non-existent segment
        assert!(allocator.get_segment_meta(0).is_none());
        assert!(allocator.get_segment_meta(999).is_none());

        // Create segment
        let segment_id = allocator.create_segment(
            KeyRange {
                start: 5000,
                size: 1000,
            },
            2,
        );

        // Test existing segment
        let meta = allocator.get_segment_meta(segment_id).unwrap();
        assert_eq!(meta.cache_key, 0);
        assert_eq!(meta.run_length, 1);
        assert_eq!(meta.key_offset, 5000);
        assert_eq!(meta.numa_node, 2);

        // Test still non-existent segment
        assert!(allocator.get_segment_meta(segment_id + 1).is_none());
    }

    #[test]
    fn test_get_segment_meta_mut() {
        let mut allocator = ArenaAllocator::new();

        // Test non-existent segment
        assert!(allocator.get_segment_meta_mut(0).is_none());

        // Create segment
        let segment_id = allocator.create_segment(
            KeyRange {
                start: 1000,
                size: 2000,
            },
            0,
        );

        // Test mutable access
        {
            let meta = allocator.get_segment_meta_mut(segment_id).unwrap();
            assert_eq!(meta.numa_node, 0);

            // Modify metadata (simulate defragmentation)
            meta.cache_key = 42;
            meta.numa_node = 3;
        }

        // Verify changes
        let meta = allocator.get_segment_meta(segment_id).unwrap();
        assert_eq!(meta.cache_key, 42);
        assert_eq!(meta.numa_node, 3);
        assert_eq!(meta.key_offset, 1000); // Unchanged
    }

    #[test]
    fn test_get_segment_cache() {
        let mut allocator = ArenaAllocator::new();

        // Test non-existent segment
        assert!(allocator.get_segment_cache(0).is_none());
        assert!(allocator.get_segment_cache(999).is_none());

        // Create segment
        let segment_id = allocator.create_segment(
            KeyRange {
                start: 2000,
                size: 1000,
            },
            1,
        );

        // Test existing segment cache
        let cache = allocator.get_segment_cache(segment_id).unwrap();
        assert_eq!(cache.leaf_idx(), u32::MAX); // Empty cache
        assert!(!cache.is_valid(123)); // Invalid cache

        // Test still non-existent segment
        assert!(allocator.get_segment_cache(segment_id + 1).is_none());
    }

    #[test]
    fn test_get_segment_cache_mut() {
        let mut allocator = ArenaAllocator::new();

        // Test non-existent segment
        assert!(allocator.get_segment_cache_mut(0).is_none());

        // Create segment
        let segment_id = allocator.create_segment(
            KeyRange {
                start: 3000,
                size: 2000,
            },
            0,
        );

        // Test mutable cache access
        {
            let cache = allocator.get_segment_cache_mut(segment_id).unwrap();
            assert!(!cache.is_valid(456));

            // Update cache (simulate cache hit)
            cache.update(0x1234567890ABCDEF, 42, &[10, 20, 30], &[1, 2, 3]);
        }

        // Verify cache changes
        let cache = allocator.get_segment_cache(segment_id).unwrap();
        assert!(cache.is_valid(0x1234567890ABCDEF));
        assert_eq!(cache.leaf_idx(), 42);
        assert_eq!(cache.path_node(0), 10);
        assert_eq!(cache.path_byte(0), 1);
    }

    #[test]
    fn test_get_node_arena() {
        let mut allocator = ArenaAllocator::new();

        // Test non-existent arena
        assert!(allocator.get_node_arena(0).is_none());
        assert!(allocator.get_node_arena(999).is_none());

        // Create segment (reserves arena slot but doesn't allocate)
        let segment_id = allocator.create_segment(
            KeyRange {
                start: 1000,
                size: 2000,
            },
            0,
        );
        let meta = allocator.get_segment_meta(segment_id).unwrap();
        let arena_idx = meta.cache_key;

        // Arena should be reserved but not allocated (lazy allocation)
        assert!(allocator.get_node_arena(arena_idx).is_none());

        // Manually allocate arena for testing
        allocator.node_arenas[arena_idx as usize] = Some(Arena::new());

        // Now arena should exist
        let arena = allocator.get_node_arena(arena_idx).unwrap();
        assert_eq!(arena.len(), 0); // Empty arena
    }

    #[test]
    fn test_get_node_arena_mut() {
        let mut allocator = ArenaAllocator::new();

        // Create segment and manually allocate arena
        let segment_id = allocator.create_segment(
            KeyRange {
                start: 2000,
                size: 1000,
            },
            1,
        );
        let meta = allocator.get_segment_meta(segment_id).unwrap();
        let arena_idx = meta.cache_key;

        allocator.node_arenas[arena_idx as usize] = Some(Arena::with_capacity(10));

        // Test mutable access
        {
            let arena = allocator.get_node_arena_mut(arena_idx).unwrap();
            assert_eq!(arena.len(), 0);

            // Add node to arena
            let _node_idx = arena.alloc();
        }

        // Verify changes
        let arena = allocator.get_node_arena(arena_idx).unwrap();
        assert_eq!(arena.len(), 1); // One node added
    }

    #[test]
    fn test_get_leaf_arena() {
        let mut allocator = ArenaAllocator::new();

        // Test non-existent arena
        assert!(allocator.get_leaf_arena(0).is_none());

        // Create segment
        let segment_id = allocator.create_segment(
            KeyRange {
                start: 3000,
                size: 1000,
            },
            0,
        );
        let meta = allocator.get_segment_meta(segment_id).unwrap();
        let arena_idx = meta.cache_key;

        // Arena should be reserved but not allocated
        assert!(allocator.get_leaf_arena(arena_idx).is_none());

        // Manually allocate arena
        allocator.leaf_arenas[arena_idx as usize] = Some(Arena::new());

        // Now arena should exist
        let arena = allocator.get_leaf_arena(arena_idx).unwrap();
        assert_eq!(arena.len(), 0);
    }

    #[test]
    fn test_get_leaf_arena_mut() {
        let mut allocator = ArenaAllocator::new();

        // Create segment and manually allocate arena
        let segment_id = allocator.create_segment(
            KeyRange {
                start: 4000,
                size: 500,
            },
            2,
        );
        let meta = allocator.get_segment_meta(segment_id).unwrap();
        let arena_idx = meta.cache_key;

        allocator.leaf_arenas[arena_idx as usize] = Some(Arena::with_capacity(5));

        // Test mutable access
        {
            let arena = allocator.get_leaf_arena_mut(arena_idx).unwrap();
            assert_eq!(arena.len(), 0);

            // Add leaf to arena
            let _leaf_idx = arena.alloc(0x1234567890ABCDEF);
        }

        // Verify changes
        let arena = allocator.get_leaf_arena(arena_idx).unwrap();
        assert_eq!(arena.len(), 1); // One leaf added
    }

    #[test]
    fn test_allocate_arena() {
        let mut allocator = ArenaAllocator::new();

        // Test non-existent segment
        assert!(allocator.allocate_arena(999).is_err());

        // Create segment
        let segment_id = allocator.create_segment(
            KeyRange {
                start: 5000,
                size: 1000,
            },
            0,
        );
        let meta = allocator.get_segment_meta(segment_id).unwrap();
        let arena_idx = meta.cache_key;

        // Initially arenas should not be allocated
        assert!(allocator.get_node_arena(arena_idx).is_none());
        assert!(allocator.get_leaf_arena(arena_idx).is_none());

        // Allocate arenas
        assert!(allocator.allocate_arena(segment_id).is_ok());

        // Now arenas should exist
        assert!(allocator.get_node_arena(arena_idx).is_some());
        assert!(allocator.get_leaf_arena(arena_idx).is_some());

        // Both arenas should be empty initially
        assert_eq!(allocator.get_node_arena(arena_idx).unwrap().len(), 0);
        assert_eq!(allocator.get_leaf_arena(arena_idx).unwrap().len(), 0);

        // Calling allocate_arena again should be safe (idempotent)
        assert!(allocator.allocate_arena(segment_id).is_ok());
        assert!(allocator.get_node_arena(arena_idx).is_some());
        assert!(allocator.get_leaf_arena(arena_idx).is_some());
    }
}
