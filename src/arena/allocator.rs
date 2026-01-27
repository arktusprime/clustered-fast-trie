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
/// - node_arenas: Vec<Option<Arena<Node>>> - internal node storage
/// - leaf_arenas: Vec<Option<Arena<Leaf>>> - leaf node storage
///
/// # Memory Layout
/// - Lazy allocation: segments, arenas, nodes/leaves created on-demand
/// - Per-segment isolation: each segment has own arena range
/// - Cache locality: related data stored in same arena
#[derive(Debug)]
pub struct ArenaAllocator {
    /// Segment metadata indexed by permanent key (perm_key).
    /// None = segment not created, Some = segment exists.
    segments: Vec<Option<SegmentMeta>>,

    /// Per-segment caches for hot path optimization.
    /// Parallel to segments Vec, same indexing by perm_key.
    segment_caches: Vec<Option<SegmentCache>>,

    /// Internal node arenas for trie structure.
    /// Indexed by cache_key from SegmentMeta.
    node_arenas: Vec<Option<Arena<Node>>>,

    /// Leaf node arenas for bitmap storage.
    /// Indexed by cache_key from SegmentMeta.
    leaf_arenas: Vec<Option<Arena<Leaf>>>,
}

impl ArenaAllocator {
    /// Create a new empty arena allocator.
    ///
    /// Initializes all storage vectors as empty. Segments, arenas, and caches
    /// are allocated lazily on first use for optimal memory efficiency.
    ///
    /// # Performance
    /// O(1) - creates empty vectors with zero allocations
    ///
    /// # Memory Usage
    /// ~96 bytes (4 empty Vec headers × 24 bytes each)
    pub fn new() -> Self {
        Self {
            segments: Vec::new(),
            segment_caches: Vec::new(),
            node_arenas: Vec::new(),
            leaf_arenas: Vec::new(),
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
        // Calculate cache_key (physical position in arena Vec)
        let cache_key = self.node_arenas.len() as u32;

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

        // Reserve arena slots (but don't allocate yet - lazy allocation)
        self.node_arenas.push(None);
        self.leaf_arenas.push(None);

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
    /// Arena may be None if not yet allocated (lazy allocation).
    ///
    /// # Arguments
    /// * `arena_idx` - Arena index (cache_key from SegmentMeta)
    ///
    /// # Returns
    /// Option<&Arena<Node>> - Some if arena exists, None if not allocated
    ///
    /// # Performance
    /// O(1) - direct Vec indexing
    pub fn get_node_arena(&self, arena_idx: u32) -> Option<&Arena<Node>> {
        self.node_arenas
            .get(arena_idx as usize)
            .and_then(|opt| opt.as_ref())
    }

    /// Get mutable node arena by arena index.
    ///
    /// Returns mutable reference to node arena for modifications.
    /// Arena may be None if not yet allocated (lazy allocation).
    ///
    /// # Arguments
    /// * `arena_idx` - Arena index (cache_key from SegmentMeta)
    ///
    /// # Returns
    /// Option<&mut Arena<Node>> - Some if arena exists, None if not allocated
    ///
    /// # Performance
    /// O(1) - direct Vec indexing
    pub fn get_node_arena_mut(&mut self, arena_idx: u32) -> Option<&mut Arena<Node>> {
        self.node_arenas
            .get_mut(arena_idx as usize)
            .and_then(|opt| opt.as_mut())
    }

    /// Get leaf arena by arena index.
    ///
    /// Returns reference to leaf arena for bitmap operations.
    /// Arena may be None if not yet allocated (lazy allocation).
    ///
    /// # Arguments
    /// * `arena_idx` - Arena index (cache_key from SegmentMeta)
    ///
    /// # Returns
    /// Option<&Arena<Leaf>> - Some if arena exists, None if not allocated
    ///
    /// # Performance
    /// O(1) - direct Vec indexing
    pub fn get_leaf_arena(&self, arena_idx: u32) -> Option<&Arena<Leaf>> {
        self.leaf_arenas
            .get(arena_idx as usize)
            .and_then(|opt| opt.as_ref())
    }

    /// Get mutable leaf arena by arena index.
    ///
    /// Returns mutable reference to leaf arena for modifications.
    /// Arena may be None if not yet allocated (lazy allocation).
    ///
    /// # Arguments
    /// * `arena_idx` - Arena index (cache_key from SegmentMeta)
    ///
    /// # Returns
    /// Option<&mut Arena<Leaf>> - Some if arena exists, None if not allocated
    ///
    /// # Performance
    /// O(1) - direct Vec indexing
    pub fn get_leaf_arena_mut(&mut self, arena_idx: u32) -> Option<&mut Arena<Leaf>> {
        self.leaf_arenas
            .get_mut(arena_idx as usize)
            .and_then(|opt| opt.as_mut())
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
    /// O(1) - direct Vec indexing and arena creation
    ///
    /// # Memory Usage
    /// Allocates empty arenas (~24 bytes each) that grow as nodes/leaves are added
    pub fn allocate_arena(&mut self, segment_id: SegmentId) -> Result<(), &'static str> {
        // Get segment metadata to find arena index
        let segment_meta = self
            .get_segment_meta(segment_id)
            .ok_or("Segment not found")?;

        let arena_idx = segment_meta.cache_key as usize;

        // Ensure vectors are large enough
        if arena_idx >= self.node_arenas.len() {
            return Err("Arena index out of bounds");
        }

        // Allocate node arena if not exists
        if self.node_arenas[arena_idx].is_none() {
            self.node_arenas[arena_idx] = Some(Arena::new());
        }

        // Allocate leaf arena if not exists
        if self.leaf_arenas[arena_idx].is_none() {
            self.leaf_arenas[arena_idx] = Some(Arena::new());
        }

        Ok(())
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
