//! Segment manager for multi-tenant memory management.

use crate::arena::{KeyRange, SegmentCache, SegmentId, SegmentMeta};
use crate::key::TrieKey;
use alloc::vec::Vec;

/// Segment manager for multi-tenant trie.
///
/// Manages segments and per-segment caches for optimal performance.
/// Arenas are now stored in nodes (hierarchical allocation), not here.
///
/// # Type Parameters
/// * `K` - Key type (u32, u64, or u128)
///
/// # Architecture
/// - segments: Vec<Option<SegmentMeta>> - segment metadata indexed by perm_key
/// - segment_caches: Vec<Option<SegmentCache<K>>> - per-segment hot path caches
///
/// # Memory Layout
/// - Lazy allocation: segments and caches created on-demand
/// - Per-segment isolation: each segment has own metadata and cache
/// - No global arena storage: arenas live in nodes at split levels
///
/// # Performance
/// - O(1) segment access by perm_key
/// - O(1) cache access for hot path optimization
/// - Minimal memory overhead (only metadata, no arenas)
#[derive(Debug)]
#[allow(dead_code)]
pub struct SegmentManager<K: TrieKey> {
    /// Segment metadata indexed by permanent key (perm_key).
    /// None = segment not created, Some = segment exists.
    segments: Vec<Option<SegmentMeta>>,

    /// Per-segment caches for hot path optimization.
    /// Parallel to segments Vec, same indexing by perm_key.
    segment_caches: Vec<Option<SegmentCache<K>>>,
}

#[allow(dead_code)]
impl<K: TrieKey> SegmentManager<K> {
    /// Create a new empty segment manager.
    ///
    /// # Performance
    /// O(1) - creates empty vectors
    pub fn new() -> Self {
        Self {
            segments: Vec::new(),
            segment_caches: Vec::new(),
        }
    }

    /// Create a new segment with given key range.
    ///
    /// Allocates segment metadata and cache at the next available perm_key.
    ///
    /// # Arguments
    /// * `key_range` - Range of keys this segment covers
    /// * `numa_node` - NUMA node for this segment (0 for UMA systems)
    ///
    /// # Returns
    /// SegmentId (perm_key) for the new segment
    ///
    /// # Performance
    /// O(1) - Vec push operations
    pub fn create_segment(&mut self, key_range: KeyRange, numa_node: u8) -> SegmentId {
        let perm_key = self.segments.len();

        let meta = SegmentMeta {
            cache_key: 0, // Root arena always at index 0 (in root node)
            key_offset: key_range.start,
            numa_node,
        };

        let cache = SegmentCache::new();

        self.segments.push(Some(meta));
        self.segment_caches.push(Some(cache));

        perm_key as u32
    }

    /// Get segment metadata by segment ID.
    ///
    /// # Arguments
    /// * `segment_id` - Segment identifier
    ///
    /// # Returns
    /// Reference to segment metadata, or None if segment doesn't exist
    ///
    /// # Performance
    /// O(1) - direct Vec indexing
    pub fn get_segment_meta(&self, segment_id: SegmentId) -> Option<&SegmentMeta> {
        let perm_key = segment_id as usize;
        self.segments.get(perm_key).and_then(|opt| opt.as_ref())
    }

    /// Get mutable segment metadata by segment ID.
    ///
    /// # Arguments
    /// * `segment_id` - Segment identifier
    ///
    /// # Returns
    /// Mutable reference to segment metadata, or None if segment doesn't exist
    ///
    /// # Performance
    /// O(1) - direct Vec indexing
    pub fn get_segment_meta_mut(&mut self, segment_id: SegmentId) -> Option<&mut SegmentMeta> {
        let perm_key = segment_id as usize;
        self.segments.get_mut(perm_key).and_then(|opt| opt.as_mut())
    }

    /// Get segment cache by segment ID.
    ///
    /// # Arguments
    /// * `segment_id` - Segment identifier
    ///
    /// # Returns
    /// Reference to segment cache, or None if segment doesn't exist
    ///
    /// # Performance
    /// O(1) - direct Vec indexing
    pub fn get_segment_cache(&self, segment_id: SegmentId) -> Option<&SegmentCache<K>> {
        let perm_key = segment_id as usize;
        self.segment_caches
            .get(perm_key)
            .and_then(|opt| opt.as_ref())
    }

    /// Get mutable segment cache by segment ID.
    ///
    /// # Arguments
    /// * `segment_id` - Segment identifier
    ///
    /// # Returns
    /// Mutable reference to segment cache, or None if segment doesn't exist
    ///
    /// # Performance
    /// O(1) - direct Vec indexing
    pub fn get_segment_cache_mut(&mut self, segment_id: SegmentId) -> Option<&mut SegmentCache<K>> {
        let perm_key = segment_id as usize;
        self.segment_caches
            .get_mut(perm_key)
            .and_then(|opt| opt.as_mut())
    }
}

impl<K: TrieKey> Default for SegmentManager<K> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_segment_manager() {
        let manager = SegmentManager::<u64>::new();
        assert_eq!(manager.segments.len(), 0);
        assert_eq!(manager.segment_caches.len(), 0);
    }

    #[test]
    fn test_create_segment() {
        let mut manager = SegmentManager::<u64>::new();

        let key_range = KeyRange {
            start: 0,
            size: u64::MAX as u128,
        };

        let segment_id = manager.create_segment(key_range, 0);
        assert_eq!(segment_id, 0);

        let meta = manager.get_segment_meta(segment_id).unwrap();
        assert_eq!(meta.cache_key, 0);
        assert_eq!(meta.key_offset, 0);
        assert_eq!(meta.numa_node, 0);
    }

    #[test]
    fn test_get_segment_meta() {
        let mut manager = SegmentManager::<u64>::new();

        let key_range = KeyRange {
            start: 100,
            size: 1000,
        };

        let segment_id = manager.create_segment(key_range, 1);

        let meta = manager.get_segment_meta(segment_id).unwrap();
        assert_eq!(meta.key_offset, 100);
        assert_eq!(meta.numa_node, 1);
    }

    #[test]
    fn test_get_segment_cache() {
        let mut manager = SegmentManager::<u64>::new();

        let key_range = KeyRange {
            start: 0,
            size: 1000,
        };

        let segment_id = manager.create_segment(key_range, 0);

        let cache = manager.get_segment_cache(segment_id);
        assert!(cache.is_some());
    }

    #[test]
    fn test_multiple_segments() {
        let mut manager = SegmentManager::<u64>::new();

        let seg1 = manager.create_segment(
            KeyRange {
                start: 0,
                size: 1000,
            },
            0,
        );
        let seg2 = manager.create_segment(
            KeyRange {
                start: 1000,
                size: 1000,
            },
            1,
        );

        assert_eq!(seg1, 0);
        assert_eq!(seg2, 1);

        let meta1 = manager.get_segment_meta(seg1).unwrap();
        let meta2 = manager.get_segment_meta(seg2).unwrap();

        assert_eq!(meta1.key_offset, 0);
        assert_eq!(meta2.key_offset, 1000);
    }

    #[test]
    fn test_default() {
        let manager = SegmentManager::<u64>::default();
        assert_eq!(manager.segments.len(), 0);
    }
}
