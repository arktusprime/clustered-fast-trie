//! Segment metadata and management

use crate::key::TrieKey;

/// Segment identifier (permanent key).
///
/// This is the stable identifier given to clients.
/// It never changes even during defragmentation.
pub type SegmentId = u32;

/// Segment metadata structure.
///
/// Maps permanent segment ID to physical arena location with key transposition.
///
/// # Memory Layout
/// - `cache_key`: 4 bytes - physical position in arena Vec
/// - `run_length`: 4 bytes - number of consecutive arenas
/// - `key_offset`: 16 bytes - transposition offset for key range (u128 for all key types)
/// - `numa_node`: 1 byte - NUMA node for locality
/// - Total: 25 bytes (padded to 32 bytes by compiler)
///
/// # Two-Level Addressing
/// - Client uses `SegmentId` (perm_key)
/// - System maps to `cache_key` for O(1) arena access
/// - Enables transparent defragmentation
///
/// # Key Transposition
/// - Client keys: arbitrary range [start, start + size)
/// - Internal keys: relative_key = client_key - key_offset
/// - Arena selection: arena_idx = cache_key + (relative_key / 2^32)
///
/// # Key Type Support
/// - Works with u32, u64, u128 keys via TrieKey trait
/// - key_offset is always u128 (sufficient for all key types)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SegmentMeta {
    /// Physical position in arena Vec.
    ///
    /// Points to first arena of this segment.
    /// Updated during defragmentation.
    pub cache_key: u32,

    /// Number of consecutive arenas allocated for this segment.
    ///
    /// Segment can hold up to `run_length × 2^32` keys.
    /// For u32 keys: typically 1 arena (covers full range)
    /// For u64/u128 keys: multiple arenas as needed
    pub run_length: u32,

    /// Transposition offset for key range mapping.
    ///
    /// Converts client keys to internal relative keys:
    /// `relative_key = client_key - key_offset`
    ///
    /// Uses u128 to support all key types (u32, u64, u128).
    /// Allows segments to use arbitrary key ranges.
    pub key_offset: u128,

    /// NUMA node for memory allocation.
    ///
    /// All arenas of this segment allocated on same NUMA node
    /// for optimal memory locality and performance.
    pub numa_node: u8,
}

/// Key range specification for segment creation.
///
/// Defines the logical key range that a segment will handle.
/// Uses u128 to support all key types.
pub struct KeyRange {
    /// Start of the key range (inclusive).
    pub start: u128,
    /// Size of the key range (number of keys).
    pub size: u128,
}

impl SegmentMeta {
    /// Create new segment metadata.
    ///
    /// # Arguments
    /// * `cache_key` - Physical position in arena Vec
    /// * `run_length` - Number of consecutive arenas
    /// * `key_offset` - Transposition offset for key mapping
    /// * `numa_node` - NUMA node for allocation
    ///
    /// # Returns
    /// New segment metadata
    ///
    /// # Performance
    /// O(1) - simple struct initialization
    #[inline(always)]
    pub fn new(cache_key: u32, run_length: u32, key_offset: u128, numa_node: u8) -> Self {
        SegmentMeta {
            cache_key,
            run_length,
            key_offset,
            numa_node,
        }
    }

    /// Get physical arena index for a given key.
    ///
    /// # Arguments
    /// * `key` - Client key to map (u32, u64, or u128)
    ///
    /// # Returns
    /// Physical arena index in the arena Vec
    ///
    /// # Performance
    /// O(1) - arithmetic operations only
    ///
    /// # Formula
    /// ```text
    /// relative_key = key - key_offset
    /// arena_offset = relative_key / 2^32
    /// arena_idx = cache_key + arena_offset
    /// ```
    #[inline(always)]
    pub fn arena_index<K: TrieKey>(&self, key: K) -> u32 {
        let key_u128 = key.to_u128();
        let relative_key = key_u128.wrapping_sub(self.key_offset);
        let arena_offset = (relative_key >> 32) as u32;
        self.cache_key.wrapping_add(arena_offset)
    }

    /// Get local key within an arena.
    ///
    /// # Arguments
    /// * `key` - Client key to map (u32, u64, or u128)
    ///
    /// # Returns
    /// Local key within the arena (0 to 2^32-1)
    ///
    /// # Performance
    /// O(1) - arithmetic operations only
    ///
    /// # Formula
    /// ```text
    /// relative_key = key - key_offset
    /// local_key = relative_key % 2^32
    /// ```
    #[inline(always)]
    pub fn local_key<K: TrieKey>(&self, key: K) -> u32 {
        let key_u128 = key.to_u128();
        let relative_key = key_u128.wrapping_sub(self.key_offset);
        relative_key as u32
    }
}

impl Default for SegmentMeta {
    fn default() -> Self {
        Self::new(0, 1, 0, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_segment_meta() {
        let meta = SegmentMeta::new(100, 5, 1000, 0);

        assert_eq!(meta.cache_key, 100);
        assert_eq!(meta.run_length, 5);
        assert_eq!(meta.key_offset, 1000);
        assert_eq!(meta.numa_node, 0);
    }

    #[test]
    fn test_arena_index_u32_no_offset() {
        let meta = SegmentMeta::new(0, 1, 0, 0);

        // u32 keys always in first arena
        assert_eq!(meta.arena_index(0u32), 0);
        assert_eq!(meta.arena_index(100u32), 0);
        assert_eq!(meta.arena_index(u32::MAX), 0);
    }

    #[test]
    fn test_arena_index_u64_no_offset() {
        let meta = SegmentMeta::new(0, 1, 0, 0);

        // Keys in first arena (0 to 2^32-1)
        assert_eq!(meta.arena_index(0u64), 0);
        assert_eq!(meta.arena_index(100u64), 0);
        assert_eq!(meta.arena_index(u32::MAX as u64), 0);

        // Keys in second arena (2^32 to 2×2^32-1)
        assert_eq!(meta.arena_index(1u64 << 32), 1);
        assert_eq!(meta.arena_index((1u64 << 32) + 100), 1);
    }

    #[test]
    fn test_arena_index_u128_no_offset() {
        let meta = SegmentMeta::new(0, 1, 0, 0);

        // Keys in first arena
        assert_eq!(meta.arena_index(0u128), 0);
        assert_eq!(meta.arena_index(100u128), 0);

        // Keys in second arena
        assert_eq!(meta.arena_index(1u128 << 32), 1);

        // Keys in higher arenas
        assert_eq!(meta.arena_index(1u128 << 64), 1u32 << 32);
    }

    #[test]
    fn test_arena_index_with_offset() {
        let meta = SegmentMeta::new(10, 5, 1000, 0);

        // u64: Key 1000 maps to relative_key 0 → arena 10
        assert_eq!(meta.arena_index(1000u64), 10);

        // u64: Key 1000 + 2^32 maps to relative_key 2^32 → arena 11
        assert_eq!(meta.arena_index(1000u64 + (1u64 << 32)), 11);

        // u128: Key 1000 maps to relative_key 0 → arena 10
        assert_eq!(meta.arena_index(1000u128), 10);
    }

    #[test]
    fn test_local_key_u32() {
        let meta = SegmentMeta::new(0, 1, 0, 0);

        assert_eq!(meta.local_key(0u32), 0);
        assert_eq!(meta.local_key(100u32), 100);
        assert_eq!(meta.local_key(u32::MAX), u32::MAX);
    }

    #[test]
    fn test_local_key_u64_no_offset() {
        let meta = SegmentMeta::new(0, 1, 0, 0);

        assert_eq!(meta.local_key(0u64), 0);
        assert_eq!(meta.local_key(100u64), 100);
        assert_eq!(meta.local_key(u32::MAX as u64), u32::MAX);

        // Keys in second arena
        assert_eq!(meta.local_key(1u64 << 32), 0);
        assert_eq!(meta.local_key((1u64 << 32) + 100), 100);
    }

    #[test]
    fn test_local_key_u128_no_offset() {
        let meta = SegmentMeta::new(0, 1, 0, 0);

        assert_eq!(meta.local_key(0u128), 0);
        assert_eq!(meta.local_key(100u128), 100);

        // Keys in second arena
        assert_eq!(meta.local_key(1u128 << 32), 0);
        assert_eq!(meta.local_key((1u128 << 32) + 100), 100);
    }

    #[test]
    fn test_local_key_with_offset() {
        let meta = SegmentMeta::new(10, 5, 1000, 0);

        // u64: Key 1000 → relative_key 0 → local_key 0
        assert_eq!(meta.local_key(1000u64), 0);

        // u64: Key 1100 → relative_key 100 → local_key 100
        assert_eq!(meta.local_key(1100u64), 100);

        // u64: Key 1000 + 2^32 → relative_key 2^32 → local_key 0
        assert_eq!(meta.local_key(1000u64 + (1u64 << 32)), 0);

        // u128: Key 1000 → relative_key 0 → local_key 0
        assert_eq!(meta.local_key(1000u128), 0);
    }

    #[test]
    fn test_default() {
        let meta = SegmentMeta::default();

        assert_eq!(meta.cache_key, 0);
        assert_eq!(meta.run_length, 1);
        assert_eq!(meta.key_offset, 0);
        assert_eq!(meta.numa_node, 0);
    }

    #[test]
    fn test_segment_meta_size() {
        use core::mem::size_of;

        // Should be reasonable (25 bytes + padding)
        assert!(size_of::<SegmentMeta>() <= 32);
    }
}

