//! Segment metadata and management

use crate::key::TrieKey;

/// Segment identifier (permanent key).
///
/// This is the stable identifier given to clients.
/// It never changes even during defragmentation.
pub type SegmentId = u32;

/// Segment metadata structure.
///
/// Maps segment ID to key range with prefix-based partitioning.
/// All segments share the same hierarchical arena structure.
///
/// # Architecture
/// - Single-tenant: 1 segment covering entire key space
/// - Multi-tenant: N segments with prefix-based partitioning (segment_id = key >> segment_shift)
///
/// # Memory Layout
/// - `cache_key`: 8 bytes - root arena index in sparse/dense mapping
/// - `key_offset`: 16 bytes - transposition offset for normalized keys
/// - `numa_node`: 1 byte - NUMA node for locality
/// - Total: 25 bytes (padded to 32 bytes by compiler)
///
/// # Key Transposition
/// - Client keys: arbitrary range [start, start + size)
/// - Normalized keys: normalized_key = client_key - key_offset
/// - Shared hierarchy: all segments use same arena structure
///
/// # Prefix-based Partitioning (Multi-tenant)
/// - segment_id = key >> segment_shift (O(1) lookup)
/// - Fixed segment sizes = guaranteed performance per tenant
/// - Clustered keys stay in same segment (locality preserved)
///
/// # Hierarchical Arenas
/// - Root arena: cache_key (unique per segment in multi-tenant)
/// - Child arenas: calculated via TrieKey::arena_idx_at_level() from normalized key
/// - Shared structure: no duplication, efficient memory usage
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SegmentMeta {
    /// Root arena index in sparse/dense mapping.
    ///
    /// - Single-tenant: always 0 (one root arena)
    /// - Multi-tenant: unique per segment for isolation
    ///
    /// Child arenas are calculated from normalized keys,
    /// not stored in SegmentMeta (shared hierarchy).
    pub cache_key: u64,

    /// Transposition offset for key normalization.
    ///
    /// Converts client keys to normalized keys:
    /// `normalized_key = client_key - key_offset`
    ///
    /// - Single-tenant: 0 (no transposition)
    /// - Multi-tenant: start of segment's key range
    ///
    /// Uses u128 to support all key types (u32, u64, u128).
    pub key_offset: u128,

    /// NUMA node for memory allocation.
    ///
    /// All arenas accessed by this segment allocated on same NUMA node
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
    /// * `cache_key` - Root arena index in sparse/dense mapping
    /// * `key_offset` - Transposition offset for key normalization
    /// * `numa_node` - NUMA node for allocation
    ///
    /// # Returns
    /// New segment metadata
    ///
    /// # Performance
    /// O(1) - simple struct initialization
    #[inline(always)]
    pub fn new(cache_key: u64, key_offset: u128, numa_node: u8) -> Self {
        SegmentMeta {
            cache_key,
            key_offset,
            numa_node,
        }
    }

    /// Get root arena index for this segment.
    ///
    /// Root arena is used for trie levels 0 to first split level.
    ///
    /// # Returns
    /// Root arena index (cache_key)
    ///
    /// # Performance
    /// O(1) - direct field access
    #[inline(always)]
    pub fn root_arena(&self) -> u64 {
        self.cache_key
    }

    /// Normalize client key to internal key.
    ///
    /// Applies key transposition to convert arbitrary client key ranges
    /// to normalized keys starting from 0.
    ///
    /// # Arguments
    /// * `key` - Client key (u32, u64, or u128)
    ///
    /// # Returns
    /// Normalized key for internal trie operations
    ///
    /// # Performance
    /// O(1) - single subtraction
    ///
    /// # Examples
    /// ```text
    /// Single-tenant: key_offset = 0
    ///   client_key = 12345 → normalized_key = 12345
    ///
    /// Multi-tenant: key_offset = 1000000
    ///   client_key = 1012345 → normalized_key = 12345
    /// ```
    #[inline(always)]
    pub fn normalize_key<K: TrieKey>(&self, key: K) -> K {
        let normalized = key.to_u128().wrapping_sub(self.key_offset);
        K::from_u128(normalized)
    }

    /// Get arena index for a key at specific trie level.
    ///
    /// Uses hierarchical arena calculation with key normalization.
    /// Root levels use cache_key, child levels use key-based calculation.
    ///
    /// # Arguments
    /// * `key` - Client key (u32, u64, or u128)
    /// * `level` - Trie level (0 to K::LEVELS-1)
    ///
    /// # Returns
    /// Arena index for the specified level
    ///
    /// # Performance
    /// O(1) - arithmetic operations only
    ///
    /// # Behavior
    /// - Root levels (0 to first split): returns cache_key
    /// - Child levels (after split): calculates from normalized key prefix
    ///
    /// # Examples
    /// ```text
    /// u64 with SPLIT_LEVELS = [4]:
    ///   level 0-3: cache_key (root arena)
    ///   level 4-7: arena_idx_at_level(normalized_key, level)
    /// ```
    #[inline(always)]
    pub fn arena_at_level<K: TrieKey>(&self, key: K, level: usize) -> u64 {
        // Normalize key first
        let normalized_key = self.normalize_key(key);

        // Root levels: use cache_key
        if K::SPLIT_LEVELS.is_empty() || level < K::SPLIT_LEVELS[0] {
            return self.cache_key;
        }

        // Child levels: calculate from normalized key
        normalized_key.arena_idx_at_level(level)
    }
}

impl Default for SegmentMeta {
    fn default() -> Self {
        Self::new(0, 0, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_segment_meta() {
        let meta = SegmentMeta::new(100, 1000, 0);

        assert_eq!(meta.cache_key, 100);
        assert_eq!(meta.key_offset, 1000);
        assert_eq!(meta.numa_node, 0);
    }

    #[test]
    fn test_root_arena() {
        let meta = SegmentMeta::new(42, 0, 0);
        assert_eq!(meta.root_arena(), 42);
    }

    #[test]
    fn test_normalize_key_no_offset() {
        let meta = SegmentMeta::new(0, 0, 0);

        // No transposition
        assert_eq!(meta.normalize_key(0u32), 0u32);
        assert_eq!(meta.normalize_key(12345u32), 12345u32);
        assert_eq!(meta.normalize_key(12345u64), 12345u64);
        assert_eq!(meta.normalize_key(12345u128), 12345u128);
    }

    #[test]
    fn test_normalize_key_with_offset() {
        let meta = SegmentMeta::new(0, 1000, 0);

        // With transposition
        assert_eq!(meta.normalize_key(1000u64), 0u64);
        assert_eq!(meta.normalize_key(1100u64), 100u64);
        assert_eq!(meta.normalize_key(1000u128), 0u128);
        assert_eq!(meta.normalize_key(1100u128), 100u128);
    }

    #[test]
    fn test_arena_at_level_u32() {
        let meta = SegmentMeta::new(42, 0, 0);

        // u32: no split levels, always root arena
        assert_eq!(meta.arena_at_level(0u32, 0), 42);
        assert_eq!(meta.arena_at_level(12345u32, 0), 42);
        assert_eq!(meta.arena_at_level(12345u32, 1), 42);
        assert_eq!(meta.arena_at_level(12345u32, 2), 42);
    }

    #[test]
    fn test_arena_at_level_u64_no_offset() {
        let meta = SegmentMeta::new(0, 0, 0);
        let key = 0x123456789ABCDEFu64;

        // Levels 0-3: root arena
        assert_eq!(meta.arena_at_level(key, 0), 0);
        assert_eq!(meta.arena_at_level(key, 1), 0);
        assert_eq!(meta.arena_at_level(key, 2), 0);
        assert_eq!(meta.arena_at_level(key, 3), 0);

        // Levels 4-7: child arena (upper 4 bytes)
        assert_eq!(meta.arena_at_level(key, 4), 0x01234567);
        assert_eq!(meta.arena_at_level(key, 5), 0x01234567);
        assert_eq!(meta.arena_at_level(key, 6), 0x01234567);
    }

    #[test]
    fn test_arena_at_level_u64_with_offset() {
        let meta = SegmentMeta::new(100, 0x1000000000000000, 0);
        let key = 0x1000000012345678u64;

        // After normalization: 0x12345678
        // Levels 0-3: root arena
        assert_eq!(meta.arena_at_level(key, 0), 100);
        assert_eq!(meta.arena_at_level(key, 3), 100);

        // Levels 4-7: child arena from normalized key (0x00000000)
        assert_eq!(meta.arena_at_level(key, 4), 0);
    }

    #[test]
    fn test_arena_at_level_u128_no_offset() {
        let meta = SegmentMeta::new(0, 0, 0);
        let key = 0x0102030405060708090A0B0C0D0E0F10u128;

        // Levels 0-3: root arena
        assert_eq!(meta.arena_at_level(key, 0), 0);
        assert_eq!(meta.arena_at_level(key, 3), 0);

        // Levels 4-11: L1 child arena (bytes 4-7)
        let expected_l1 = 0x05060708u64;
        assert_eq!(meta.arena_at_level(key, 4), expected_l1);
        assert_eq!(meta.arena_at_level(key, 11), expected_l1);

        // Levels 12-15: L2 child arena (bytes 0-11)
        let expected_l2 = (0x0102030405060708u64 << 32) | 0x090A0B0Cu64;
        assert_eq!(meta.arena_at_level(key, 12), expected_l2);
        assert_eq!(meta.arena_at_level(key, 14), expected_l2);
    }

    #[test]
    fn test_default() {
        let meta = SegmentMeta::default();

        assert_eq!(meta.cache_key, 0);
        assert_eq!(meta.key_offset, 0);
        assert_eq!(meta.numa_node, 0);
    }

    #[test]
    fn test_segment_meta_size() {
        use core::mem::size_of;

        // Should be compact: 8 + 16 + 1 = 25 bytes (padded to 32)
        assert!(size_of::<SegmentMeta>() <= 32);
    }
}
