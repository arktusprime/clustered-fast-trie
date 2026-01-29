//! Trait for trie key types (u32, u64, u128).

/// Trait for unsigned integer keys used in the trie.
///
/// Supports u32, u64, and u128 with zero-cost abstraction.
/// Each type uses native register sizes for optimal performance.
pub trait TrieKey: Copy + Eq + PartialOrd + Sized {
    /// Number of internal levels (excludes leaf level).
    ///
    /// - u32: 3 levels (4 bytes, last byte in leaf)
    /// - u64: 7 levels (8 bytes, last byte in leaf)
    /// - u128: 15 levels (16 bytes, last byte in leaf)
    const LEVELS: usize;

    /// Arena split levels for hierarchical arena allocation.
    ///
    /// Defines at which levels to create separate arenas for better cache locality.
    /// Nodes at these levels store child_arena_idx for their subtrees.
    ///
    /// - u32: &[] (no splits, single arena)
    /// - u64: &[4] (split at level 4: levels 0-3 in root, 4-7 in child arenas)
    /// - u128: &[4, 12] (splits at levels 4 and 12: three-level hierarchy)
    ///
    /// # Rationale
    /// - Root arena (levels 0-3): always hot in cache, ~256-65K nodes
    /// - Child arenas: created lazily per key prefix, sparse allocation
    /// - Each split reduces arena size by factor of 256^4 = 4B
    const SPLIT_LEVELS: &'static [usize];

    /// Extract byte at given level using big-endian ordering.
    ///
    /// # Arguments
    /// * `level` - Level index (0 = most significant byte)
    ///
    /// # Returns
    /// Byte value at the specified level (0-255)
    ///
    /// # Performance
    /// O(1) - single shift and mask operation in native register size
    fn byte_at(self, level: usize) -> u8;

    /// Get key prefix (all bytes except last).
    ///
    /// Used to identify which leaf a key belongs to.
    ///
    /// # Returns
    /// Key with last byte cleared (masked to 0)
    ///
    /// # Performance
    /// O(1) - single bitwise AND operation
    fn prefix(self) -> Self;

    /// Get last byte of key.
    ///
    /// Used as bit index within a leaf (0-255).
    ///
    /// # Returns
    /// Least significant byte of the key
    ///
    /// # Performance
    /// O(1) - single cast operation
    fn last_byte(self) -> u8;

    /// Convert key to u128 for arithmetic operations.
    ///
    /// Used for key transposition and arena index calculations.
    ///
    /// # Returns
    /// Key value as u128
    ///
    /// # Performance
    /// O(1) - zero-cost for u128, single cast for u32/u64
    fn to_u128(self) -> u128;

    /// Convert u128 back to key type.
    ///
    /// Used for reconstructing keys from u128 representation.
    /// Truncates to appropriate size for u32/u64.
    ///
    /// # Arguments
    /// * `value` - u128 value to convert
    ///
    /// # Returns
    /// Key value of type Self
    ///
    /// # Performance
    /// O(1) - zero-cost for u128, single cast for u32/u64
    fn from_u128(value: u128) -> Self;

    /// Get maximum value for this key type.
    ///
    /// Used for key range calculations and segment sizing.
    ///
    /// # Returns
    /// Maximum possible value for this key type
    ///
    /// # Performance
    /// O(1) - compile-time constant
    fn max_value() -> u128;

    /// Get bit shift for arena index calculation.
    ///
    /// Determines how many bits to shift right to get arena offset.
    /// - u32/u64: 32 bits (arena covers 2^32 keys)
    /// - u128: 64 bits (arena covers 2^64 keys)
    ///
    /// # Returns
    /// Number of bits to shift for arena calculation
    ///
    /// # Performance
    /// O(1) - compile-time constant
    fn arena_shift() -> u32;

    /// Calculate arena index from key.
    ///
    /// Determines which arena this key belongs to based on key prefix.
    /// - u32: always 0 (single arena per segment)
    /// - u64: upper 4 bytes (2^32 possible arenas)
    /// - u128: upper 8 bytes (2^64 possible arenas)
    ///
    /// # Returns
    /// Arena index (u64) for sparse arena allocation
    ///
    /// # Performance
    /// O(1) - single shift operation
    fn arena_idx(self) -> u64;

    /// Calculate arena index for key at specific trie level.
    ///
    /// Used for hierarchical arena allocation with split levels.
    /// Returns arena index based on key prefix up to the given level.
    ///
    /// # Arguments
    /// * `level` - Trie level (0 to LEVELS-1)
    ///
    /// # Returns
    /// Arena index for nodes at this level
    ///
    /// # Behavior by Level
    /// - Before first split: returns 0 (root arena)
    /// - At/after split: returns arena_idx based on key prefix
    ///
    /// # Examples
    /// ```text
    /// u64 key = 0x1234567890ABCDEF, SPLIT_LEVELS = [4]
    ///
    /// level 0-3: arena_idx_at_level(key, 0-3) = 0 (root arena)
    /// level 4-7: arena_idx_at_level(key, 4-7) = 0x12345678 (upper 4 bytes)
    /// ```
    ///
    /// # Performance
    /// O(1) - compile-time specialized per key type, always inlined
    fn arena_idx_at_level(self, level: usize) -> u64;
}

impl TrieKey for u32 {
    const LEVELS: usize = 3;
    const SPLIT_LEVELS: &'static [usize] = &[];

    #[inline(always)]
    fn byte_at(self, level: usize) -> u8 {
        debug_assert!(level < Self::LEVELS, "level out of bounds");
        let shift = (Self::LEVELS - level) * 8;
        (self >> shift) as u8
    }

    #[inline(always)]
    fn prefix(self) -> Self {
        self & !0xFF
    }

    #[inline(always)]
    fn last_byte(self) -> u8 {
        self as u8
    }

    #[inline(always)]
    fn to_u128(self) -> u128 {
        self as u128
    }

    #[inline(always)]
    fn from_u128(value: u128) -> Self {
        value as u32
    }

    #[inline(always)]
    fn max_value() -> u128 {
        u32::MAX as u128
    }

    #[inline(always)]
    fn arena_shift() -> u32 {
        32
    }

    #[inline(always)]
    fn arena_idx(self) -> u64 {
        // u32: single arena per segment
        0
    }

    #[inline(always)]
    fn arena_idx_at_level(self, _level: usize) -> u64 {
        // u32: no splits, always root arena
        0
    }
}

impl TrieKey for u64 {
    const LEVELS: usize = 7;
    const SPLIT_LEVELS: &'static [usize] = &[4];

    #[inline(always)]
    fn byte_at(self, level: usize) -> u8 {
        debug_assert!(level < Self::LEVELS, "level out of bounds");
        let shift = (Self::LEVELS - level) * 8;
        (self >> shift) as u8
    }

    #[inline(always)]
    fn prefix(self) -> Self {
        self & !0xFF
    }

    #[inline(always)]
    fn last_byte(self) -> u8 {
        self as u8
    }

    #[inline(always)]
    fn to_u128(self) -> u128 {
        self as u128
    }

    #[inline(always)]
    fn from_u128(value: u128) -> Self {
        value as u64
    }

    #[inline(always)]
    fn max_value() -> u128 {
        u64::MAX as u128
    }

    #[inline(always)]
    fn arena_shift() -> u32 {
        32
    }

    #[inline(always)]
    fn arena_idx(self) -> u64 {
        // u64: upper 4 bytes as arena index
        (self >> 32) as u64
    }

    #[inline(always)]
    fn arena_idx_at_level(self, level: usize) -> u64 {
        // u64: split at level 4
        // Levels 0-3: root arena (0)
        // Levels 4-7: child arena (upper 4 bytes)
        if level < 4 {
            0
        } else {
            self >> 32
        }
    }
}

impl TrieKey for u128 {
    const LEVELS: usize = 15;
    const SPLIT_LEVELS: &'static [usize] = &[4, 12];

    #[inline(always)]
    fn byte_at(self, level: usize) -> u8 {
        debug_assert!(level < Self::LEVELS, "level out of bounds");
        let shift = (Self::LEVELS - level) * 8;
        (self >> shift) as u8
    }

    #[inline(always)]
    fn prefix(self) -> Self {
        self & !0xFF
    }

    #[inline(always)]
    fn last_byte(self) -> u8 {
        self as u8
    }

    #[inline(always)]
    fn to_u128(self) -> u128 {
        self
    }

    #[inline(always)]
    fn from_u128(value: u128) -> Self {
        value
    }

    #[inline(always)]
    fn max_value() -> u128 {
        u128::MAX
    }

    #[inline(always)]
    fn arena_shift() -> u32 {
        64
    }

    #[inline(always)]
    fn arena_idx(self) -> u64 {
        // u128: upper 8 bytes as arena index
        (self >> 64) as u64
    }

    #[inline(always)]
    fn arena_idx_at_level(self, level: usize) -> u64 {
        // u128: splits at levels 4 and 12
        // Levels 0-3: root arena (0)
        // Levels 4-11: L1 child arena (bytes 4-7 of key)
        // Levels 12-15: L2 child arena (bytes 0-11 of key)
        if level < 4 {
            0
        } else if level < 12 {
            // Extract bytes 4-7 (middle 4 bytes)
            // Key: [bytes 0-3][bytes 4-7][bytes 8-11][bytes 12-15]
            // Shift right by 64 bits to get upper half, then take lower 32 bits
            ((self >> 64) & 0xFFFFFFFF) as u64
        } else {
            // Extract bytes 0-11 (upper 12 bytes)
            // This is: (bytes 0-7 << 32) | (bytes 8-11)
            let upper = (self >> 64) as u64; // bytes 0-7
            let mid = ((self >> 32) & 0xFFFFFFFF) as u64; // bytes 8-11
            (upper << 32) | mid
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_u32_byte_at() {
        let key: u32 = 0x12345678;

        assert_eq!(key.byte_at(0), 0x12);
        assert_eq!(key.byte_at(1), 0x34);
        assert_eq!(key.byte_at(2), 0x56);
    }

    #[test]
    fn test_u32_prefix() {
        let key: u32 = 0x12345678;
        assert_eq!(key.prefix(), 0x12345600);
    }

    #[test]
    fn test_u32_last_byte() {
        let key: u32 = 0x12345678;
        assert_eq!(key.last_byte(), 0x78);
    }

    #[test]
    fn test_u64_byte_at() {
        let key: u64 = 0x123456789ABCDEF0;

        assert_eq!(key.byte_at(0), 0x12);
        assert_eq!(key.byte_at(1), 0x34);
        assert_eq!(key.byte_at(2), 0x56);
        assert_eq!(key.byte_at(3), 0x78);
        assert_eq!(key.byte_at(4), 0x9A);
        assert_eq!(key.byte_at(5), 0xBC);
        assert_eq!(key.byte_at(6), 0xDE);
    }

    #[test]
    fn test_u64_prefix() {
        let key: u64 = 0x123456789ABCDEF0;
        assert_eq!(key.prefix(), 0x123456789ABCDE00);
    }

    #[test]
    fn test_u64_last_byte() {
        let key: u64 = 0x123456789ABCDEF0;
        assert_eq!(key.last_byte(), 0xF0);
    }

    #[test]
    fn test_u128_byte_at() {
        let key: u128 = 0x0102030405060708090A0B0C0D0E0F10;

        assert_eq!(key.byte_at(0), 0x01);
        assert_eq!(key.byte_at(7), 0x08);
        assert_eq!(key.byte_at(14), 0x0F);
    }

    #[test]
    fn test_u128_prefix() {
        let key: u128 = 0x0102030405060708090A0B0C0D0E0F10;
        assert_eq!(key.prefix(), 0x0102030405060708090A0B0C0D0E0F00);
    }

    #[test]
    fn test_u128_last_byte() {
        let key: u128 = 0x0102030405060708090A0B0C0D0E0F10;
        assert_eq!(key.last_byte(), 0x10);
    }

    #[test]
    fn test_levels_constants() {
        assert_eq!(u32::LEVELS, 3);
        assert_eq!(u64::LEVELS, 7);
        assert_eq!(u128::LEVELS, 15);
    }

    #[test]
    fn test_u32_arena_idx() {
        // u32: always returns 0
        assert_eq!(0u32.arena_idx(), 0);
        assert_eq!(u32::MAX.arena_idx(), 0);
        assert_eq!(0x12345678u32.arena_idx(), 0);
    }

    #[test]
    fn test_u64_arena_idx() {
        // u64: upper 4 bytes
        assert_eq!(0x0000000000000000u64.arena_idx(), 0x00000000);
        assert_eq!(0x0000000100000000u64.arena_idx(), 0x00000001);
        assert_eq!(0x123456789ABCDEFu64.arena_idx(), 0x01234567);
        assert_eq!(0xFFFFFFFF00000000u64.arena_idx(), 0xFFFFFFFF);
    }

    #[test]
    fn test_u128_arena_idx() {
        // u128: upper 8 bytes
        assert_eq!(
            0x00000000000000000000000000000000u128.arena_idx(),
            0x0000000000000000
        );
        assert_eq!(
            0x00000000000000010000000000000000u128.arena_idx(),
            0x0000000000000001
        );
        assert_eq!(
            0x0102030405060708090A0B0C0D0E0F10u128.arena_idx(),
            0x0102030405060708
        );
        assert_eq!(
            0xFFFFFFFFFFFFFFFF0000000000000000u128.arena_idx(),
            0xFFFFFFFFFFFFFFFF
        );
    }

    #[test]
    fn test_u32_arena_idx_at_level() {
        let key = 0x12345678u32;
        // u32: no splits, always 0
        assert_eq!(key.arena_idx_at_level(0), 0);
        assert_eq!(key.arena_idx_at_level(1), 0);
        assert_eq!(key.arena_idx_at_level(2), 0);
    }

    #[test]
    fn test_u64_arena_idx_at_level() {
        let key = 0x123456789ABCDEFu64;

        // Levels 0-3: root arena
        assert_eq!(key.arena_idx_at_level(0), 0);
        assert_eq!(key.arena_idx_at_level(1), 0);
        assert_eq!(key.arena_idx_at_level(2), 0);
        assert_eq!(key.arena_idx_at_level(3), 0);

        // Levels 4-7: child arena (upper 4 bytes)
        assert_eq!(key.arena_idx_at_level(4), 0x01234567);
        assert_eq!(key.arena_idx_at_level(5), 0x01234567);
        assert_eq!(key.arena_idx_at_level(6), 0x01234567);
    }

    #[test]
    fn test_u128_arena_idx_at_level() {
        let key = 0x0102030405060708090A0B0C0D0E0F10u128;

        // Levels 0-3: root arena
        assert_eq!(key.arena_idx_at_level(0), 0);
        assert_eq!(key.arena_idx_at_level(1), 0);
        assert_eq!(key.arena_idx_at_level(2), 0);
        assert_eq!(key.arena_idx_at_level(3), 0);

        // Levels 4-11: L1 child arena (bytes 4-7 of key)
        // Key bytes: 01 02 03 04 | 05 06 07 08 | 09 0A 0B 0C | 0D 0E 0F 10
        //            [0-3]        [4-7]        [8-11]       [12-15]
        // Bytes 4-7 = 0x05060708
        assert_eq!(key.arena_idx_at_level(4), 0x05060708);
        assert_eq!(key.arena_idx_at_level(5), 0x05060708);
        assert_eq!(key.arena_idx_at_level(11), 0x05060708);

        // Levels 12-15: L2 child arena (bytes 0-11)
        // Bytes 0-11 = 0x010203040506070809 0A0B0C
        let expected = (0x0102030405060708u64 << 32) | 0x090A0B0Cu64;
        assert_eq!(key.arena_idx_at_level(12), expected);
        assert_eq!(key.arena_idx_at_level(13), expected);
        assert_eq!(key.arena_idx_at_level(14), expected);
    }
}
