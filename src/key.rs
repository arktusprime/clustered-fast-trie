//! Trie key trait and implementations for unsigned integer types.

/// Trait for trie key types (u32, u64, u128).
/// All methods are compile-time optimized via monomorphization.
pub trait TrieKey: Copy + Ord + Eq {
    /// Number of levels in trie (bytes in key)
    const LEVELS: usize;

    /// Number of internal levels (LEVELS - 1, leaf is last level)
    const INTERNAL_LEVELS: usize = Self::LEVELS - 1;

    /// Extract byte at given level (0 = most significant byte)
    /// Level must be < LEVELS
    fn byte_at(self, level: usize) -> u8;

    /// Get prefix (upper bits, all except last byte) for leaf lookup
    fn prefix(self) -> Self;

    /// Get suffix (last byte) as bit index in leaf
    fn suffix(self) -> u8;
}

impl TrieKey for u32 {
    const LEVELS: usize = 4;

    #[inline]
    fn byte_at(self, level: usize) -> u8 {
        // Big-endian: level 0 = bits 24-31, level 3 = bits 0-7
        ((self >> (24 - level * 8)) & 0xFF) as u8
    }

    #[inline]
    fn prefix(self) -> Self {
        self & !0xFF
    }

    #[inline]
    fn suffix(self) -> u8 {
        (self & 0xFF) as u8
    }
}

impl TrieKey for u64 {
    const LEVELS: usize = 8;

    #[inline]
    fn byte_at(self, level: usize) -> u8 {
        // Big-endian: level 0 = bits 56-63, level 7 = bits 0-7
        ((self >> (56 - level * 8)) & 0xFF) as u8
    }

    #[inline]
    fn prefix(self) -> Self {
        self & !0xFF
    }

    #[inline]
    fn suffix(self) -> u8 {
        (self & 0xFF) as u8
    }
}

impl TrieKey for u128 {
    const LEVELS: usize = 16;

    #[inline]
    fn byte_at(self, level: usize) -> u8 {
        // Big-endian: level 0 = bits 120-127, level 15 = bits 0-7
        ((self >> (120 - level * 8)) & 0xFF) as u8
    }

    #[inline]
    fn prefix(self) -> Self {
        self & !0xFF
    }

    #[inline]
    fn suffix(self) -> u8 {
        (self & 0xFF) as u8
    }
}
