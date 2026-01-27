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
}

impl TrieKey for u32 {
    const LEVELS: usize = 3;

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
}

impl TrieKey for u64 {
    const LEVELS: usize = 7;

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
}

impl TrieKey for u128 {
    const LEVELS: usize = 15;

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
}
