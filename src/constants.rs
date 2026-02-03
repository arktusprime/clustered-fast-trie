//! Core constants and type definitions for clustered-fast-trie.
#![allow(dead_code)]

/// Sentinel value for empty/null arena index.
///
/// Used to indicate:
/// - Empty child slot in Node.children
/// - End of linked list in Leaf.next/prev
/// - Empty root/first_leaf/last_leaf in Trie
/// - Empty cache in hot path optimization
pub const EMPTY: u32 = u32::MAX;

/// Number of u64 words in a 256-bit bitmap (256 / 64 = 4)
pub const BITMAP_WORDS: usize = 4;

/// Number of children per internal node (2^8 = 256)
pub const NODE_CHILDREN: usize = 256;

/// Number of bits per level (byte-indexed)
pub const BITS_PER_LEVEL: usize = 8;

/// Number of internal levels for u32 (4 bytes - 1)
pub const U32_LEVELS: usize = 3;

/// Number of internal levels for u64 (8 bytes - 1)
pub const U64_LEVELS: usize = 7;

/// Number of internal levels for u128 (16 bytes - 1)
pub const U128_LEVELS: usize = 15;
