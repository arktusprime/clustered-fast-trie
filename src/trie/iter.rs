//! Iterator support for Trie traversal.
//!
//! Provides efficient iteration over keys using the linked list of leaves.
//! - O(1) per element for full iteration
//! - O(log log U) initial setup for range queries

use crate::key::TrieKey;
use crate::trie::{unpack_link, Trie, EMPTY_LINK};
use core::ops::{Bound, RangeBounds};

/// Iterator over keys in ascending order.
///
/// Iterates through all keys in the trie by traversing the linked list of leaves.
/// Each leaf contains up to 256 keys stored as bits in a bitmap.
///
/// # Algorithm
/// 1. Start at first leaf (cached in Trie::first_leaf)
/// 2. For each leaf:
///    - Iterate through set bits in bitmap (keys 0-255)
///    - Move to next leaf via leaf.next pointer
/// 3. Stop when next == EMPTY_LINK
///
/// # Performance
/// - O(1) per element amortized (direct leaf traversal)
/// - No trie structure traversal needed
/// - Memory: ~32 bytes (leaf link, bit index, phantom data)
///
/// # Example
/// ```rust
/// use clustered_fast_trie::Trie;
///
/// let mut trie = Trie::<u64>::new();
/// trie.insert(10);
/// trie.insert(20);
/// trie.insert(30);
///
/// let keys: Vec<u64> = trie.iter().collect();
/// assert_eq!(keys, vec![10, 20, 30]);
/// ```
#[derive(Debug)]
pub struct Iter<'a, K: TrieKey> {
    /// Reference to the trie being iterated
    trie: &'a Trie<K>,

    /// Current leaf being processed (packed: arena_idx << 32 | leaf_idx)
    current_leaf: u64,

    /// Current bit index within leaf bitmap (0-255)
    current_bit: u8,

    /// Phantom data for key type
    _phantom: core::marker::PhantomData<K>,
}

impl<'a, K: TrieKey> Iter<'a, K> {
    /// Create a new iterator starting from the first leaf.
    ///
    /// # Arguments
    /// * `trie` - Reference to the trie to iterate over
    ///
    /// # Returns
    /// Iterator positioned at the first key, or empty if trie is empty
    ///
    /// # Performance
    /// O(1) - uses cached first_leaf
    pub(crate) fn new(trie: &'a Trie<K>) -> Self {
        Iter {
            trie,
            current_leaf: trie.first_leaf_link(),
            current_bit: 0,
            _phantom: core::marker::PhantomData,
        }
    }

    /// Advance to next set bit in current leaf or next leaf.
    ///
    /// # Returns
    /// Next key if found, None if end of iteration
    ///
    /// # Performance
    /// O(1) amortized - bitmap scan is fast, leaf traversal is O(1)
    fn advance(&mut self) -> Option<K> {
        // Check if we've reached the end
        if self.current_leaf == EMPTY_LINK {
            return None;
        }

        // Get current leaf
        let (arena_idx, leaf_idx) = unpack_link(self.current_leaf);

        // Check if arena exists
        if arena_idx as usize >= self.trie.arenas_len() {
            self.current_leaf = EMPTY_LINK;
            return None;
        }

        let leaf_arena = self.trie.get_leaf_arena(arena_idx as u32);

        // Bounds check leaf_idx
        if leaf_idx as usize >= leaf_arena.len() {
            self.current_leaf = EMPTY_LINK;
            return None;
        }

        let leaf = leaf_arena.get(leaf_idx);

        // Find next set bit in current leaf
        // Special case: if current_bit == 0, find first set bit (don't skip bit 0)
        let bit_opt = if self.current_bit == 0 {
            use crate::bitmap::min_bit;
            min_bit(&leaf.bitmap)
        } else {
            use crate::bitmap::next_set_bit;
            // next_set_bit finds bits AFTER current_bit-1, so we pass current_bit-1
            next_set_bit(&leaf.bitmap, self.current_bit - 1)
        };

        if let Some(bit) = bit_opt {
            // Found next bit in current leaf
            // Handle overflow: if bit == 255, we need to move to next leaf
            if bit == 255 {
                self.current_leaf = leaf.next;
                self.current_bit = 0;
            } else {
                self.current_bit = bit + 1; // Advance for next call
            }

            // Reconstruct key from leaf prefix and bit
            // prefix already includes position for last byte, just OR the bit
            let key_value = leaf.prefix.to_u128() | (bit as u128);
            return Some(K::from_u128(key_value));
        }

        // No more bits in current leaf - move to next leaf
        self.current_leaf = leaf.next;
        self.current_bit = 0;

        // Recursively try next leaf
        self.advance()
    }
}

impl<'a, K: TrieKey> Iterator for Iter<'a, K> {
    type Item = K;

    fn next(&mut self) -> Option<Self::Item> {
        self.advance()
    }
}

/// Range iterator over keys within a specified range.
///
/// Iterates through keys in [start, end) by:
/// 1. Finding the first key >= start using successor
/// 2. Iterating through linked list until key >= end
///
/// # Performance
/// - O(log log U) initial setup to find start
/// - O(1) per element during iteration
///
/// # Example
/// ```rust
/// use clustered_fast_trie::Trie;
///
/// let mut trie = Trie::<u64>::new();
/// for i in 0..100 {
///     trie.insert(i);
/// }
///
/// let keys: Vec<u64> = trie.range(10..20).collect();
/// assert_eq!(keys.len(), 10);
/// assert_eq!(keys[0], 10);
/// assert_eq!(keys[9], 19);
/// ```
#[derive(Debug)]
pub struct RangeIter<'a, K: TrieKey> {
    /// Reference to the trie being iterated
    trie: &'a Trie<K>,

    /// Current leaf being processed (packed: arena_idx << 32 | leaf_idx)
    current_leaf: u64,

    /// Current bit index within leaf bitmap (0-255)
    current_bit: u8,

    /// End bound (exclusive)
    end: Bound<K>,

    /// Phantom data for key type
    _phantom: core::marker::PhantomData<K>,
}

impl<'a, K: TrieKey> RangeIter<'a, K> {
    /// Create a new range iterator.
    ///
    /// # Arguments
    /// * `trie` - Reference to the trie to iterate over
    /// * `range` - Range bounds (start..end, start..=end, ..end, etc.)
    ///
    /// # Returns
    /// Iterator positioned at first key in range
    ///
    /// # Performance
    /// O(log log U) - uses successor to find start position
    pub(crate) fn new<R>(trie: &'a Trie<K>, range: R) -> Self
    where
        R: RangeBounds<K>,
    {
        use Bound::*;

        // Determine start key
        let start_key = match range.start_bound() {
            Included(&key) => Some(key),
            Excluded(&key) => {
                // Find successor of excluded start
                trie.successor(key)
            }
            Unbounded => trie.min(), // Start from minimum
        };

        // Find leaf and bit for start key
        let (current_leaf, current_bit) = if let Some(key) = start_key {
            // Find the leaf containing this key or its successor
            Self::find_leaf_for_key(trie, key)
        } else {
            // No valid start - empty iterator
            (EMPTY_LINK, 0)
        };

        RangeIter {
            trie,
            current_leaf,
            current_bit,
            end: range.end_bound().cloned(),
            _phantom: core::marker::PhantomData,
        }
    }

    /// Find the leaf and bit position for a given key or its successor.
    ///
    /// # Arguments
    /// * `trie` - Reference to the trie
    /// * `key` - Key to find position for
    ///
    /// # Returns
    /// (leaf_link, bit_index) tuple
    ///
    /// # Performance
    /// O(log log U) - traverses trie structure once
    fn find_leaf_for_key(trie: &Trie<K>, key: K) -> (u64, u8) {
        use crate::trie::pack_link;

        // Start at root arena
        let mut current_arena_idx = 0u32;
        let mut current_node_idx = 0u32;

        // Traverse trie to find leaf
        for level in 0..(K::LEVELS - 1) {
            let byte = key.byte_at(level);

            // Get current node
            let node_arena = trie.get_node_arena(current_arena_idx);
            if node_arena.is_empty() {
                return (EMPTY_LINK, 0);
            }

            let node = node_arena.get(current_node_idx);

            // Check if child exists
            if !node.has_child(byte) {
                // Path doesn't exist - need to find successor
                return Self::find_successor_leaf(trie, key);
            }

            // Move to child
            current_node_idx = node.get_child(byte);

            // Check if we need to switch arenas at split level
            if K::SPLIT_LEVELS.contains(&(level + 1)) {
                let child_arena_idx = node.child_arena_idx;
                if child_arena_idx as usize >= trie.arenas_len() {
                    return (EMPTY_LINK, 0);
                }
                current_arena_idx = child_arena_idx;
            }
        }

        // Final level: get leaf
        let last_node_byte = key.byte_at(K::LEVELS - 1);
        let node_arena = trie.get_node_arena(current_arena_idx);
        let final_node = node_arena.get(current_node_idx);

        if !final_node.has_child(last_node_byte) {
            // Leaf doesn't exist - find successor
            return Self::find_successor_leaf(trie, key);
        }

        let leaf_idx = final_node.get_child(last_node_byte);
        let leaf_link = pack_link(current_arena_idx as u64, leaf_idx);

        // Get the bit index for this key
        let bit_idx = key.last_byte();

        (leaf_link, bit_idx)
    }

    /// Find the successor leaf for a key that doesn't exist.
    ///
    /// Uses successor operation to find next key, then locates its leaf.
    ///
    /// # Performance
    /// O(log log U) - uses successor operation
    fn find_successor_leaf(trie: &Trie<K>, key: K) -> (u64, u8) {
        if let Some(succ_key) = trie.successor(key) {
            Self::find_leaf_for_key(trie, succ_key)
        } else {
            (EMPTY_LINK, 0)
        }
    }

    /// Advance to next key in range.
    ///
    /// # Returns
    /// Next key if found and within range, None if end reached
    ///
    /// # Performance
    /// O(1) amortized
    fn advance(&mut self) -> Option<K> {
        // Check if we've reached the end
        if self.current_leaf == EMPTY_LINK {
            return None;
        }

        // Get current leaf
        let (arena_idx, leaf_idx) = unpack_link(self.current_leaf);

        // Check if arena exists
        if arena_idx as usize >= self.trie.arenas_len() {
            self.current_leaf = EMPTY_LINK;
            return None;
        }

        let leaf_arena = self.trie.get_leaf_arena(arena_idx as u32);

        // Bounds check leaf_idx
        if leaf_idx as usize >= leaf_arena.len() {
            self.current_leaf = EMPTY_LINK;
            return None;
        }

        let leaf = leaf_arena.get(leaf_idx);

        // Find next set bit in current leaf
        // Special case: if current_bit == 0, find first set bit (don't skip bit 0)
        let bit_opt = if self.current_bit == 0 {
            use crate::bitmap::min_bit;
            min_bit(&leaf.bitmap)
        } else {
            use crate::bitmap::next_set_bit;
            // next_set_bit finds bits AFTER current_bit-1, so we pass current_bit-1
            next_set_bit(&leaf.bitmap, self.current_bit - 1)
        };

        if let Some(bit) = bit_opt {
            // Reconstruct key from leaf prefix and bit
            // prefix already includes position for last byte, just OR the bit
            let key_value = leaf.prefix.to_u128() | (bit as u128);
            let key = K::from_u128(key_value);

            // Check if key is within range
            if self.is_past_end(&key) {
                self.current_leaf = EMPTY_LINK; // Stop iteration
                return None;
            }

            // Found next bit in current leaf - advance for next call
            // Handle overflow: if bit == 255, we need to move to next leaf
            if bit == 255 {
                self.current_leaf = leaf.next;
                self.current_bit = 0;
            } else {
                self.current_bit = bit + 1;
            }

            return Some(key);
        }

        // No more bits in current leaf - move to next leaf
        self.current_leaf = leaf.next;
        self.current_bit = 0;

        // Recursively try next leaf
        self.advance()
    }

    /// Check if key is past the end bound.
    ///
    /// # Arguments
    /// * `key` - Key to check
    ///
    /// # Returns
    /// true if key is past end bound, false otherwise
    fn is_past_end(&self, key: &K) -> bool {
        use Bound::*;

        match self.end {
            Included(ref end_key) => key > end_key,
            Excluded(ref end_key) => key >= end_key,
            Unbounded => false,
        }
    }
}

impl<'a, K: TrieKey> Iterator for RangeIter<'a, K> {
    type Item = K;

    fn next(&mut self) -> Option<Self::Item> {
        self.advance()
    }
}

/// Public API methods for Trie iterators.
impl<K: TrieKey> Trie<K> {
    /// Create an iterator over all keys in ascending order.
    ///
    /// Traverses the linked list of leaves, yielding all keys in sorted order.
    /// Very efficient for clustered data due to O(1) leaf-to-leaf traversal.
    ///
    /// # Performance
    /// - O(1) per element amortized
    /// - O(n) total for n elements
    ///
    /// # Example
    /// ```rust
    /// use clustered_fast_trie::Trie;
    ///
    /// let mut trie = Trie::<u64>::new();
    /// trie.insert(10);
    /// trie.insert(20);
    /// trie.insert(30);
    ///
    /// let keys: Vec<u64> = trie.iter().collect();
    /// assert_eq!(keys, vec![10, 20, 30]);
    /// ```
    #[inline]
    pub fn iter(&self) -> Iter<'_, K> {
        Iter::new(self)
    }

    /// Create an iterator over keys within a specified range.
    ///
    /// Supports all range types:
    /// - `trie.range(10..20)` - half-open range [10, 20)
    /// - `trie.range(10..=20)` - closed range [10, 20]
    /// - `trie.range(..20)` - unbounded start, bounded end
    /// - `trie.range(10..)` - bounded start, unbounded end
    /// - `trie.range(..)` - full range (same as iter())
    ///
    /// # Arguments
    /// * `range` - Range bounds (implements RangeBounds<K>)
    ///
    /// # Returns
    /// Iterator over keys in the specified range
    ///
    /// # Performance
    /// - O(log log U) initial setup to find start
    /// - O(1) per element amortized
    /// - O(k) total for k elements in range
    ///
    /// # Example
    /// ```rust
    /// use clustered_fast_trie::Trie;
    ///
    /// let mut trie = Trie::<u64>::new();
    /// for i in 0..100 {
    ///     trie.insert(i);
    /// }
    ///
    /// // Half-open range
    /// let keys: Vec<u64> = trie.range(10..20).collect();
    /// assert_eq!(keys.len(), 10);
    ///
    /// // Closed range
    /// let keys: Vec<u64> = trie.range(10..=20).collect();
    /// assert_eq!(keys.len(), 11);
    ///
    /// // Unbounded ranges
    /// let keys: Vec<u64> = trie.range(..50).collect();
    /// assert_eq!(keys.len(), 50);
    ///
    /// let keys: Vec<u64> = trie.range(50..).collect();
    /// assert_eq!(keys.len(), 50);
    /// ```
    #[inline]
    pub fn range<R>(&self, range: R) -> RangeIter<'_, K>
    where
        R: RangeBounds<K>,
    {
        RangeIter::new(self, range)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    #[test]
    fn test_simple_two_leaves() {
        let mut trie = Trie::<u32>::new();
        
        // Insert first key
        trie.insert(0);
        assert!(trie.contains(0));
        assert_eq!(trie.get_leaf_arena(0).len(), 1, "Should have 1 leaf after first insert");
        
        // Verify first_leaf is set
        use crate::trie::{unpack_link, EMPTY_LINK};
        assert_ne!(trie.first_leaf_link(), EMPTY_LINK, "first_leaf should be set after first insert");
        
        // Insert second key that should go to different leaf
        // Key 0 = 0x00000000 -> differs at level 2: byte=0x00
        // Key 256 = 0x00000100 -> differs at level 2: byte=0x01
        trie.insert(256);
        
        // Check both keys exist
        assert!(trie.contains(0));
        assert!(trie.contains(256));
        
        // Check total leaves in arena
        let total_leaves = trie.get_leaf_arena(0).len();
        assert_eq!(total_leaves, 2, "Should have exactly 2 leaves, found {}", total_leaves);
        
        // Check linked list
        let mut current_link = trie.first_leaf_link();
        let mut linked_leaf_count = 0;
        let mut prefixes = alloc::vec::Vec::new();
        
        while current_link != EMPTY_LINK && linked_leaf_count < 10 {
            let (arena_idx, leaf_idx) = unpack_link(current_link);
            let leaf_arena = trie.get_leaf_arena(arena_idx as u32);
            let leaf = leaf_arena.get(leaf_idx);
            prefixes.push(leaf.prefix);
            
            current_link = leaf.next;
            linked_leaf_count += 1;
        }
        
        assert_eq!(linked_leaf_count, 2, "Should have 2 linked leaves, found {}", linked_leaf_count);
        assert_eq!(prefixes[0], 0, "First leaf should have prefix 0");
        assert_eq!(prefixes[1], 0x100, "Second leaf should have prefix 0x100 (256)");
    }

    #[test]
    fn test_leaf_linking() {
        let mut trie = Trie::<u32>::new();
        // Insert keys that will span multiple leaves
        // Keys 0-255 go to one leaf, 256-511 to another
        for i in 0..512 {
            trie.insert(i);
        }
        
        // Verify all keys are actually inserted
        assert_eq!(trie.len(), 512, "Expected 512 keys in trie");
        assert!(trie.contains(0), "Key 0 should exist");
        assert!(trie.contains(255), "Key 255 should exist");
        assert!(trie.contains(256), "Key 256 should exist");
        assert!(trie.contains(511), "Key 511 should exist");
        
        // Check first_leaf is set
        assert_ne!(trie.first_leaf_link(), crate::trie::EMPTY_LINK, "first_leaf should be set");
        
        // Check total leaves in arena
        let total_leaves = trie.get_leaf_arena(0).len();
        assert!(total_leaves >= 2, "Should have at least 2 leaves in arena, found {}", total_leaves);
        
        // Try to manually iterate through leaves via linked list
        use crate::trie::{unpack_link, EMPTY_LINK};
        let mut current_link = trie.first_leaf_link();
        let mut linked_leaf_count = 0;
        
        while current_link != EMPTY_LINK && linked_leaf_count < 10 {  // Safety limit
            let (arena_idx, leaf_idx) = unpack_link(current_link);
            let leaf_arena = trie.get_leaf_arena(arena_idx as u32);
            let leaf = leaf_arena.get(leaf_idx);
            
            current_link = leaf.next;
            linked_leaf_count += 1;
        }
        
        assert_eq!(linked_leaf_count, total_leaves, 
            "Linked list should contain all {} leaves, but only found {} linked", 
            total_leaves, linked_leaf_count);
    }

    #[test]
    fn test_iter_empty() {
        let trie = Trie::<u32>::new();
        let keys: Vec<u32> = trie.iter().collect();
        assert_eq!(keys, Vec::<u32>::new());
    }

    #[test]
    fn test_iter_single() {
        let mut trie = Trie::<u32>::new();
        trie.insert(42);

        let keys: Vec<u32> = trie.iter().collect();
        assert_eq!(keys, vec![42]);
    }

    #[test]
    fn test_iter_multiple() {
        let mut trie = Trie::<u32>::new();
        let expected = vec![10, 20, 30, 40, 50];

        for &key in &expected {
            trie.insert(key);
        }

        let keys: Vec<u32> = trie.iter().collect();
        assert_eq!(keys, expected);
    }

    #[test]
    fn test_iter_sorted() {
        let mut trie = Trie::<u64>::new();
        let mut expected = vec![100, 50, 200, 25, 150, 75];

        for &key in &expected {
            trie.insert(key);
        }

        expected.sort();
        let keys: Vec<u64> = trie.iter().collect();
        assert_eq!(keys, expected);
    }

    #[test]
    fn test_iter_sequential_small() {
        let mut trie = Trie::<u32>::new();
        // Test with 300 elements to cover multiple leaves
        // Each leaf holds 256 keys, so 300 will span 2 leaves
        for i in 0..300 {
            trie.insert(i);
        }

        let keys: Vec<u32> = trie.iter().collect();
        assert_eq!(keys.len(), 300, "Expected 300 keys, got {}", keys.len());
        
        for (i, &key) in keys.iter().enumerate() {
            assert_eq!(key, i as u32, "Key mismatch at index {}: expected {}, got {}", i, i, key);
        }
    }

    #[test]
    fn test_iter_sequential() {
        let mut trie = Trie::<u32>::new();
        for i in 0..1000 {
            trie.insert(i);
        }

        let keys: Vec<u32> = trie.iter().collect();
        assert_eq!(keys.len(), 1000);
        for (i, &key) in keys.iter().enumerate() {
            assert_eq!(key, i as u32);
        }
    }

    #[test]
    fn test_range_empty() {
        let trie = Trie::<u32>::new();
        let keys: Vec<u32> = trie.range(10..20).collect();
        assert_eq!(keys, Vec::<u32>::new());
    }

    #[test]
    fn test_range_half_open() {
        let mut trie = Trie::<u64>::new();
        for i in 0..100 {
            trie.insert(i);
        }

        let keys: Vec<u64> = trie.range(10..20).collect();
        assert_eq!(keys.len(), 10);
        assert_eq!(keys[0], 10);
        assert_eq!(keys[9], 19);
    }

    #[test]
    fn test_range_closed() {
        let mut trie = Trie::<u64>::new();
        for i in 0..100 {
            trie.insert(i);
        }

        let keys: Vec<u64> = trie.range(10..=20).collect();
        assert_eq!(keys.len(), 11);
        assert_eq!(keys[0], 10);
        assert_eq!(keys[10], 20);
    }

    #[test]
    fn test_range_unbounded_start() {
        let mut trie = Trie::<u64>::new();
        for i in 0..100 {
            trie.insert(i);
        }

        let keys: Vec<u64> = trie.range(..50).collect();
        assert_eq!(keys.len(), 50);
        assert_eq!(keys[0], 0);
        assert_eq!(keys[49], 49);
    }

    #[test]
    fn test_range_unbounded_end() {
        let mut trie = Trie::<u64>::new();
        for i in 0..100 {
            trie.insert(i);
        }

        let keys: Vec<u64> = trie.range(50..).collect();
        assert_eq!(keys.len(), 50);
        assert_eq!(keys[0], 50);
        assert_eq!(keys[49], 99);
    }

    #[test]
    fn test_range_full() {
        let mut trie = Trie::<u64>::new();
        for i in 0..100 {
            trie.insert(i);
        }

        let keys: Vec<u64> = trie.range(..).collect();
        assert_eq!(keys.len(), 100);
        assert_eq!(keys[0], 0);
        assert_eq!(keys[99], 99);
    }

    #[test]
    fn test_range_sparse() {
        let mut trie = Trie::<u64>::new();
        // Insert only even numbers
        for i in (0..100).step_by(2) {
            trie.insert(i);
        }

        // Range includes odd numbers, but they don't exist
        let keys: Vec<u64> = trie.range(10..20).collect();
        assert_eq!(keys, vec![10, 12, 14, 16, 18]);
    }

    #[test]
    fn test_range_outside() {
        let mut trie = Trie::<u64>::new();
        for i in 0..50 {
            trie.insert(i);
        }

        // Range completely outside trie
        let keys: Vec<u64> = trie.range(100..200).collect();
        assert_eq!(keys, Vec::<u64>::new());
    }

    #[test]
    fn test_range_partial_overlap() {
        let mut trie = Trie::<u64>::new();
        for i in 0..50 {
            trie.insert(i);
        }

        // Range partially overlaps
        let keys: Vec<u64> = trie.range(40..60).collect();
        assert_eq!(keys.len(), 10); // 40-49
        assert_eq!(keys[0], 40);
        assert_eq!(keys[9], 49);
    }

    #[test]
    fn test_u128_iter_large() {
        let mut trie = Trie::<u128>::new();
        let large = 1u128 << 100;
        
        trie.insert(large);
        trie.insert(large + 1);
        trie.insert(large + 2);
        
        assert_eq!(trie.len(), 3);
        assert!(trie.contains(large));
        
        let keys: Vec<u128> = trie.iter().collect();
        assert_eq!(keys.len(), 3);
        assert_eq!(keys[0], large, "Expected {}, got {}", large, keys[0]);
        assert_eq!(keys[1], large + 1);
        assert_eq!(keys[2], large + 2);
    }

    #[test]
    fn test_range_u128_simple() {
        // Test with simple u128 values first
        let mut trie = Trie::<u128>::new();

        for i in 0..100 {
            trie.insert(i);
        }

        let keys: Vec<u128> = trie.range(10..20).collect();
        assert_eq!(keys.len(), 10, "Expected 10 keys, got {}", keys.len());
        if !keys.is_empty() {
            assert_eq!(keys[0], 10);
            if keys.len() >= 10 {
                assert_eq!(keys[9], 19);
            }
        }
    }

    #[test]
    fn test_range_u128() {
        let mut trie = Trie::<u128>::new();
        let large_base = 1u128 << 100;

        for i in 0..100 {
            trie.insert(large_base + i);
        }

        // Check trie is set up correctly
        assert_eq!(trie.len(), 100);
        assert!(trie.contains(large_base + 10));
        assert!(trie.contains(large_base + 19));
        assert!(trie.contains(large_base + 20)); // 20 is IN range 0..100
        assert!(!trie.contains(large_base + 100)); // 100 is OUT of range

        let keys: Vec<u128> = trie.range((large_base + 10)..(large_base + 20)).collect();
        
        // Debug: check what we actually got
        if keys.len() != 10 {
            // Check first few and last few keys
            let sample_size = 5.min(keys.len());
            let first_few: Vec<u128> = keys.iter().take(sample_size).copied().collect();
            let expected_first: Vec<u128> = (0..sample_size).map(|i| large_base + 10 + i as u128).collect();
            
            assert_eq!(first_few, expected_first, "First keys don't match expected range");
        }
        
        assert_eq!(keys.len(), 10, "Expected 10 keys in range, got {}", keys.len());
        assert_eq!(keys[0], large_base + 10);
        assert_eq!(keys[9], large_base + 19);
    }
}
