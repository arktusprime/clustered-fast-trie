//! # clustered-fast-trie
//!
//! Fast ordered integer set (u32/u64/u128) optimized for clustered data.
//!
//! A specialized data structure for storing sets of integers with excellent performance
//! for sequential or clustered keys (e.g., Kafka offsets, timestamps, auto-increment IDs).
//!
//! ## Features
//!
//! - **Fast operations**: O(log log U) insert, contains, remove, successor, predecessor
//! - **O(1) min/max**: Cached minimum and maximum values
//! - **Efficient iteration**: O(1) per element via linked list of leaf nodes
//! - **Generic keys**: Supports u32, u64, and u128
//! - **Memory efficient**: Bitmap compression for clustered data
//! - **no_std compatible**: Requires only `alloc`
//!
//! ## Example
//!
//! ```rust
//! use clustered_fast_trie::Trie;
//!
//! // Create a trie for u64 keys
//! let mut trie = Trie::<u64>::new();
//!
//! // Insert some keys
//! trie.insert(100);
//! trie.insert(200);
//! trie.insert(150);
//!
//! // Check membership
//! assert!(trie.contains(100));
//! assert!(!trie.contains(999));
//!
//! // Get min/max (O(1))
//! assert_eq!(trie.min(), Some(100));
//! assert_eq!(trie.max(), Some(200));
//!
//! // Navigate the set
//! assert_eq!(trie.successor(100), Some(150));
//! assert_eq!(trie.predecessor(200), Some(150));
//!
//! // Iterate in sorted order
//! let keys: Vec<u64> = trie.iter().collect();
//! assert_eq!(keys, vec![100, 150, 200]);
//!
//! // Range queries
//! let range: Vec<u64> = trie.range(100..200).collect();
//! assert_eq!(range, vec![100, 150]);
//!
//! // Remove keys
//! trie.remove(150);
//! assert_eq!(trie.len(), 2);
//! ```
//!
//! ## Use Cases
//!
//! This data structure excels when:
//! - Keys are sequential or clustered (e.g., 1000-1255, 2000-2100)
//! - You need ordered operations (min/max/successor/predecessor)
//! - You need efficient iteration in sorted order
//! - Memory efficiency matters for large ranges
//!
//! Examples: Kafka offset tracking, time-series data, ID ranges.
//!
//! ## Performance Characteristics
//!
//! | Operation | Time Complexity | Notes |
//! |-----------|----------------|-------|
//! | insert    | O(log log U)   | U = key universe size |
//! | contains  | O(log log U)   | |
//! | remove    | O(log log U)   | |
//! | min/max   | O(1)           | Cached |
//! | successor | O(log log U)   | O(1) for adjacent keys |
//! | predecessor | O(log log U) | O(1) for adjacent keys |
//! | iter      | O(1) per element | Via linked list |
//! | range     | O(log log U) + O(k) | k = elements in range |
//!
//! ## Threading Modes
//!
//! - **Default**: Thread-safe using atomic operations
//! - **Single-threaded**: Compile with `--features single-threaded` for 10-15% performance boost

#![no_std]

extern crate alloc;

mod arena;
mod atomic;
mod bitmap;
mod constants;
mod key;
mod trie;

pub use key::TrieKey;
pub use trie::{Iter, RangeIter, Trie};
