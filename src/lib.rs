//! # clustered-fast-trie
//!
//! Ordered integer set (u32/u64/u128) for data with locality.
//! O(1) range counting. Stable O(log log U) latency.
//!
//! ## Features
//! - O(1) min/max and range counting
//! - O(log log U) insert, contains, successor, predecessor
//! - Optimized for sequential and clustered data
//! - no_std compatible (requires alloc)

#![no_std]

extern crate alloc;

mod key;

pub use key::TrieKey;
