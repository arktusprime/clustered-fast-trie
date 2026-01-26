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
//!
//! ## Threading Modes
//! - Multi-threaded (default): thread-safe atomics
//! - Single-threaded: compile with `--features single-threaded` for 10-15% performance boost

#![no_std]

extern crate alloc;

mod arena;
mod atomic;
mod bitmap;
mod constants;
mod key;
mod trie;

pub use key::TrieKey;
