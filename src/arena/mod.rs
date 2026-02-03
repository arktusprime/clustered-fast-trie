//! Arena allocator for multi-tenant memory management
//!
//! This module provides the arena allocation system for clustered-fast-trie,
//! supporting flexible key ranges, lazy allocation, and transparent defragmentation.

#[allow(clippy::module_inception)]
pub mod arena;
pub mod cache;
pub mod defrag;
pub mod free_list;
pub mod handle;
pub mod numa;
pub mod segment;
pub mod segment_manager;

// Re-exports
pub use arena::Arena;
#[allow(unused_imports)]
pub use cache::SegmentCache;
#[allow(unused_imports)]
pub use segment::{KeyRange, SegmentId, SegmentMeta};
#[allow(unused_imports)]
pub use segment_manager::SegmentManager;
