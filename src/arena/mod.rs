//! Arena allocator for multi-tenant memory management
//!
//! This module provides the arena allocation system for clustered-fast-trie,
//! supporting flexible key ranges, lazy allocation, and transparent defragmentation.

pub mod allocator;
pub mod arena;
pub mod cache;
pub mod defrag;
pub mod free_list;
pub mod handle;
pub mod numa;
pub mod segment;
pub mod segment_manager;

// Re-exports
#[deprecated(note = "Use SegmentManager instead - arenas are now stored in nodes")]
pub use allocator::ArenaAllocator;
pub use arena::Arena;
pub use cache::SegmentCache;
pub use segment::{KeyRange, SegmentId, SegmentMeta};
pub use segment_manager::SegmentManager;
