//! Segment metadata and management

/// Segment identifier (permanent key)
pub type SegmentId = u32;

/// Segment metadata structure
pub struct SegmentMeta {
    // TODO: Implementation
}

/// Key range specification
pub struct KeyRange {
    /// Start of the key range
    pub start: u64,
    /// Size of the key range
    pub size: u64,
}

impl SegmentMeta {
    /// Create new segment metadata
    pub fn new() -> Self {
        todo!()
    }
}
