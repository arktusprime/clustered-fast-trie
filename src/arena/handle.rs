//! Segment handle for fast API

use super::SegmentId;

/// Segment handle for cached access (fast API)
#[allow(dead_code)]
pub struct SegmentHandle {
    segment_id: SegmentId,
    // TODO: Implementation
}

#[allow(dead_code)]
impl SegmentHandle {
    /// Create a new segment handle
    pub fn new(_segment_id: SegmentId) -> Self {
        todo!()
    }
}
