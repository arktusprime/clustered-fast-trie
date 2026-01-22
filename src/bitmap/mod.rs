//! Low-level bitmap operations using CPU intrinsics.
//!
//! These functions provide efficient bit manipulation for 256-bit bitmaps
//! represented as arrays of 4 u64 words.

mod basic;
mod bulk;
mod check;
mod intrinsics;
mod search;

// Re-export all public functions
pub use basic::{clear_bit, is_set, set_bit};
pub use bulk::{clear_all, clear_bits, clear_range, set_all, set_bits, set_range};
pub use check::{are_bits_set, is_empty, is_full, is_range_set};
pub use intrinsics::{leading_zeros, popcount, trailing_zeros};
pub use search::{count_range, next_set_bit, prev_set_bit};
