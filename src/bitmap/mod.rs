//! Low-level bitmap operations using CPU intrinsics.
//!
//! These functions provide efficient bit manipulation for 256-bit bitmaps
//! represented as arrays of 4 u64 words.

mod basic;
mod bulk;
mod check;
mod intrinsics;
mod search;

// Re-export currently used functions
pub use basic::{clear_bit, is_set, set_bit, test_and_set_bit};
pub use bulk::{set_bits, set_range};
pub use check::{is_empty, is_full};
pub use intrinsics::{leading_zeros, popcount, trailing_zeros};
pub use search::count_bits;
