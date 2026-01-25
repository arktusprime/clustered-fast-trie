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
pub use basic::{clear_bit, is_set, is_set_seqlock, set_bit, test_and_set_bit};
pub use bulk::{
    clear_all, clear_bits, clear_bits_seqlock, clear_range, clear_range_seqlock, set_all, set_bits,
    set_bits_seqlock, set_range, set_range_seqlock,
};
pub use check::{
    are_bits_set, are_bits_set_seqlock, is_empty, is_empty_seqlock, is_full, is_full_seqlock,
    is_range_set, is_range_set_seqlock,
};
pub use intrinsics::{leading_zeros, popcount, trailing_zeros};
pub use search::{
    count_bits, count_bits_seqlock, count_range, count_range_seqlock, max_bit, max_bit_seqlock,
    min_bit, min_bit_seqlock, next_set_bit, next_set_bit_seqlock, prev_set_bit,
    prev_set_bit_seqlock,
};
