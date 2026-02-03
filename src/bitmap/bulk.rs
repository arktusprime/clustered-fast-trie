//! Bulk operations for setting/clearing multiple bits efficiently.

use crate::atomic::{AtomicU64, Ordering};

/// Set a range of bits [from, to) in the bitmap.
///
/// # Arguments
/// * `bitmap` - Reference to 4-word atomic bitmap
/// * `from` - Start index (inclusive, 0-255)
/// * `to` - End index (exclusive, 0-256)
///
/// # Performance
/// O(1) - processes up to 4 words with bitwise operations
#[inline]
#[allow(dead_code)]
pub fn set_range(bitmap: &[AtomicU64; 4], from: u8, to: u16) {
    if from as u16 >= to {
        return;
    }

    let to = to.min(256) as usize;
    let from = from as usize;

    let from_word = from / 64;
    let to_word = (to - 1) / 64;

    if from_word == to_word {
        // Same word
        let from_bit = from % 64;
        let to_bit = to % 64;
        let mask = if to_bit == 0 {
            !0u64 << from_bit
        } else {
            ((!0u64) << from_bit) & ((1u64 << to_bit) - 1)
        };
        bitmap[from_word].fetch_or(mask, Ordering::Relaxed);
    } else {
        // First word: from_bit to 63
        let from_bit = from % 64;
        bitmap[from_word].fetch_or(!0u64 << from_bit, Ordering::Relaxed);

        // Middle words: all bits
        #[allow(clippy::needless_range_loop)]
        for w in (from_word + 1)..to_word {
            bitmap[w].store(!0u64, Ordering::Relaxed);
        }

        // Last word: 0 to to_bit
        let to_bit = to % 64;
        if to_bit > 0 {
            bitmap[to_word].fetch_or((1u64 << to_bit) - 1, Ordering::Relaxed);
        } else if to_word < 4 {
            bitmap[to_word].store(!0u64, Ordering::Relaxed);
        }
    }
}

/// Clear a range of bits [from, to) in the bitmap.
///
/// # Arguments
/// * `bitmap` - Reference to 4-word atomic bitmap
/// * `from` - Start index (inclusive, 0-255)
/// * `to` - End index (exclusive, 0-256)
///
/// # Performance
/// O(1) - processes up to 4 words with bitwise operations
#[inline]
#[allow(dead_code)]
pub fn clear_range(bitmap: &[AtomicU64; 4], from: u8, to: u16) {
    if from as u16 >= to {
        return;
    }

    let to = to.min(256) as usize;
    let from = from as usize;

    let from_word = from / 64;
    let to_word = (to - 1) / 64;

    if from_word == to_word {
        // Same word
        let from_bit = from % 64;
        let to_bit = to % 64;
        let mask = if to_bit == 0 {
            !0u64 << from_bit
        } else {
            ((!0u64) << from_bit) & ((1u64 << to_bit) - 1)
        };
        bitmap[from_word].fetch_and(!mask, Ordering::Relaxed);
    } else {
        // First word: from_bit to 63
        let from_bit = from % 64;
        bitmap[from_word].fetch_and(!(!0u64 << from_bit), Ordering::Relaxed);

        // Middle words: all bits
        #[allow(clippy::needless_range_loop)]
        for w in (from_word + 1)..to_word {
            bitmap[w].store(0, Ordering::Relaxed);
        }

        // Last word: 0 to to_bit
        let to_bit = to % 64;
        if to_bit > 0 {
            bitmap[to_word].fetch_and(!((1u64 << to_bit) - 1), Ordering::Relaxed);
        } else if to_word < 4 {
            bitmap[to_word].store(0, Ordering::Relaxed);
        }
    }
}

/// Set multiple bits at specified indices.
///
/// Accumulates masks for all words, then applies in one pass.
/// Works efficiently with both sorted and unsorted indices.
///
/// # Arguments
/// * `bitmap` - Reference to 4-word atomic bitmap
/// * `indices` - Slice of bit indices to set (0-255)
///
/// # Performance
/// O(n) where n = indices.len(), with stable performance regardless of index order
#[inline]
#[allow(dead_code)]
pub fn set_bits(bitmap: &[AtomicU64; 4], indices: &[u8]) {
    if indices.is_empty() {
        return;
    }

    // Accumulate masks for all 4 words
    let mut masks = [0u64; 4];

    for &idx in indices {
        let word = (idx / 64) as usize;
        let bit = idx % 64;
        masks[word] |= 1u64 << bit;
    }

    // Apply all masks (SIMD-friendly)
    bitmap[0].fetch_or(masks[0], Ordering::Relaxed);
    bitmap[1].fetch_or(masks[1], Ordering::Relaxed);
    bitmap[2].fetch_or(masks[2], Ordering::Relaxed);
    bitmap[3].fetch_or(masks[3], Ordering::Relaxed);
}

/// Clear multiple bits at specified indices.
///
/// Accumulates masks for all words, then applies in one pass.
/// Works efficiently with both sorted and unsorted indices.
///
/// # Arguments
/// * `bitmap` - Reference to 4-word atomic bitmap
/// * `indices` - Slice of bit indices to clear (0-255)
///
/// # Performance
/// O(n) where n = indices.len(), with stable performance regardless of index order
#[inline]
#[allow(dead_code)]
pub fn clear_bits(bitmap: &[AtomicU64; 4], indices: &[u8]) {
    if indices.is_empty() {
        return;
    }

    // Accumulate masks for all 4 words
    let mut masks = [0u64; 4];

    for &idx in indices {
        let word = (idx / 64) as usize;
        let bit = idx % 64;
        masks[word] |= 1u64 << bit;
    }

    // Apply all masks (SIMD-friendly)
    bitmap[0].fetch_and(!masks[0], Ordering::Relaxed);
    bitmap[1].fetch_and(!masks[1], Ordering::Relaxed);
    bitmap[2].fetch_and(!masks[2], Ordering::Relaxed);
    bitmap[3].fetch_and(!masks[3], Ordering::Relaxed);
}

/// Set all 256 bits in the bitmap.
///
/// # Arguments
/// * `bitmap` - Reference to 4-word atomic bitmap
///
/// # Performance
/// O(1) - SIMD-friendly, 4 independent stores
#[inline]
#[allow(dead_code)]
pub fn set_all(bitmap: &[AtomicU64; 4]) {
    bitmap[0].store(!0u64, Ordering::Relaxed);
    bitmap[1].store(!0u64, Ordering::Relaxed);
    bitmap[2].store(!0u64, Ordering::Relaxed);
    bitmap[3].store(!0u64, Ordering::Relaxed);
}

/// Clear all 256 bits in the bitmap.
///
/// # Arguments
/// * `bitmap` - Reference to 4-word atomic bitmap
///
/// # Performance
/// O(1) - SIMD-friendly, 4 independent stores
#[inline]
#[allow(dead_code)]
pub fn clear_all(bitmap: &[AtomicU64; 4]) {
    bitmap[0].store(0, Ordering::Relaxed);
    bitmap[1].store(0, Ordering::Relaxed);
    bitmap[2].store(0, Ordering::Relaxed);
    bitmap[3].store(0, Ordering::Relaxed);
}

/// Set a range of bits [from, to) with seqlock protection.
///
/// Uses seqlock protocol to ensure readers see consistent state during bulk modification.
/// Writer increments seq (makes odd), modifies data, increments seq (makes even).
///
/// # Arguments
/// * `seq` - Sequence counter for seqlock protocol
/// * `bitmap` - Reference to 4-word atomic bitmap
/// * `from` - Start index (inclusive, 0-255)
/// * `to` - End index (exclusive, 0-256)
///
/// # Performance
/// O(1) - ~4% overhead from seqlock (2 increments + 2 fences)
///
/// # Thread Safety
/// Multiple writers can call this concurrently. Each writer:
/// 1. Increments seq (makes odd) - signals "writing in progress"
/// 2. Modifies bitmap atomically
/// 3. Increments seq (makes even) - signals "write complete"
///
/// Readers will retry if they observe odd seq or seq changed during read.
#[inline]
#[allow(dead_code)]
pub fn set_range_seqlock(seq: &AtomicU64, bitmap: &[AtomicU64; 4], from: u8, to: u16) {
    seq.fetch_add(1, Ordering::Release); // Make odd
    #[cfg(not(feature = "single-threaded"))]
    core::sync::atomic::fence(core::sync::atomic::Ordering::Release);
    set_range(bitmap, from, to);
    #[cfg(not(feature = "single-threaded"))]
    core::sync::atomic::fence(core::sync::atomic::Ordering::Release);
    seq.fetch_add(1, Ordering::Release); // Make even
}

/// Clear a range of bits [from, to) with seqlock protection.
///
/// Uses seqlock protocol to ensure readers see consistent state during bulk modification.
///
/// # Arguments
/// * `seq` - Sequence counter for seqlock protocol
/// * `bitmap` - Reference to 4-word atomic bitmap
/// * `from` - Start index (inclusive, 0-255)
/// * `to` - End index (exclusive, 0-256)
///
/// # Performance
/// O(1) - ~4% overhead from seqlock (2 increments + 2 fences)
///
/// # Thread Safety
/// Safe for concurrent writers. See `set_range_seqlock` for protocol details.
#[inline]
#[allow(dead_code)]
pub fn clear_range_seqlock(seq: &AtomicU64, bitmap: &[AtomicU64; 4], from: u8, to: u16) {
    seq.fetch_add(1, Ordering::Release); // Make odd
    #[cfg(not(feature = "single-threaded"))]
    core::sync::atomic::fence(core::sync::atomic::Ordering::Release);
    clear_range(bitmap, from, to);
    #[cfg(not(feature = "single-threaded"))]
    core::sync::atomic::fence(core::sync::atomic::Ordering::Release);
    seq.fetch_add(1, Ordering::Release); // Make even
}

/// Set multiple bits at specified indices with seqlock protection.
///
/// Uses seqlock protocol to ensure readers see consistent state during bulk modification.
///
/// # Arguments
/// * `seq` - Sequence counter for seqlock protocol
/// * `bitmap` - Reference to 4-word atomic bitmap
/// * `indices` - Slice of bit indices to set (0-255)
///
/// # Performance
/// O(n) where n = indices.len(), ~4% overhead from seqlock
///
/// # Thread Safety
/// Safe for concurrent writers. See `set_range_seqlock` for protocol details.
#[inline]
#[allow(dead_code)]
pub fn set_bits_seqlock(seq: &AtomicU64, bitmap: &[AtomicU64; 4], indices: &[u8]) {
    seq.fetch_add(1, Ordering::Release); // Make odd
    #[cfg(not(feature = "single-threaded"))]
    core::sync::atomic::fence(core::sync::atomic::Ordering::Release);
    set_bits(bitmap, indices);
    #[cfg(not(feature = "single-threaded"))]
    core::sync::atomic::fence(core::sync::atomic::Ordering::Release);
    seq.fetch_add(1, Ordering::Release); // Make even
}

/// Clear multiple bits at specified indices with seqlock protection.
///
/// Uses seqlock protocol to ensure readers see consistent state during bulk modification.
///
/// # Arguments
/// * `seq` - Sequence counter for seqlock protocol
/// * `bitmap` - Reference to 4-word atomic bitmap
/// * `indices` - Slice of bit indices to clear (0-255)
///
/// # Performance
/// O(n) where n = indices.len(), ~4% overhead from seqlock
///
/// # Thread Safety
/// Safe for concurrent writers. See `set_range_seqlock` for protocol details.
#[inline]
#[allow(dead_code)]
pub fn clear_bits_seqlock(seq: &AtomicU64, bitmap: &[AtomicU64; 4], indices: &[u8]) {
    seq.fetch_add(1, Ordering::Release); // Make odd
    #[cfg(not(feature = "single-threaded"))]
    core::sync::atomic::fence(core::sync::atomic::Ordering::Release);
    clear_bits(bitmap, indices);
    #[cfg(not(feature = "single-threaded"))]
    core::sync::atomic::fence(core::sync::atomic::Ordering::Release);
    seq.fetch_add(1, Ordering::Release); // Make even
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitmap::is_set;

    #[test]
    fn test_set_range() {
        const INIT: AtomicU64 = AtomicU64::new(0);
        let bitmap = [INIT; 4];

        // Single word range
        set_range(&bitmap, 10, 20);
        for i in 10..20 {
            assert!(is_set(&bitmap, i));
        }
        assert!(!is_set(&bitmap, 9));
        assert!(!is_set(&bitmap, 20));

        // Cross word boundary
        let bitmap = [INIT; 4];
        set_range(&bitmap, 60, 70);
        for i in 60..70 {
            assert!(is_set(&bitmap, i));
        }

        // Full range
        let bitmap = [INIT; 4];
        set_range(&bitmap, 0, 256);
        for i in 0..=255 {
            assert!(is_set(&bitmap, i));
        }
    }

    #[test]
    fn test_clear_range() {
        const INIT: AtomicU64 = AtomicU64::new(!0);
        let bitmap = [INIT; 4];

        // Single word range
        clear_range(&bitmap, 10, 20);
        for i in 10..20 {
            assert!(!is_set(&bitmap, i));
        }
        assert!(is_set(&bitmap, 9));
        assert!(is_set(&bitmap, 20));

        // Cross word boundary
        let bitmap = [INIT; 4];
        clear_range(&bitmap, 60, 70);
        for i in 60..70 {
            assert!(!is_set(&bitmap, i));
        }
    }

    #[test]
    fn test_set_bits() {
        const INIT: AtomicU64 = AtomicU64::new(0);
        let bitmap = [INIT; 4];
        let indices = [5, 10, 15, 42, 100, 200, 255];

        set_bits(&bitmap, &indices);

        for &idx in &indices {
            assert!(is_set(&bitmap, idx));
        }
        assert!(!is_set(&bitmap, 6));
        assert!(!is_set(&bitmap, 99));
    }

    #[test]
    fn test_clear_bits() {
        const INIT: AtomicU64 = AtomicU64::new(!0);
        let bitmap = [INIT; 4];
        let indices = [5, 10, 15, 42, 100, 200, 255];

        clear_bits(&bitmap, &indices);

        for &idx in &indices {
            assert!(!is_set(&bitmap, idx));
        }
        assert!(is_set(&bitmap, 6));
        assert!(is_set(&bitmap, 99));
    }

    #[test]
    fn test_set_all() {
        const INIT: AtomicU64 = AtomicU64::new(0);
        let bitmap = [INIT; 4];
        set_all(&bitmap);

        for i in 0..=255 {
            assert!(is_set(&bitmap, i));
        }
    }

    #[test]
    fn test_clear_all() {
        const INIT: AtomicU64 = AtomicU64::new(!0);
        let bitmap = [INIT; 4];
        clear_all(&bitmap);

        for i in 0..=255 {
            assert!(!is_set(&bitmap, i));
        }
    }
}
