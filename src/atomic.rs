//! Zero-overhead atomic wrapper supporting both single-threaded and multi-threaded modes

/// Zero-overhead atomic wrapper.
///
/// Provides unified API for both single-threaded and multi-threaded modes:
/// - Multi-threaded (default): uses `core::sync::atomic::AtomicU64`
/// - Single-threaded (feature flag): uses `core::cell::Cell<u64>`
///
/// # Performance
/// - Multi-threaded: thread-safe, ~10-15% overhead from atomics
/// - Single-threaded: 10-15% faster, no atomic overhead
///
/// # Compile-time Selection
/// ```bash
/// cargo build                          # Multi-threaded (default)
/// cargo build --features single-threaded  # Single-threaded (faster)
/// ```
#[derive(Debug)]
pub struct AtomicU64 {
    #[cfg(feature = "single-threaded")]
    inner: core::cell::Cell<u64>,

    #[cfg(not(feature = "single-threaded"))]
    inner: core::sync::atomic::AtomicU64,
}

// Re-export Ordering for convenience
#[cfg(not(feature = "single-threaded"))]
pub use core::sync::atomic::Ordering;

#[cfg(feature = "single-threaded")]
#[derive(Clone, Copy, Debug)]
pub enum Ordering {
    Relaxed,
    Acquire,
    Release,
    AcqRel,
    SeqCst,
}

impl AtomicU64 {
    /// Create a new atomic with initial value.
    ///
    /// # Performance
    /// O(1) - compile-time selection, zero runtime overhead
    #[inline(always)]
    pub const fn new(val: u64) -> Self {
        #[cfg(feature = "single-threaded")]
        return AtomicU64 {
            inner: core::cell::Cell::new(val),
        };

        #[cfg(not(feature = "single-threaded"))]
        return AtomicU64 {
            inner: core::sync::atomic::AtomicU64::new(val),
        };
    }

    /// Load value with specified ordering.
    ///
    /// # Arguments
    /// * `ordering` - Memory ordering (ignored in single-threaded mode)
    ///
    /// # Performance
    /// - Single-threaded: O(1) - direct Cell::get()
    /// - Multi-threaded: O(1) - atomic load
    #[inline(always)]
    pub fn load(&self, ordering: Ordering) -> u64 {
        #[cfg(feature = "single-threaded")]
        {
            let _ = ordering; // Suppress unused warning
            self.inner.get()
        }

        #[cfg(not(feature = "single-threaded"))]
        self.inner.load(ordering)
    }

    /// Store value with specified ordering.
    ///
    /// # Arguments
    /// * `val` - Value to store
    /// * `ordering` - Memory ordering (ignored in single-threaded mode)
    ///
    /// # Performance
    /// - Single-threaded: O(1) - direct Cell::set()
    /// - Multi-threaded: O(1) - atomic store
    #[inline(always)]
    pub fn store(&self, val: u64, ordering: Ordering) {
        #[cfg(feature = "single-threaded")]
        {
            let _ = ordering; // Suppress unused warning
            self.inner.set(val);
        }

        #[cfg(not(feature = "single-threaded"))]
        self.inner.store(val, ordering);
    }

    /// Bitwise OR with current value, returns previous value.
    ///
    /// # Arguments
    /// * `val` - Value to OR with current
    /// * `ordering` - Memory ordering (ignored in single-threaded mode)
    ///
    /// # Returns
    /// Previous value before OR operation
    ///
    /// # Performance
    /// - Single-threaded: O(1) - get + set
    /// - Multi-threaded: O(1) - atomic fetch_or
    #[inline(always)]
    pub fn fetch_or(&self, val: u64, ordering: Ordering) -> u64 {
        #[cfg(feature = "single-threaded")]
        {
            let _ = ordering; // Suppress unused warning
            let old = self.inner.get();
            self.inner.set(old | val);
            old
        }

        #[cfg(not(feature = "single-threaded"))]
        self.inner.fetch_or(val, ordering)
    }

    /// Bitwise AND with current value, returns previous value.
    ///
    /// # Arguments
    /// * `val` - Value to AND with current
    /// * `ordering` - Memory ordering (ignored in single-threaded mode)
    ///
    /// # Returns
    /// Previous value before AND operation
    ///
    /// # Performance
    /// - Single-threaded: O(1) - get + set
    /// - Multi-threaded: O(1) - atomic fetch_and
    #[inline(always)]
    pub fn fetch_and(&self, val: u64, ordering: Ordering) -> u64 {
        #[cfg(feature = "single-threaded")]
        {
            let _ = ordering; // Suppress unused warning
            let old = self.inner.get();
            self.inner.set(old & val);
            old
        }

        #[cfg(not(feature = "single-threaded"))]
        self.inner.fetch_and(val, ordering)
    }

    /// Compare and swap operation.
    ///
    /// # Arguments
    /// * `current` - Expected current value
    /// * `new` - New value to set if current matches
    /// * `success` - Memory ordering for successful operation
    /// * `failure` - Memory ordering for failed operation
    ///
    /// # Returns
    /// Result with previous value and success flag
    ///
    /// # Performance
    /// - Single-threaded: O(1) - compare + conditional set
    /// - Multi-threaded: O(1) - atomic compare_exchange
    #[inline(always)]
    pub fn compare_exchange(
        &self,
        current: u64,
        new: u64,
        success: Ordering,
        failure: Ordering,
    ) -> Result<u64, u64> {
        #[cfg(feature = "single-threaded")]
        {
            let _ = (success, failure); // Suppress unused warnings
            let old = self.inner.get();
            if old == current {
                self.inner.set(new);
                Ok(old)
            } else {
                Err(old)
            }
        }

        #[cfg(not(feature = "single-threaded"))]
        self.inner.compare_exchange(current, new, success, failure)
    }

    /// Add to current value, returns previous value.
    ///
    /// # Arguments
    /// * `val` - Value to add to current
    /// * `ordering` - Memory ordering (ignored in single-threaded mode)
    ///
    /// # Returns
    /// Previous value before addition
    ///
    /// # Performance
    /// - Single-threaded: O(1) - get + set
    /// - Multi-threaded: O(1) - atomic fetch_add
    #[inline(always)]
    pub fn fetch_add(&self, val: u64, ordering: Ordering) -> u64 {
        #[cfg(feature = "single-threaded")]
        {
            let _ = ordering; // Suppress unused warning
            let old = self.inner.get();
            self.inner.set(old.wrapping_add(val));
            old
        }

        #[cfg(not(feature = "single-threaded"))]
        self.inner.fetch_add(val, ordering)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let atomic = AtomicU64::new(42);
        assert_eq!(atomic.load(Ordering::Relaxed), 42);
    }

    #[test]
    fn test_load_store() {
        let atomic = AtomicU64::new(0);
        atomic.store(123, Ordering::Relaxed);
        assert_eq!(atomic.load(Ordering::Relaxed), 123);
    }

    #[test]
    fn test_fetch_or() {
        let atomic = AtomicU64::new(0b1010);
        let old = atomic.fetch_or(0b1100, Ordering::Relaxed);
        assert_eq!(old, 0b1010);
        assert_eq!(atomic.load(Ordering::Relaxed), 0b1110);
    }

    #[test]
    fn test_fetch_and() {
        let atomic = AtomicU64::new(0b1110);
        let old = atomic.fetch_and(0b1100, Ordering::Relaxed);
        assert_eq!(old, 0b1110);
        assert_eq!(atomic.load(Ordering::Relaxed), 0b1100);
    }

    #[test]
    fn test_compare_exchange() {
        let atomic = AtomicU64::new(42);

        // Successful exchange
        let result = atomic.compare_exchange(42, 100, Ordering::Relaxed, Ordering::Relaxed);
        assert_eq!(result, Ok(42));
        assert_eq!(atomic.load(Ordering::Relaxed), 100);

        // Failed exchange
        let result = atomic.compare_exchange(42, 200, Ordering::Relaxed, Ordering::Relaxed);
        assert_eq!(result, Err(100));
        assert_eq!(atomic.load(Ordering::Relaxed), 100);
    }

    #[test]
    fn test_fetch_add() {
        let atomic = AtomicU64::new(10);
        let old = atomic.fetch_add(5, Ordering::Relaxed);
        assert_eq!(old, 10);
        assert_eq!(atomic.load(Ordering::Relaxed), 15);

        // Test wrapping
        let atomic = AtomicU64::new(u64::MAX);
        let old = atomic.fetch_add(1, Ordering::Relaxed);
        assert_eq!(old, u64::MAX);
        assert_eq!(atomic.load(Ordering::Relaxed), 0);
    }
}
