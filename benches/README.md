# Benchmarks

Performance comparison of `clustered-fast-trie` with `BTreeSet` from the standard library.

## Running benchmarks

### All benchmarks
```bash
cargo bench
```

### Specific benchmark file
```bash
# Comparison with full datasets
cargo bench --bench comparison

# Single operations with varying sizes
cargo bench --bench single_ops
```

### Specific test
```bash
cargo bench --bench comparison insert_sequential
cargo bench --bench comparison contains
cargo bench --bench single_ops single_insert
```

## Benchmark structure

### `comparison.rs` - Comparison with BTreeSet

**Test scenarios:**

1. **insert_sequential** - sequential key insertion (0, 1, 2, ...)
   - Sizes: 1K, 10K, 100K elements
   - Shows performance on ideal data

2. **insert_clustered** - clustered data insertion (multiple ranges)
   - Pattern: 0-1000, 10000-11000, 20000-21000, 30000-31000
   - Realistic scenario for Kafka offsets, timestamps

3. **insert_random** - pseudorandom key insertion
   - 10K elements
   - Worst case for Trie

4. **contains** - key lookup
   - existing: search for present keys
   - missing: search for absent keys
   - 10K elements in dataset

5. **contains_clustered** - lookup in clustered data
   - Multiple clusters of 1K elements each
   - Shows cache-friendly patterns

6. **remove** - key removal
   - sequential: removal in order
   - sparse: removal of every other key
   - 10K elements

7. **mixed_workload** - mixed workload
   - insert → contains → remove → insert
   - Realistic usage scenario

### `single_ops.rs` - Single operations

**Test scenarios:**

1. **single_insert** - one insert into dataset of varying size
   - Sizes: 100, 1K, 10K, 100K
   - Shows how latency grows with size

2. **single_contains** - one lookup in dataset of varying size
   - hit: key found (in the middle)
   - miss: key not found
   - Sizes: 100, 1K, 10K, 100K

3. **single_remove** - one removal from dataset of varying size
   - Removal from the middle
   - Sizes: 100, 1K, 10K, 100K

4. **sequential_pattern** - sequential inserts
   - Forward order (0→1000)
   - Reverse order (1000→0)
   - Shows cache benefits

5. **worst_case_insert** - worst case for insertion
   - Alternating pattern: far-apart keys
   - No cache benefits

## Expected results

### Where Trie should be faster:
- ✅ Sequential inserts (cache hot path)
- ✅ Clustered data (cache locality)
- ✅ Large datasets (O(log log U) vs O(log n))
- ✅ Contains on clustered data

### Where BTreeSet may be faster:
- ⚠️ Random inserts (no cache benefits)
- ⚠️ Very small datasets (<100 elements)
- ⚠️ Sparse random keys

### O(log log U) vs O(log n):
- Trie: depends on key size (u64 = 8 levels)
- BTreeSet: depends on number of elements
- Crossover point: ~256 elements (log₂ 256 = 8)

## Interpreting results

Criterion outputs:
- **time**: mean operation time
- **thrpt**: throughput (ops/sec)
- **change**: change relative to previous run

Look at:
1. Absolute values (ns/op)
2. Scaling with size
3. Variance (latency stability)

## Visualization

After running benchmarks, Criterion creates HTML reports:

```bash
# Enhance report (values in tables, violin plots) and open
cargo run -p report-enhancer
start target/criterion/report/index.html  # Windows
open target/criterion/report/index.html   # macOS
xdg-open target/criterion/report/index.html  # Linux
```

## Notes

- Use `--release` for realistic results (cargo bench does this automatically)
- Close other applications for stable measurements
- First run creates baseline for comparison
- Subsequent runs are compared against baseline
