# clustered-fast-trie

In-memory ordered set for integer keys (u32/u64/u128) optimized for clustered data.

## Key Features

- **O(log log U) complexity** — performance independent of dataset size n
- **Stable latency** — predictable regardless of data volume
- **Fast sequential inserts** — 0.8-1.2ns via hot path caching
- **Bulk operations** — 10-100x faster than individual inserts
- **Multi-tenancy** — unlimited isolation via segmented arena
- **Zero dependencies** — no_std compatible (requires alloc)

## Positioning

Specialized **in-memory ordered set** for integer keys with:
- High-throughput tracking (Kafka offsets, event IDs)
- Time-series indexing (timestamp presence checks)
- Multi-tenant isolation (per-user/per-metric sets)
- Real-time analytics (fast range counting)
- Event sourcing (sequence number tracking)

## Quick Start

```rust
use clustered_fast_trie::Trie;

// Create a trie for u64 keys
let mut trie = Trie::<u64>::new();

// Insert keys
trie.insert(100);
trie.insert(200);
trie.insert(150);

// Check membership
assert!(trie.contains(150));
assert!(!trie.contains(999));

// Get min/max (O(1))
assert_eq!(trie.min(), Some(100));
assert_eq!(trie.max(), Some(200));

// Navigate the set
assert_eq!(trie.successor(100), Some(150));
assert_eq!(trie.predecessor(200), Some(150));

// Iterate in sorted order
let keys: Vec<u64> = trie.iter().collect();
assert_eq!(keys, vec![100, 150, 200]);

// Range queries
let range: Vec<u64> = trie.range(100..200).collect();
assert_eq!(range, vec![100, 150]);

// Remove keys
trie.remove(150);
assert_eq!(trie.len(), 2);
```

## Use Cases

- Kafka offset tracking (per-partition)
- Time-series databases (per-metric timestamps)
- Event sourcing (sequence numbers)
- Multi-tenant systems
- IP routing tables

## Architecture

Inspired by van Emde Boas trees and X-fast tries, combining:

- **256-way branching trie** — byte-indexed levels for O(log log U) operations
- **Hierarchical bitmaps** — fast min/max/successor via TZCNT/LZCNT intrinsics
- **Linked list of leaves** — O(1) per-element iteration
- **Arena allocation** — cache-friendly sequential memory layout
- **Segmented architecture** — per-tenant isolation for multi-tenancy

## API Overview

### Core Operations

| Operation | Time Complexity | Description |
|-----------|----------------|-------------|
| `insert(key)` | O(log log U) | Insert a key |
| `contains(key)` | O(log log U) | Check if key exists |
| `remove(key)` | O(log log U) | Remove a key |
| `min()` | O(1) | Get minimum key (cached) |
| `max()` | O(1) | Get maximum key (cached) |
| `successor(key)` | O(log log U) | Next key after given key |
| `predecessor(key)` | O(log log U) | Previous key before given key |
| `iter()` | O(1) per element | Iterate in sorted order |
| `range(start..end)` | O(log log U) + O(k) | Iterate over range |
| `len()` | O(1) | Number of keys |
| `is_empty()` | O(1) | Check if empty |

### Supported Key Types

- **u32**: 4 bytes, 3 internal levels
- **u64**: 8 bytes, 7 internal levels  
- **u128**: 16 bytes, 15 internal levels

## Examples

Run the basic usage example:

```bash
cargo run --example basic_usage
```

## Trade-offs

✅ Excellent for sequential/clustered data  
✅ Predictable O(log log U) performance  
✅ Fast iteration via linked list  
✅ O(1) min/max operations  

❌ Memory-heavy for sparse random keys  
❌ Not optimized for extremely large key ranges  

## Documentation

Generate and view full API documentation:

```bash
cargo doc --no-deps --open
```

## License

MIT OR Apache-2.0
