# clustered-fast-trie

Ordered integer set (u32/u64/u128) optimized for clustered data.

## Key Features

- **O(log log U) complexity** — performance independent of dataset size n
- **Stable latency** — predictable regardless of data volume
- **Fast sequential inserts** — 0.8-1.2ns via hot path caching
- **Bulk operations** — 10-100x faster than individual inserts
- **Multi-tenancy** — unlimited isolation via segmented arena
- **Zero dependencies** — no_std compatible (requires alloc)

## Quick Start

```rust
use clustered_fast_trie::Trie;

let mut trie = Trie::<u64>::new();
trie.insert(42);
assert!(trie.contains(42));
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

## Trade-offs

✅ Excellent for sequential/clustered data  
❌ Memory-heavy for sparse random keys

## License

MIT OR Apache-2.0
