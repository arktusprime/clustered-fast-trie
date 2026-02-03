use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use clustered_fast_trie::Trie;
use std::collections::BTreeSet;

/// Benchmark insert operation with sequential keys
fn bench_insert_sequential(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_sequential");
    
    for size in [1000, 10_000, 100_000].iter() {
        group.bench_with_input(BenchmarkId::new("Trie", size), size, |b, &size| {
            b.iter(|| {
                let mut trie = Trie::<u64>::new();
                for i in 0..size {
                    black_box(trie.insert(i));
                }
            });
        });
        
        group.bench_with_input(BenchmarkId::new("BTreeSet", size), size, |b, &size| {
            b.iter(|| {
                let mut btree = BTreeSet::new();
                for i in 0..size {
                    black_box(btree.insert(i));
                }
            });
        });
    }
    
    group.finish();
}

/// Benchmark insert operation with clustered keys
fn bench_insert_clustered(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_clustered");
    
    // Clustered data: multiple ranges with gaps
    let clusters = vec![
        (0, 1000),
        (10_000, 11_000),
        (20_000, 21_000),
        (30_000, 31_000),
    ];
    
    group.bench_function("Trie", |b| {
        b.iter(|| {
            let mut trie = Trie::<u64>::new();
            for (start, end) in &clusters {
                for i in *start..*end {
                    black_box(trie.insert(i));
                }
            }
        });
    });
    
    group.bench_function("BTreeSet", |b| {
        b.iter(|| {
            let mut btree = BTreeSet::new();
            for (start, end) in &clusters {
                for i in *start..*end {
                    black_box(btree.insert(i));
                }
            }
        });
    });
    
    group.finish();
}

/// Benchmark insert operation with random keys
fn bench_insert_random(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_random");
    
    // Pre-generate random keys to ensure fair comparison
    let random_keys: Vec<u64> = (0..10_000)
        .map(|i| {
            // Simple LCG for reproducible "random" keys
            let a = 1664525u64;
            let c = 1013904223u64;
            a.wrapping_mul(i).wrapping_add(c)
        })
        .collect();
    
    group.bench_function("Trie", |b| {
        b.iter(|| {
            let mut trie = Trie::<u64>::new();
            for &key in &random_keys {
                black_box(trie.insert(key));
            }
        });
    });
    
    group.bench_function("BTreeSet", |b| {
        b.iter(|| {
            let mut btree = BTreeSet::new();
            for &key in &random_keys {
                black_box(btree.insert(key));
            }
        });
    });
    
    group.finish();
}

/// Benchmark contains operation
fn bench_contains(c: &mut Criterion) {
    let mut group = c.benchmark_group("contains");
    
    // Setup: insert 10k sequential keys
    let size = 10_000u64;
    
    let mut trie = Trie::<u64>::new();
    let mut btree = BTreeSet::new();
    for i in 0..size {
        trie.insert(i);
        btree.insert(i);
    }
    
    // Benchmark lookups (mix of existing and non-existing keys)
    group.bench_function("Trie_existing", |b| {
        b.iter(|| {
            for i in (0..size).step_by(10) {
                black_box(trie.contains(i));
            }
        });
    });
    
    group.bench_function("BTreeSet_existing", |b| {
        b.iter(|| {
            for i in (0..size).step_by(10) {
                black_box(btree.contains(&i));
            }
        });
    });
    
    group.bench_function("Trie_missing", |b| {
        b.iter(|| {
            for i in (size..size + 1000).step_by(10) {
                black_box(trie.contains(i));
            }
        });
    });
    
    group.bench_function("BTreeSet_missing", |b| {
        b.iter(|| {
            for i in (size..size + 1000).step_by(10) {
                black_box(btree.contains(&i));
            }
        });
    });
    
    group.finish();
}

/// Benchmark contains with clustered lookups
fn bench_contains_clustered(c: &mut Criterion) {
    let mut group = c.benchmark_group("contains_clustered");
    
    // Setup: insert clustered data
    let mut trie = Trie::<u64>::new();
    let mut btree = BTreeSet::new();
    
    for cluster_start in (0..100_000).step_by(10_000) {
        for i in cluster_start..cluster_start + 1000 {
            trie.insert(i);
            btree.insert(i);
        }
    }
    
    // Benchmark lookups in the first cluster
    group.bench_function("Trie", |b| {
        b.iter(|| {
            for i in 0..1000 {
                black_box(trie.contains(i));
            }
        });
    });
    
    group.bench_function("BTreeSet", |b| {
        b.iter(|| {
            for i in 0..1000 {
                black_box(btree.contains(&i));
            }
        });
    });
    
    group.finish();
}

/// Benchmark remove operation
fn bench_remove(c: &mut Criterion) {
    let mut group = c.benchmark_group("remove");
    
    let size = 10_000u64;
    
    // Benchmark removing sequential keys
    group.bench_function("Trie_sequential", |b| {
        b.iter_batched(
            || {
                let mut trie = Trie::<u64>::new();
                for i in 0..size {
                    trie.insert(i);
                }
                trie
            },
            |mut trie| {
                for i in 0..size {
                    black_box(trie.remove(i));
                }
            },
            criterion::BatchSize::SmallInput,
        );
    });
    
    group.bench_function("BTreeSet_sequential", |b| {
        b.iter_batched(
            || {
                let mut btree = BTreeSet::new();
                for i in 0..size {
                    btree.insert(i);
                }
                btree
            },
            |mut btree| {
                for i in 0..size {
                    black_box(btree.remove(&i));
                }
            },
            criterion::BatchSize::SmallInput,
        );
    });
    
    // Benchmark removing every other key
    group.bench_function("Trie_sparse", |b| {
        b.iter_batched(
            || {
                let mut trie = Trie::<u64>::new();
                for i in 0..size {
                    trie.insert(i);
                }
                trie
            },
            |mut trie| {
                for i in (0..size).step_by(2) {
                    black_box(trie.remove(i));
                }
            },
            criterion::BatchSize::SmallInput,
        );
    });
    
    group.bench_function("BTreeSet_sparse", |b| {
        b.iter_batched(
            || {
                let mut btree = BTreeSet::new();
                for i in 0..size {
                    btree.insert(i);
                }
                btree
            },
            |mut btree| {
                for i in (0..size).step_by(2) {
                    black_box(btree.remove(&i));
                }
            },
            criterion::BatchSize::SmallInput,
        );
    });
    
    group.finish();
}

/// Benchmark mixed workload (insert, contains, remove)
fn bench_mixed_workload(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixed_workload");
    
    let size = 10_000u64;
    
    group.bench_function("Trie", |b| {
        b.iter(|| {
            let mut trie = Trie::<u64>::new();
            
            // Insert
            for i in 0..size {
                trie.insert(i);
            }
            
            // Contains
            for i in (0..size).step_by(10) {
                black_box(trie.contains(i));
            }
            
            // Remove half
            for i in (0..size).step_by(2) {
                trie.remove(i);
            }
            
            // Insert again
            for i in (0..size).step_by(2) {
                trie.insert(i);
            }
        });
    });
    
    group.bench_function("BTreeSet", |b| {
        b.iter(|| {
            let mut btree = BTreeSet::new();
            
            // Insert
            for i in 0..size {
                btree.insert(i);
            }
            
            // Contains
            for i in (0..size).step_by(10) {
                black_box(btree.contains(&i));
            }
            
            // Remove half
            for i in (0..size).step_by(2) {
                btree.remove(&i);
            }
            
            // Insert again
            for i in (0..size).step_by(2) {
                btree.insert(i);
            }
        });
    });
    
    group.finish();
}

/// Benchmark successor operation
fn bench_successor(c: &mut Criterion) {
    let mut group = c.benchmark_group("successor");
    
    // Clustered data: keys with gaps
    let keys: Vec<u64> = (0..100_000)
        .filter(|x| x % 100 < 80) // 80% filled, 20% gaps
        .collect();
    
    // Build datasets
    let mut trie = Trie::<u64>::new();
    let mut btree = BTreeSet::new();
    for &key in &keys {
        trie.insert(key);
        btree.insert(key);
    }
    
    // Queries: find successor for keys in gaps
    let queries: Vec<u64> = (0..100_000)
        .filter(|x| x % 100 >= 80) // Query gaps
        .step_by(10)
        .collect();
    
    group.bench_function("Trie", |b| {
        b.iter(|| {
            for &query in &queries {
                black_box(trie.successor(query));
            }
        });
    });
    
    group.bench_function("BTreeSet", |b| {
        b.iter(|| {
            for &query in &queries {
                black_box(btree.range(query..).next().copied());
            }
        });
    });
    
    group.finish();
}

/// Benchmark predecessor operation
fn bench_predecessor(c: &mut Criterion) {
    let mut group = c.benchmark_group("predecessor");
    
    // Clustered data: keys with gaps
    let keys: Vec<u64> = (0..100_000)
        .filter(|x| x % 100 < 80) // 80% filled, 20% gaps
        .collect();
    
    // Build datasets
    let mut trie = Trie::<u64>::new();
    let mut btree = BTreeSet::new();
    for &key in &keys {
        trie.insert(key);
        btree.insert(key);
    }
    
    // Queries: find predecessor for keys in gaps
    let queries: Vec<u64> = (0..100_000)
        .filter(|x| x % 100 >= 80) // Query gaps
        .step_by(10)
        .collect();
    
    group.bench_function("Trie", |b| {
        b.iter(|| {
            for &query in &queries {
                black_box(trie.predecessor(query));
            }
        });
    });
    
    group.bench_function("BTreeSet", |b| {
        b.iter(|| {
            for &query in &queries {
                black_box(btree.range(..query).next_back().copied());
            }
        });
    });
    
    group.finish();
}

/// Benchmark successor/predecessor on sequential data (worst case)
fn bench_successor_sequential(c: &mut Criterion) {
    let mut group = c.benchmark_group("successor_sequential");
    
    // Sequential data: no gaps (worst case for successor - always next key)
    let mut trie = Trie::<u64>::new();
    let mut btree = BTreeSet::new();
    for i in 0..100_000 {
        trie.insert(i);
        btree.insert(i);
    }
    
    // Queries: find successor for existing keys
    let queries: Vec<u64> = (0..100_000).step_by(100).collect();
    
    group.bench_function("Trie", |b| {
        b.iter(|| {
            for &query in &queries {
                black_box(trie.successor(query));
            }
        });
    });
    
    group.bench_function("BTreeSet", |b| {
        b.iter(|| {
            for &query in &queries {
                black_box(btree.range(query..).nth(1));
            }
        });
    });
    
    group.finish();
}

/// Benchmark full iteration over all elements
fn bench_iter_full(c: &mut Criterion) {
    let mut group = c.benchmark_group("iter_full");
    
    // Build datasets with 100k elements
    let mut trie = Trie::<u64>::new();
    let mut btree = BTreeSet::new();
    for i in 0..100_000 {
        trie.insert(i);
        btree.insert(i);
    }
    
    group.bench_function("Trie", |b| {
        b.iter(|| {
            let mut count = 0u64;
            for key in trie.iter() {
                count = count.wrapping_add(black_box(key));
            }
            black_box(count)
        });
    });
    
    group.bench_function("BTreeSet", |b| {
        b.iter(|| {
            let mut count = 0u64;
            for &key in btree.iter() {
                count = count.wrapping_add(black_box(key));
            }
            black_box(count)
        });
    });
    
    group.finish();
}

/// Benchmark range queries over different range sizes
fn bench_range_queries(c: &mut Criterion) {
    let mut group = c.benchmark_group("range_queries");
    
    // Build clustered dataset: [0-10K], [20K-30K], [40K-50K], [60K-70K]
    let mut trie = Trie::<u64>::new();
    let mut btree = BTreeSet::new();
    
    for cluster_start in [0, 20_000, 40_000, 60_000] {
        for i in cluster_start..(cluster_start + 10_000) {
            trie.insert(i);
            btree.insert(i);
        }
    }
    
    // Small range (100 elements)
    group.bench_function("Trie/small_range", |b| {
        b.iter(|| {
            let mut count = 0u64;
            for key in trie.range(1000..1100) {
                count = count.wrapping_add(black_box(key));
            }
            black_box(count)
        });
    });
    
    group.bench_function("BTreeSet/small_range", |b| {
        b.iter(|| {
            let mut count = 0u64;
            for &key in btree.range(1000..1100) {
                count = count.wrapping_add(black_box(key));
            }
            black_box(count)
        });
    });
    
    // Medium range (1000 elements)
    group.bench_function("Trie/medium_range", |b| {
        b.iter(|| {
            let mut count = 0u64;
            for key in trie.range(5000..6000) {
                count = count.wrapping_add(black_box(key));
            }
            black_box(count)
        });
    });
    
    group.bench_function("BTreeSet/medium_range", |b| {
        b.iter(|| {
            let mut count = 0u64;
            for &key in btree.range(5000..6000) {
                count = count.wrapping_add(black_box(key));
            }
            black_box(count)
        });
    });
    
    // Large range (10k elements - entire cluster)
    group.bench_function("Trie/large_range", |b| {
        b.iter(|| {
            let mut count = 0u64;
            for key in trie.range(0..10_000) {
                count = count.wrapping_add(black_box(key));
            }
            black_box(count)
        });
    });
    
    group.bench_function("BTreeSet/large_range", |b| {
        b.iter(|| {
            let mut count = 0u64;
            for &key in btree.range(0..10_000) {
                count = count.wrapping_add(black_box(key));
            }
            black_box(count)
        });
    });
    
    // Cross-cluster range (spans gap)
    group.bench_function("Trie/cross_cluster", |b| {
        b.iter(|| {
            let mut count = 0u64;
            for key in trie.range(15_000..25_000) {
                count = count.wrapping_add(black_box(key));
            }
            black_box(count)
        });
    });
    
    group.bench_function("BTreeSet/cross_cluster", |b| {
        b.iter(|| {
            let mut count = 0u64;
            for &key in btree.range(15_000..25_000) {
                count = count.wrapping_add(black_box(key));
            }
            black_box(count)
        });
    });
    
    group.finish();
}

/// Benchmark range iteration on sparse data
fn bench_range_sparse(c: &mut Criterion) {
    let mut group = c.benchmark_group("range_sparse");
    
    // Sparse data: 1% filled (every 100th element)
    let mut trie = Trie::<u64>::new();
    let mut btree = BTreeSet::new();
    for i in (0..100_000).step_by(100) {
        trie.insert(i);
        btree.insert(i);
    }
    
    // Query wide range but few elements (1000 elements out of 100k range)
    group.bench_function("Trie", |b| {
        b.iter(|| {
            let mut count = 0u64;
            for key in trie.range(0..100_000) {
                count = count.wrapping_add(black_box(key));
            }
            black_box(count)
        });
    });
    
    group.bench_function("BTreeSet", |b| {
        b.iter(|| {
            let mut count = 0u64;
            for &key in btree.range(0..100_000) {
                count = count.wrapping_add(black_box(key));
            }
            black_box(count)
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_insert_sequential,
    bench_insert_clustered,
    bench_insert_random,
    bench_contains,
    bench_contains_clustered,
    bench_remove,
    bench_mixed_workload,
    bench_successor,
    bench_predecessor,
    bench_successor_sequential,
    bench_iter_full,
    bench_range_queries,
    bench_range_sparse,
);
criterion_main!(benches);
