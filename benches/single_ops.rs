use clustered_fast_trie::Trie;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::collections::BTreeSet;

/// Benchmark single insert operation with varying dataset sizes
fn bench_single_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_insert");

    // Test how insert performance changes as dataset grows
    for size in [100, 1_000, 10_000, 100_000].iter() {
        // Trie: insert into existing dataset
        group.bench_with_input(BenchmarkId::new("Trie", size), size, |b, &size| {
            let mut trie = Trie::<u64>::new();
            for i in 0..size {
                trie.insert(i);
            }
            let next_key = size;

            b.iter(|| {
                black_box(trie.insert(next_key));
                trie.remove(next_key); // Clean up for next iteration
            });
        });

        // BTreeSet: insert into existing dataset
        group.bench_with_input(BenchmarkId::new("BTreeSet", size), size, |b, &size| {
            let mut btree = BTreeSet::new();
            for i in 0..size {
                btree.insert(i);
            }
            let next_key = size;

            b.iter(|| {
                black_box(btree.insert(next_key));
                btree.remove(&next_key); // Clean up for next iteration
            });
        });
    }

    group.finish();
}

/// Benchmark single contains operation with varying dataset sizes
fn bench_single_contains(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_contains");

    for size in [100, 1_000, 10_000, 100_000].iter() {
        // Trie: lookup in middle of dataset
        group.bench_with_input(BenchmarkId::new("Trie_hit", size), size, |b, &size| {
            let mut trie = Trie::<u64>::new();
            for i in 0..size {
                trie.insert(i);
            }
            let lookup_key = size / 2;

            b.iter(|| black_box(trie.contains(lookup_key)));
        });

        // BTreeSet: lookup in middle of dataset
        group.bench_with_input(BenchmarkId::new("BTreeSet_hit", size), size, |b, &size| {
            let mut btree = BTreeSet::new();
            for i in 0..size {
                btree.insert(i);
            }
            let lookup_key = size / 2;

            b.iter(|| black_box(btree.contains(&lookup_key)));
        });

        // Trie: lookup miss
        group.bench_with_input(BenchmarkId::new("Trie_miss", size), size, |b, &size| {
            let mut trie = Trie::<u64>::new();
            for i in 0..size {
                trie.insert(i);
            }
            let lookup_key = size + 1000;

            b.iter(|| black_box(trie.contains(lookup_key)));
        });

        // BTreeSet: lookup miss
        group.bench_with_input(BenchmarkId::new("BTreeSet_miss", size), size, |b, &size| {
            let mut btree = BTreeSet::new();
            for i in 0..size {
                btree.insert(i);
            }
            let lookup_key = size + 1000;

            b.iter(|| black_box(btree.contains(&lookup_key)));
        });
    }

    group.finish();
}

/// Benchmark single remove operation with varying dataset sizes
fn bench_single_remove(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_remove");

    for size in [100, 1_000, 10_000, 100_000].iter() {
        // Trie: remove from middle of dataset
        group.bench_with_input(BenchmarkId::new("Trie", size), size, |b, &size| {
            b.iter_batched(
                || {
                    let mut trie = Trie::<u64>::new();
                    for i in 0..size {
                        trie.insert(i);
                    }
                    (trie, size / 2)
                },
                |(mut trie, key)| black_box(trie.remove(key)),
                criterion::BatchSize::SmallInput,
            );
        });

        // BTreeSet: remove from middle of dataset
        group.bench_with_input(BenchmarkId::new("BTreeSet", size), size, |b, &size| {
            b.iter_batched(
                || {
                    let mut btree = BTreeSet::new();
                    for i in 0..size {
                        btree.insert(i);
                    }
                    (btree, size / 2)
                },
                |(mut btree, key)| black_box(btree.remove(&key)),
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

/// Benchmark sequential insert pattern (hot path optimization)
fn bench_sequential_pattern(c: &mut Criterion) {
    let mut group = c.benchmark_group("sequential_pattern");

    // This should show Trie's advantage with sequential inserts
    group.bench_function("Trie_sequential_1000", |b| {
        b.iter(|| {
            let mut trie = Trie::<u64>::new();
            for i in 0..1000 {
                black_box(trie.insert(i));
            }
        });
    });

    group.bench_function("BTreeSet_sequential_1000", |b| {
        b.iter(|| {
            let mut btree = BTreeSet::new();
            for i in 0..1000 {
                black_box(btree.insert(i));
            }
        });
    });

    // Reverse sequential (should be similar)
    group.bench_function("Trie_reverse_1000", |b| {
        b.iter(|| {
            let mut trie = Trie::<u64>::new();
            for i in (0..1000).rev() {
                black_box(trie.insert(i));
            }
        });
    });

    group.bench_function("BTreeSet_reverse_1000", |b| {
        b.iter(|| {
            let mut btree = BTreeSet::new();
            for i in (0..1000).rev() {
                black_box(btree.insert(i));
            }
        });
    });

    group.finish();
}

/// Benchmark worst-case insert patterns
fn bench_worst_case_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("worst_case_insert");

    // Alternating pattern (no cache benefits)
    let alternating: Vec<u64> = (0..1000)
        .map(|i| if i % 2 == 0 { i / 2 } else { 500 + i / 2 })
        .collect();

    group.bench_function("Trie_alternating", |b| {
        b.iter(|| {
            let mut trie = Trie::<u64>::new();
            for &key in &alternating {
                black_box(trie.insert(key));
            }
        });
    });

    group.bench_function("BTreeSet_alternating", |b| {
        b.iter(|| {
            let mut btree = BTreeSet::new();
            for &key in &alternating {
                black_box(btree.insert(key));
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_single_insert,
    bench_single_contains,
    bench_single_remove,
    bench_sequential_pattern,
    bench_worst_case_insert,
);
criterion_main!(benches);
