//! Basic usage example for clustered-fast-trie.
//!
//! This example demonstrates the core functionality of the trie data structure.

use clustered_fast_trie::Trie;

fn main() {
    println!("=== Clustered Fast Trie - Basic Usage Example ===\n");

    // Create a new trie for u64 keys
    let mut trie = Trie::<u64>::new();
    println!("Created empty trie");

    // Insert some keys
    println!("\nInserting keys: 100, 200, 150, 300");
    trie.insert(100);
    trie.insert(200);
    trie.insert(150);
    trie.insert(300);
    println!("Trie now contains {} keys", trie.len());

    // Check membership
    println!("\nMembership checks:");
    println!("  contains(150): {}", trie.contains(150));
    println!("  contains(999): {}", trie.contains(999));

    // Get min/max (O(1))
    println!("\nMin/Max (O(1)):");
    println!("  min: {:?}", trie.min());
    println!("  max: {:?}", trie.max());

    // Navigate the set
    println!("\nNavigation:");
    println!("  successor(100): {:?}", trie.successor(100));
    println!("  successor(175): {:?}", trie.successor(175));
    println!("  predecessor(200): {:?}", trie.predecessor(200));
    println!("  predecessor(175): {:?}", trie.predecessor(175));

    // Iterate in sorted order
    println!("\nIteration (sorted order):");
    print!("  Keys: ");
    for key in trie.iter() {
        print!("{} ", key);
    }
    println!();

    // Range queries
    println!("\nRange queries:");
    let range: Vec<u64> = trie.range(100..200).collect();
    println!("  range(100..200): {:?}", range);

    let range: Vec<u64> = trie.range(100..=200).collect();
    println!("  range(100..=200): {:?}", range);

    // Remove keys
    println!("\nRemoving key 150:");
    trie.remove(150);
    println!("  contains(150): {}", trie.contains(150));
    println!("  len: {}", trie.len());

    // Demonstrate with clustered data
    println!("\n=== Clustered Data Example ===\n");
    let mut clustered = Trie::<u64>::new();

    // Insert clustered ranges (simulating real-world use cases)
    println!("Inserting clustered ranges:");
    println!("  Range 1000-1099 (100 keys)");
    for i in 1000..1100 {
        clustered.insert(i);
    }

    println!("  Range 2000-2099 (100 keys)");
    for i in 2000..2100 {
        clustered.insert(i);
    }

    println!("\nClustered trie stats:");
    println!("  Total keys: {}", clustered.len());
    println!("  Min: {:?}", clustered.min());
    println!("  Max: {:?}", clustered.max());

    // Efficient iteration over clustered data
    println!("\nFirst 5 keys:");
    for (i, key) in clustered.iter().take(5).enumerate() {
        println!("  {}: {}", i + 1, key);
    }

    println!("\nLast 5 keys:");
    let all_keys: Vec<u64> = clustered.iter().collect();
    for (i, &key) in all_keys.iter().rev().take(5).rev().enumerate() {
        println!("  {}: {}", i + 1, key);
    }

    // Range query across clusters
    println!("\nRange query across gap:");
    let gap_range: Vec<u64> = clustered.range(1095..2005).collect();
    println!("  range(1095..2005) has {} keys", gap_range.len());
    println!("  First 3: {:?}", &gap_range[0..3]);
    println!("  Last 3: {:?}", &gap_range[gap_range.len() - 3..]);

    println!("\n=== Example Complete ===");
}
