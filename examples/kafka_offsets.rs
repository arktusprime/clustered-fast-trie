//! Kafka offset tracking example.
//!
//! This example demonstrates using the trie to track processed Kafka message offsets.
//! In a real Kafka consumer, you need to track which message offsets have been processed
//! to enable efficient exactly-once semantics and gap detection.

use clustered_fast_trie::Trie;

fn main() {
    println!("=== Kafka Offset Tracking Example ===\n");

    // Simulate a Kafka partition with processed offsets
    let mut processed_offsets = Trie::<u64>::new();

    // Simulate processing messages in order (mostly sequential)
    println!("Processing messages 1000-1099...");
    for offset in 1000..1100 {
        processed_offsets.insert(offset);
    }

    // Process another batch with a gap (simulating a skipped/failed message)
    println!("Processing messages 1101-1150 (gap at 1100)...");
    for offset in 1101..=1150 {
        processed_offsets.insert(offset);
    }

    println!("\nOffset tracking stats:");
    println!("  Total processed: {}", processed_offsets.len());
    println!("  Earliest offset: {:?}", processed_offsets.min());
    println!("  Latest offset: {:?}", processed_offsets.max());

    // Check if a specific offset was processed
    println!("\nOffset checks:");
    println!(
        "  Offset 1050 processed: {}",
        processed_offsets.contains(1050)
    );
    println!(
        "  Offset 1100 processed: {}",
        processed_offsets.contains(1100)
    );

    // Find gaps in processed offsets
    println!("\nDetecting gaps:");
    detect_gaps(&processed_offsets, 1000, 1150);

    // Find the next unprocessed offset after a given point
    println!("\nFinding next unprocessed offset:");
    if let Some(next_processed) = processed_offsets.successor(1099) {
        if next_processed > 1100 {
            println!(
                "  Gap detected! Next processed after 1099 is {}",
                next_processed
            );
            println!("  Missing offset: 1100");
        }
    }

    // Efficient range query: check processed offsets in a time window
    println!("\nRange query (offsets 1010-1020):");
    let range: Vec<u64> = processed_offsets.range(1010..=1020).collect();
    println!("  Processed offsets: {:?}", range);
    println!("  Count: {}", range.len());

    // Simulate catching up on the gap
    println!("\nProcessing missing offset 1100...");
    processed_offsets.insert(1100);

    println!(
        "  Offset 1100 now processed: {}",
        processed_offsets.contains(1100)
    );

    // Verify no gaps remain
    println!("\nVerifying no gaps in range 1000-1150:");
    detect_gaps(&processed_offsets, 1000, 1150);

    // Demonstrate idempotent processing (re-processing same offset)
    println!("\nIdempotent processing:");
    let was_new = processed_offsets.insert(1050);
    println!("  Re-inserting offset 1050: was_new={}", was_new);
    println!("  Total count unchanged: {}", processed_offsets.len());

    // Simulate cleanup: remove old processed offsets to free memory
    println!("\nCleanup old offsets:");
    let cleanup_before = 1050;
    let offsets_to_remove: Vec<u64> = processed_offsets.range(..cleanup_before).collect();

    println!(
        "  Removing {} old offsets (< {})...",
        offsets_to_remove.len(),
        cleanup_before
    );
    for offset in offsets_to_remove {
        processed_offsets.remove(offset);
    }

    println!("  Remaining offsets: {}", processed_offsets.len());
    println!("  New earliest offset: {:?}", processed_offsets.min());

    println!("\n=== Example Complete ===");
}

/// Detect and report gaps in processed offsets within a range.
fn detect_gaps(trie: &Trie<u64>, start: u64, end: u64) {
    let mut gaps = Vec::new();
    let mut current = start;

    while current <= end {
        if !trie.contains(current) {
            // Find the start of the gap
            let gap_start = current;

            // Find where the gap ends
            let gap_end = match trie.successor(current) {
                Some(next) if next <= end => next - 1,
                _ => end,
            };

            gaps.push((gap_start, gap_end));
            current = gap_end + 1;
        } else {
            current += 1;
        }
    }

    if gaps.is_empty() {
        println!("  No gaps found in range {}-{}", start, end);
    } else {
        println!("  Found {} gap(s):", gaps.len());
        for (gap_start, gap_end) in gaps {
            if gap_start == gap_end {
                println!("    Missing offset: {}", gap_start);
            } else {
                println!("    Missing offset range: {}-{}", gap_start, gap_end);
            }
        }
    }
}
