//! Child arenas for hierarchical arena allocation.

use crate::arena::Arena;
use crate::key::TrieKey;
use crate::trie::{Leaf, Node};

/// Child arenas owned by a node at split level.
///
/// Nodes at split levels own arenas for their entire subtree,
/// enabling hierarchical arena allocation without global lookup.
///
/// # Type Parameters
/// * `K` - Key type (u32, u64, or u128)
///
/// # Memory Layout
/// - node_arena: Arena<Node> for internal nodes in subtree
/// - leaf_arena: Arena<Leaf<K>> for leaf nodes in subtree
///
/// # Ownership
/// - Created when node at split level is allocated
/// - Dropped when parent node is dropped (automatic cleanup)
/// - No reference counting needed (tree structure guarantees single owner)
///
/// # Performance
/// - O(1) arena access (direct pointer from parent node)
/// - Perfect cache locality (arena near parent node)
/// - No global lookup overhead
#[derive(Debug)]
pub struct ChildArenas<K: TrieKey> {
    /// Arena for internal nodes in this subtree
    pub node_arena: Arena<Node>,

    /// Arena for leaf nodes in this subtree
    pub leaf_arena: Arena<Leaf<K>>,
}

impl<K: TrieKey> ChildArenas<K> {
    /// Create new empty child arenas.
    ///
    /// Initializes both node and leaf arenas with default capacity.
    ///
    /// # Performance
    /// O(1) - creates empty arenas without allocation
    pub fn new() -> Self {
        Self {
            node_arena: Arena::new(),
            leaf_arena: Arena::new(),
        }
    }
}

impl<K: TrieKey> Default for ChildArenas<K> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_child_arenas() {
        let arenas = ChildArenas::<u64>::new();
        assert_eq!(arenas.node_arena.len(), 0);
        assert_eq!(arenas.leaf_arena.len(), 0);
    }

    #[test]
    fn test_default() {
        let arenas = ChildArenas::<u64>::default();
        assert_eq!(arenas.node_arena.len(), 0);
        assert_eq!(arenas.leaf_arena.len(), 0);
    }
}
