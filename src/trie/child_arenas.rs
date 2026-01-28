//! Child arenas for hierarchical arena allocation.

use crate::arena::Arena;
use crate::trie::{Leaf, Node};

/// Child arenas owned by a node at split level.
///
/// Nodes at split levels own arenas for their entire subtree,
/// enabling hierarchical arena allocation without global lookup.
///
/// # Memory Layout
/// - node_arena: Arena<Node> for internal nodes in subtree
/// - leaf_arena: Arena<Leaf> for leaf nodes in subtree
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
pub struct ChildArenas {
    /// Arena for internal nodes in this subtree
    pub node_arena: Arena<Node>,

    /// Arena for leaf nodes in this subtree
    pub leaf_arena: Arena<Leaf>,
}

impl ChildArenas {
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

impl Default for ChildArenas {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_child_arenas() {
        let arenas = ChildArenas::new();
        assert_eq!(arenas.node_arena.len(), 0);
        assert_eq!(arenas.leaf_arena.len(), 0);
    }

    #[test]
    fn test_default() {
        let arenas = ChildArenas::default();
        assert_eq!(arenas.node_arena.len(), 0);
        assert_eq!(arenas.leaf_arena.len(), 0);
    }
}
