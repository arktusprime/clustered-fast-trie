//! Trie node structures and main API.

mod basic;
mod leaf;
mod node;
mod state;
mod trie;

pub use leaf::{pack_link, unpack_link, Leaf, EMPTY_LINK};
pub use node::Node;
pub use trie::Trie;
