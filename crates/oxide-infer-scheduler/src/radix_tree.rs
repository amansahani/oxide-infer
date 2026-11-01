use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;

/// Token sequence ID or Multimodal feature hash representation.
pub type TokenId = u32;

/// A node in the Radix Tree holding a slice of tokens and associated physical block IDs.
#[derive(Debug)]
pub struct RadixNode {
    pub token_prefix: Vec<TokenId>,
    pub block_ids: Vec<usize>,
    pub children: HashMap<TokenId, Arc<RwLock<RadixNode>>>,
    pub last_accessed: std::time::Instant,
}

impl RadixNode {
    pub fn new(token_prefix: Vec<TokenId>, block_ids: Vec<usize>) -> Self {
        Self {
            token_prefix,
            block_ids,
            children: HashMap::new(),
            last_accessed: std::time::Instant::now(),
        }
    }
}

/// SGLang-Style Pure Rust Radix Tree for KV-Cache Prefix Reuse.
pub struct RadixTree {
    root: Arc<RwLock<RadixNode>>,
}

impl Default for RadixTree {
    fn default() -> Self {
        Self::new()
    }
}

impl RadixTree {
    pub fn new() -> Self {
        Self {
            root: Arc::new(RwLock::new(RadixNode::new(Vec::new(), Vec::new()))),
        }
    }

    /// Match tokens against the tree to find the longest matching prefix with cached KV blocks.
    /// Returns (matched_tokens_count, cached_block_ids).
    pub fn match_prefix(&self, tokens: &[TokenId]) -> (usize, Vec<usize>) {
        let mut matched_count = 0;
        let mut collected_blocks = Vec::new();
        let mut current_node = Arc::clone(&self.root);

        let mut token_offset = 0;
        while token_offset < tokens.len() {
            let next_token = tokens[token_offset];
            let next_node = {
                let node_guard = current_node.read();
                node_guard.children.get(&next_token).cloned()
            };

            if let Some(child_arc) = next_node {
                let child_guard = child_arc.read();
                let prefix_len = child_guard.token_prefix.len();
                let remaining_tokens = &tokens[token_offset..];

                if remaining_tokens.starts_with(&child_guard.token_prefix) {
                    matched_count += prefix_len;
                    collected_blocks.extend_from_slice(&child_guard.block_ids);
                    token_offset += prefix_len;
                    drop(child_guard);
                    current_node = child_arc;
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        (matched_count, collected_blocks)
    }

    /// Insert a newly computed sequence of tokens and physical block IDs into the radix tree.
    pub fn insert(&self, tokens: &[TokenId], block_ids: &[usize]) {
        if tokens.is_empty() {
            return;
        }

        let mut root_guard = self.root.write();
        let first_token = tokens[0];
        let new_node = Arc::new(RwLock::new(RadixNode::new(tokens.to_vec(), block_ids.to_vec())));
        root_guard.children.insert(first_token, new_node);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_radix_tree_prefix_match() {
        let tree = RadixTree::new();
        let prompt = vec![101, 2048, 512, 102];
        let blocks = vec![1, 2];

        tree.insert(&prompt, &blocks);

        // Exact match
        let (matched, matched_blocks) = tree.match_prefix(&prompt);
        assert_eq!(matched, 4);
        assert_eq!(matched_blocks, vec![1, 2]);

        // Extended prompt sharing prefix
        let extended_prompt = vec![101, 2048, 512, 102, 999, 1000];
        let (matched_ext, matched_blocks_ext) = tree.match_prefix(&extended_prompt);
        assert_eq!(matched_ext, 4);
        assert_eq!(matched_blocks_ext, vec![1, 2]);

        // Unrelated prompt
        let unrelated = vec![999, 888];
        let (matched_unrelated, _) = tree.match_prefix(&unrelated);
        assert_eq!(matched_unrelated, 0);
    }
}
