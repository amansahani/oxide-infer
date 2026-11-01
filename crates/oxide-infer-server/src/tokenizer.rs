use anyhow::{Context as AnyhowContext, Result};
use std::path::Path;
use tokenizers::Tokenizer as HfTokenizer;

/// High-performance text tokenizer and detokenizer powered by Hugging Face's native Rust implementation.
pub struct OxideTokenizer {
    inner: HfTokenizer,
}

impl OxideTokenizer {
    /// Load a tokenizer from a local `tokenizer.json` configuration file.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let inner = HfTokenizer::from_file(path.as_ref())
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer from file: {}", e))?;
        Ok(Self { inner })
    }

    /// Encode input text into token IDs.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Encoding error: {}", e))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode generated token IDs back into a UTF-8 string.
    pub fn decode(&self, token_ids: &[u32]) -> Result<String> {
        self.inner
            .decode(token_ids, true)
            .map_err(|e| anyhow::anyhow!("Decoding error: {}", e))
    }
}
