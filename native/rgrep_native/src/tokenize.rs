//! HF tokenization, so Go never reimplements BPE.
//!
//! Tokenizers are expensive to build (they parse the vocabulary), so they are
//! cached per path. rgrep serves one request at a time, so a single mutex is
//! enough and keeps the memory bound to the models actually in use.

use std::collections::HashMap;
use std::sync::Mutex;

use once_cell::sync::Lazy;
use tokenizers::Tokenizer;

static CACHE: Lazy<Mutex<HashMap<String, Tokenizer>>> = Lazy::new(|| Mutex::new(HashMap::new()));

/// tokenize returns ids for `text` using the tokenizer at `path`, without
/// special tokens: rgrep adds [CLS] / prefix / [SEP] itself.
pub fn tokenize(path: &str, text: &str) -> Result<Vec<u32>, String> {
    let mut cache = CACHE
        .lock()
        .map_err(|_| "tokenizer cache is poisoned".to_string())?;
    if !cache.contains_key(path) {
        let tk = Tokenizer::from_file(path).map_err(|e| format!("load tokenizer {path}: {e}"))?;
        cache.insert(path.to_string(), tk);
    }
    let tk = cache.get(path).ok_or_else(|| "tokenizer cache miss".to_string())?;
    let encoding = tk
        .encode(text, false)
        .map_err(|e| format!("tokenize with {path}: {e}"))?;
    Ok(encoding.get_ids().to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Runs only when a real tokenizer is available; the model is downloaded
    // separately, so CI without it must not fail here.
    #[test]
    fn tokenizes_when_a_tokenizer_is_configured() {
        let Ok(path) = std::env::var("RGREP_TOKENIZER") else {
            return;
        };
        let ids = tokenize(&path, "fn main() {}").unwrap();
        assert!(!ids.is_empty());
        // The cache must return the same result the second time.
        let again = tokenize(&path, "fn main() {}").unwrap();
        assert_eq!(ids, again);
    }

    #[test]
    fn missing_file_is_an_error() {
        assert!(tokenize("/nonexistent/tokenizer.json", "x").is_err());
    }
}
