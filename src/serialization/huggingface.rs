//! Hugging Face compatible serialisation helpers built on top of `tokenizers`.

use std::fs;
use std::path::Path;

use serde_json::{self, json, Value};
use tokenizers::Tokenizer;

use crate::error::{BbpeError, Result};
use crate::model::BpeModel;
use crate::special_tokens;

/// Builds a Hugging Face tokenizer from the trained model.
pub fn as_tokenizer(model: &BpeModel) -> Result<Tokenizer> {
    model.build_tokenizer()
}

/// Serialises the trained tokenizer to a JSON string compatible with Hugging Face.
pub fn tokenizer_json(model: &BpeModel, pretty: bool) -> Result<String> {
    let tokenizer = as_tokenizer(model)?;
    let raw = tokenizer
        .to_string(false)
        .map_err(|err| BbpeError::Tokenizers(err.to_string()))?;
    let mut value: Value =
        serde_json::from_str(&raw).map_err(|err| BbpeError::Tokenizers(err.to_string()))?;

    if value
        .get("decoder")
        .map_or(true, serde_json::Value::is_null)
    {
        value["decoder"] = serde_json::json!({"type": "Fuse"});
    }

    let mut added_tokens = Vec::new();
    for (idx, token) in special_tokens::leading_tokens().iter().enumerate() {
        added_tokens.push(json!({
            "id": idx as u32,
            "content": token,
            "single_word": false,
            "lstrip": false,
            "rstrip": false,
            "normalized": false,
            "special": true
        }));
    }
    let trailing_start = special_tokens::leading_tokens().len() + 256;
    for (offset, token) in model.special_tokens().iter().enumerate() {
        added_tokens.push(json!({
            "id": (trailing_start + offset) as u32,
            "content": token,
            "single_word": false,
            "lstrip": false,
            "rstrip": false,
            "normalized": false,
            "special": true
        }));
    }
    value["added_tokens"] = Value::Array(added_tokens);

    if pretty {
        serde_json::to_string_pretty(&value).map_err(|err| BbpeError::Tokenizers(err.to_string()))
    } else {
        serde_json::to_string(&value).map_err(|err| BbpeError::Tokenizers(err.to_string()))
    }
}

/// Persists the trained tokenizer as `tokenizer.json` compatible with Hugging Face tooling.
pub fn save_huggingface_tokenizer<P: AsRef<Path>>(
    model: &BpeModel,
    path: P,
    pretty: bool,
) -> Result<()> {
    let json = tokenizer_json(model, pretty)?;
    fs::write(path.as_ref(), json)
        .map_err(|err| BbpeError::io(err, Some(path.as_ref().to_path_buf())))
}

/// Loads a tokenizer.json file via the Hugging Face `tokenizers` crate.
pub fn load_tokenizer<P: AsRef<Path>>(path: P) -> Result<Tokenizer> {
    Tokenizer::from_file(path).map_err(|err| BbpeError::Tokenizers(err.to_string()))
}
