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

    // Ensure special tokens occupy the lowest contiguous id range [0, N).
    normalize_special_token_ids(&mut value)?;

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

/// Rewrites the tokenizer JSON so that all special tokens are assigned the
/// first contiguous id range `[0, N)`, with the base vocabulary and any
/// non-special added tokens shifted above them.
fn normalize_special_token_ids(root: &mut Value) -> Result<()> {
    let obj = match root.as_object_mut() {
        Some(obj) => obj,
        None => {
            return Err(BbpeError::Tokenizers(
                "expected tokenizer JSON to be an object".into(),
            ))
        }
    };

    let mut added_tokens_value = match obj.remove("added_tokens") {
        Some(value) => value,
        None => return Ok(()),
    };

    let added_tokens_array = match added_tokens_value.as_array_mut() {
        Some(items) => items,
        None => return Ok(()),
    };

    if added_tokens_array.is_empty() {
        // Put the (empty) array back and return.
        obj.insert("added_tokens".to_string(), added_tokens_value);
        return Ok(());
    }

    let mut special_tokens = Vec::new();
    let mut other_tokens = Vec::new();
    for token in added_tokens_array.drain(..) {
        let is_special = token
            .get("special")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        if is_special {
            special_tokens.push(token);
        } else {
            other_tokens.push(token);
        }
    }

    if special_tokens.is_empty() {
        // No special tokens to normalise; restore original array exactly as-is.
        obj.insert("added_tokens".to_string(), added_tokens_value);
        return Ok(());
    }

    // Rebuild the added_tokens array with special tokens first occupying
    // `[0, num_special)`. Non-special tokens keep their original ids.
    let mut rebuilt = Vec::with_capacity(special_tokens.len() + other_tokens.len());
    for (idx, mut token) in special_tokens.into_iter().enumerate() {
        token["id"] = Value::from(idx as u64);
        rebuilt.push(token);
    }

    rebuilt.extend(other_tokens);

    added_tokens_value = Value::Array(rebuilt);
    obj.insert("added_tokens".to_string(), added_tokens_value);
    Ok(())
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
