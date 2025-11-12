//! Hugging Face compatible serialisation helpers built on top of `tokenizers`.

use std::fs;
use std::path::Path;

use serde_json::{self, Value};
use tokenizers::Tokenizer;

use crate::error::{BbpeError, Result};
use crate::model::BpeModel;

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
