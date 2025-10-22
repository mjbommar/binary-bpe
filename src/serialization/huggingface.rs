//! Hugging Face compatible serialisation helpers built on top of `tokenizers`.

use std::path::Path;

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
    tokenizer
        .to_string(pretty)
        .map_err(|err| BbpeError::Tokenizers(err.to_string()))
}

/// Persists the trained tokenizer as `tokenizer.json` compatible with Hugging Face tooling.
pub fn save_huggingface_tokenizer<P: AsRef<Path>>(
    model: &BpeModel,
    path: P,
    pretty: bool,
) -> Result<()> {
    let tokenizer = as_tokenizer(model)?;
    tokenizer
        .save(path.as_ref(), pretty)
        .map_err(|err| BbpeError::Tokenizers(err.to_string()))
}

/// Loads a tokenizer.json file via the Hugging Face `tokenizers` crate.
pub fn load_tokenizer<P: AsRef<Path>>(path: P) -> Result<Tokenizer> {
    Tokenizer::from_file(path).map_err(|err| BbpeError::Tokenizers(err.to_string()))
}
