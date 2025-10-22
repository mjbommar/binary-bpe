//! Facilities for discovering input files and loading binary corpora.

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use walkdir::WalkDir;

use crate::config::IngestConfig;
use crate::error::{BbpeError, Result};

/// Discovers files rooted at the provided input paths according to the ingest configuration.
///
/// Directories are traversed recursively by default; set [`IngestConfig::recursive`] to `false`
/// to limit discovery to the first level.  Symlink traversal is controlled through
/// [`IngestConfig::follow_symlinks`].
pub fn collect_paths<P: AsRef<Path>>(inputs: &[P], cfg: &IngestConfig) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for input in inputs {
        let path = input.as_ref();
        if !path.exists() {
            return Err(BbpeError::InvalidConfig(format!(
                "input path {path:?} does not exist"
            )));
        }
        let metadata = path
            .symlink_metadata()
            .map_err(|err| BbpeError::io(err, Some(path.to_path_buf())))?;
        if metadata.is_dir() {
            if cfg.recursive {
                let walker = WalkDir::new(path).follow_links(cfg.follow_symlinks);
                for entry in walker {
                    let entry = entry.map_err(|err| BbpeError::Internal(err.to_string()))?;
                    if entry.file_type().is_file() {
                        files.push(entry.path().to_path_buf());
                    }
                }
            } else {
                for entry in std::fs::read_dir(path)
                    .map_err(|err| BbpeError::io(err, Some(path.to_path_buf())))?
                {
                    let entry =
                        entry.map_err(|err| BbpeError::io(err, Some(path.to_path_buf())))?;
                    let entry_path = entry.path();
                    if entry_path.is_file() {
                        files.push(entry_path);
                    }
                }
            }
        } else if metadata.is_file() {
            files.push(path.to_path_buf());
        }
    }
    if files.is_empty() {
        return Err(BbpeError::InvalidConfig(
            "no files discovered in provided inputs".into(),
        ));
    }
    Ok(files)
}

/// Loads binary corpora into memory chunks based on the ingest configuration.
///
/// Files are loaded in-order, optionally split into fixed-size chunks configured via
/// [`IngestConfig::chunk_size`].  Empty chunks are discarded to avoid degenerate training input.
pub fn load_binary_corpus<P: AsRef<Path>>(
    inputs: &[P],
    cfg: &IngestConfig,
) -> Result<Vec<Vec<u8>>> {
    let file_paths = collect_paths(inputs, cfg)?;
    let mut sequences = Vec::new();
    for file_path in file_paths {
        let mut file =
            File::open(&file_path).map_err(|err| BbpeError::io(err, Some(file_path.clone())))?;
        if cfg.chunk_size == 0 {
            let mut buffer = Vec::new();
            file.read_to_end(&mut buffer)
                .map_err(|err| BbpeError::io(err, Some(file_path.clone())))?;
            if !buffer.is_empty() {
                sequences.push(buffer);
            }
            continue;
        }

        loop {
            let mut buffer = vec![0u8; cfg.chunk_size];
            let read = file
                .read(&mut buffer)
                .map_err(|err| BbpeError::io(err, Some(file_path.clone())))?;
            if read == 0 {
                break;
            }
            buffer.truncate(read);
            sequences.push(buffer);
        }
    }
    if sequences.is_empty() {
        return Err(BbpeError::InvalidConfig(
            "no binary data could be loaded from inputs".into(),
        ));
    }
    Ok(sequences)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn collect_paths_discovers_files_recursively() {
        let dir = tempdir().expect("tempdir");
        let nested = dir.path().join("nested");
        fs::create_dir(&nested).expect("create nested directory");
        let file_a = dir.path().join("a.bin");
        let file_b = nested.join("b.bin");
        fs::write(&file_a, [1u8, 2, 3]).expect("write a");
        fs::write(&file_b, [4u8, 5, 6]).expect("write b");

        let cfg = IngestConfig {
            recursive: true,
            ..IngestConfig::default()
        };
        let mut paths = collect_paths(&[dir.path()], &cfg).expect("collect paths");
        paths.sort();
        assert_eq!(paths, vec![file_a, file_b]);
    }

    #[test]
    fn load_binary_corpus_splits_chunks() {
        let dir = tempdir().expect("tempdir");
        let file = dir.path().join("data.bin");
        let bytes: Vec<u8> = (0..=9).collect();
        fs::write(&file, &bytes).expect("write data");

        let cfg = IngestConfig {
            chunk_size: 4,
            ..IngestConfig::default()
        };
        let sequences =
            load_binary_corpus(&[file], &cfg).expect("load corpus with chunking enabled");
        assert_eq!(sequences.len(), 3);
        assert_eq!(sequences[0], vec![0, 1, 2, 3]);
        assert_eq!(sequences[1], vec![4, 5, 6, 7]);
        assert_eq!(sequences[2], vec![8, 9]);
    }

    #[test]
    fn load_binary_corpus_entire_file_when_chunk_zero() {
        let dir = tempdir().expect("tempdir");
        let file = dir.path().join("data.bin");
        let bytes: Vec<u8> = (0..=9).collect();
        fs::write(&file, &bytes).expect("write data");

        let cfg = IngestConfig {
            chunk_size: 0,
            ..IngestConfig::default()
        };
        let sequences = load_binary_corpus(&[file], &cfg).expect("load corpus without chunking");
        assert_eq!(sequences, vec![bytes]);
    }
}
