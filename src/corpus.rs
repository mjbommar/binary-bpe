//! Facilities for discovering input files and loading binary corpora.

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use walkdir::WalkDir;

use crate::config::IngestConfig;
use crate::error::{BbpeError, Result};

/// Lazily streams binary corpus chunks from disk without retaining them all in memory.
pub struct BinaryChunkStream {
    files: Vec<PathBuf>,
    cfg: IngestConfig,
    file_index: usize,
    current_file: Option<File>,
    current_path: Option<PathBuf>,
    total_chunks: usize,
    total_bytes: usize,
}

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

/// Streams binary corpus chunks from disk, avoiding materialising the entire corpus at once.
///
/// The returned iterator yields `Vec<u8>` buffers according to [`IngestConfig::chunk_size`], skipping
/// empty results.  File discovery honours the same semantics as [`load_binary_corpus`].  Metadata about
/// the total number of chunks and bytes is precomputed from filesystem information so callers can set
/// up progress reporting without forcing eager loading.
pub fn stream_binary_corpus<P: AsRef<Path>>(
    inputs: &[P],
    cfg: &IngestConfig,
) -> Result<BinaryChunkStream> {
    let files = collect_paths(inputs, cfg)?;
    let mut total_chunks = 0usize;
    let mut total_bytes = 0usize;
    let chunk_size = cfg.chunk_size;

    for path in &files {
        let metadata = path
            .metadata()
            .map_err(|err| BbpeError::io(err, Some(path.clone())))?;
        let file_len = metadata.len();
        let file_len_usize = usize::try_from(file_len).map_err(|_| {
            BbpeError::InvalidConfig(format!(
                "input file {} exceeds usize::MAX ({} bytes)",
                path.display(),
                file_len
            ))
        })?;
        total_bytes = total_bytes.saturating_add(file_len_usize);

        if file_len == 0 {
            continue;
        }

        if chunk_size == 0 {
            total_chunks = total_chunks.saturating_add(1);
        } else {
            let size = u64::try_from(chunk_size).map_err(|_| {
                BbpeError::InvalidConfig(format!("chunk size {chunk_size} exceeds u64::MAX"))
            })?;
            let mut chunks = file_len / size;
            if file_len % size != 0 {
                chunks = chunks.saturating_add(1);
            }
            let chunks_usize = usize::try_from(chunks).map_err(|_| {
                BbpeError::InvalidConfig(format!(
                    "file {} produces more chunks than usize::MAX allows",
                    path.display()
                ))
            })?;
            total_chunks = total_chunks.saturating_add(chunks_usize);
        }
    }

    if total_chunks == 0 {
        return Err(BbpeError::InvalidConfig(
            "no binary data could be loaded from inputs".into(),
        ));
    }

    Ok(BinaryChunkStream {
        files,
        cfg: cfg.clone(),
        file_index: 0,
        current_file: None,
        current_path: None,
        total_chunks,
        total_bytes,
    })
}

impl BinaryChunkStream {
    /// Returns the total number of chunks that will be produced.
    pub fn total_chunks(&self) -> usize {
        self.total_chunks
    }

    /// Returns the aggregate byte count across all inputs.
    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }
}

impl Iterator for BinaryChunkStream {
    type Item = Result<Vec<u8>>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.file_index >= self.files.len() {
                return None;
            }

            if self.current_file.is_none() {
                let path = self.files[self.file_index].clone();
                match File::open(&path) {
                    Ok(file) => {
                        self.current_path = Some(path);
                        self.current_file = Some(file);
                    }
                    Err(err) => {
                        self.file_index = self.file_index.saturating_add(1);
                        return Some(Err(BbpeError::io(err, Some(path))));
                    }
                }
            }

            let chunk_size = self.cfg.chunk_size;
            if chunk_size == 0 {
                let mut file = self
                    .current_file
                    .take()
                    .expect("current file must exist when chunk_size == 0");
                let path = self
                    .current_path
                    .take()
                    .unwrap_or_else(|| self.files[self.file_index].clone());
                let mut buffer = Vec::new();
                let result = file.read_to_end(&mut buffer);
                self.file_index = self.file_index.saturating_add(1);
                match result {
                    Ok(_) => {
                        if buffer.is_empty() {
                            continue;
                        }
                        return Some(Ok(buffer));
                    }
                    Err(err) => return Some(Err(BbpeError::io(err, Some(path)))),
                }
            } else {
                let file = self
                    .current_file
                    .as_mut()
                    .expect("current file must exist when chunk_size > 0");
                let path = self
                    .current_path
                    .as_ref()
                    .cloned()
                    .unwrap_or_else(|| self.files[self.file_index].clone());

                let mut buffer = vec![0u8; chunk_size];
                match file.read(&mut buffer) {
                    Ok(0) => {
                        self.current_file = None;
                        self.current_path = None;
                        self.file_index = self.file_index.saturating_add(1);
                        continue;
                    }
                    Ok(read) => {
                        buffer.truncate(read);
                        if buffer.is_empty() {
                            continue;
                        }
                        return Some(Ok(buffer));
                    }
                    Err(err) => {
                        self.current_file = None;
                        self.current_path = None;
                        self.file_index = self.file_index.saturating_add(1);
                        return Some(Err(BbpeError::io(err, Some(path))));
                    }
                }
            }
        }
    }
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

    #[test]
    fn stream_binary_corpus_matches_loaded_sequences() {
        let dir = tempdir().expect("tempdir");
        let file = dir.path().join("data.bin");
        let bytes: Vec<u8> = (0..=15).collect();
        fs::write(&file, &bytes).expect("write data");

        let cfg = IngestConfig {
            chunk_size: 4,
            ..IngestConfig::default()
        };
        let expected = load_binary_corpus(std::slice::from_ref(&file), &cfg)
            .expect("load corpus with chunking enabled");
        let stream =
            stream_binary_corpus(&[file], &cfg).expect("stream corpus with chunking enabled");
        assert_eq!(stream.total_chunks(), expected.len());
        assert_eq!(
            stream.total_bytes(),
            expected.iter().map(|chunk| chunk.len()).sum::<usize>()
        );
        let streamed = stream
            .map(|item| item.expect("stream item"))
            .collect::<Vec<_>>();
        assert_eq!(streamed, expected);
    }
}
