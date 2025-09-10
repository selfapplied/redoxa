use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use sha2::{Sha256, Digest};
use zstd::encode_all;
use zstd::decode_all;
use anyhow::Result;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum HeapError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("CID not found: {0}")]
    CidNotFound(String),
}

/// Content-addressed storage with compression
pub struct HeapStore {
    db_path: String,
    cache: HashMap<String, Vec<u8>>,
}

impl HeapStore {
    pub fn new(db_path: String) -> Result<Self> {
        let mut store = HeapStore {
            db_path,
            cache: HashMap::new(),
        };
        store.load_cache()?;
        Ok(store)
    }

    /// Store bytes and return CID
    pub fn put(&mut self, bytes: &[u8]) -> Result<String> {
        let cid = self.compute_cid(bytes);
        
        // Compress and store
        let compressed = encode_all(bytes, 3)?; // Level 3 compression
        self.write_to_disk(&cid, &compressed)?;
        self.cache.insert(cid.clone(), bytes.to_vec());
        
        Ok(cid)
    }

    /// Retrieve data by CID with optional type casting
    pub fn view(&self, cid: &str, data_type: &str) -> Result<Vec<u8>> {
        // Check cache first
        if let Some(data) = self.cache.get(cid) {
            return Ok(self.cast_data(data, data_type)?);
        }

        // Load from disk
        let compressed = self.read_from_disk(cid)?;
        let data = decode_all(&compressed[..])?;
        
        Ok(self.cast_data(&data, data_type)?)
    }

    /// Compute CID (Content ID) from bytes
    fn compute_cid(&self, bytes: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        let hash = hasher.finalize();
        format!("{:x}", hash)
    }

    /// Cast data to requested type
    fn cast_data(&self, data: &[u8], data_type: &str) -> Result<Vec<u8>> {
        match data_type {
            "raw" => Ok(data.to_vec()),
            "u64" => {
                if data.len() % 8 != 0 {
                    return Err(HeapError::CidNotFound("Data not 8-byte aligned for u64".to_string()).into());
                }
                Ok(data.to_vec())
            }
            "f64" => {
                if data.len() % 8 != 0 {
                    return Err(HeapError::CidNotFound("Data not 8-byte aligned for f64".to_string()).into());
                }
                Ok(data.to_vec())
            }
            "c64" => {
                if data.len() % 16 != 0 {
                    return Err(HeapError::CidNotFound("Data not 16-byte aligned for c64".to_string()).into());
                }
                Ok(data.to_vec())
            }
            _ => Err(HeapError::CidNotFound(format!("Unknown type: {}", data_type)).into())
        }
    }

    /// Write compressed data to disk
    fn write_to_disk(&self, cid: &str, data: &[u8]) -> Result<()> {
        let path = Path::new(&self.db_path).join(format!("{}.zst", cid));
        let mut file = File::create(path)?;
        file.write_all(data)?;
        Ok(())
    }

    /// Read compressed data from disk
    fn read_from_disk(&self, cid: &str) -> Result<Vec<u8>> {
        let path = Path::new(&self.db_path).join(format!("{}.zst", cid));
        let mut file = File::open(path)?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;
        Ok(data)
    }

    /// Load existing CIDs into cache
    fn load_cache(&mut self) -> Result<()> {
        let db_dir = Path::new(&self.db_path);
        if !db_dir.exists() {
            std::fs::create_dir_all(db_dir)?;
            return Ok(());
        }

        for entry in std::fs::read_dir(db_dir)? {
            let entry = entry?;
            let path = entry.path();
            if let Some(extension) = path.extension() {
                if extension == "zst" {
                    if let Some(stem) = path.file_stem() {
                        if let Some(cid) = stem.to_str() {
                            let compressed = std::fs::read(&path)?;
                            let data = decode_all(&compressed[..])?;
                            self.cache.insert(cid.to_string(), data);
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
