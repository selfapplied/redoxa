use anyhow::Result;
use zstd::encode_all;

/// MDL (Minimum Description Length) scorer for plan optimization
pub struct Scorer {
    compression_level: i32,
}

impl Scorer {
    pub fn new() -> Self {
        Scorer {
            compression_level: 3,
        }
    }

    /// Score the difference between before and after states using MDL
    pub fn score(&self, before: &[Vec<u8>], after: &[Vec<u8>]) -> Result<f64> {
        let before_size = self.compute_description_length(before)?;
        let after_size = self.compute_description_length(after)?;
        
        // MDL delta: negative means compression (good), positive means expansion (bad)
        let delta = after_size - before_size;
        Ok(delta)
    }

    /// Compute description length using zstd compression
    fn compute_description_length(&self, data: &[Vec<u8>]) -> Result<f64> {
        let mut total_size = 0.0;
        
        for chunk in data {
            // Compress the data to get its description length
            let compressed = encode_all(chunk.as_slice(), self.compression_level)?;
            total_size += compressed.len() as f64;
        }
        
        Ok(total_size)
    }

    /// Check bit-level mirror invariants
    pub fn check_mirror_invariant(&self, original: &[u8], _transformed: &[u8], back: &[u8]) -> Result<bool> {
        // Check if original == back (reversibility)
        Ok(original == back)
    }

    /// Check CR (Compression Ratio) residual hooks
    pub fn check_cr_residual(&self, data: &[u8]) -> Result<f64> {
        let original_size = data.len() as f64;
        let compressed = encode_all(data, self.compression_level)?;
        let compressed_size = compressed.len() as f64;
        
        // Return compression ratio (lower is better)
        Ok(compressed_size / original_size)
    }

    /// Score a computation step
    pub fn score_step(&self, step: &str, input_size: usize, output_size: usize) -> f64 {
        let size_ratio = output_size as f64 / input_size as f64;
        
        // Penalize expansion, reward compression
        let size_score = -size_ratio.log2();
        
        // Add step complexity penalty
        let complexity_penalty = match step {
            s if s.starts_with("mirror.") => 0.0, // Mirrors are free
            s if s.starts_with("kernel.") => 1.0, // Kernels have cost
            _ => 2.0, // Unknown steps are expensive
        };
        
        size_score - complexity_penalty
    }
}
