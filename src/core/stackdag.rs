use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use anyhow::Result;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Frame {
    pub id: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub step: String,
    pub boundary: Option<String>,
    pub parent: Option<String>,
    pub children: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Branch {
    pub id: String,
    pub frames: Vec<String>,
    pub score: f64,
}

/// Immutable stack DAG for tracking computation history
pub struct StackDag {
    frames: HashMap<String, Frame>,
    branches: HashMap<String, Branch>,
    current_branch: Option<String>,
}

impl StackDag {
    pub fn new() -> Self {
        StackDag {
            frames: HashMap::new(),
            branches: HashMap::new(),
            current_branch: None,
        }
    }

    /// Add a new frame to the current branch
    pub fn add_frame(&mut self, frame: Frame) -> Result<()> {
        let frame_id = frame.id.clone();
        self.frames.insert(frame_id.clone(), frame);

        if let Some(branch_id) = &self.current_branch {
            if let Some(branch) = self.branches.get_mut(branch_id) {
                branch.frames.push(frame_id);
            }
        }

        Ok(())
    }

    /// Create a new branch from current state
    pub fn branch(&mut self, branch_id: String) -> Result<()> {
        let current_frames = if let Some(current_branch) = &self.current_branch {
            self.branches.get(current_branch)
                .map(|b| b.frames.clone())
                .unwrap_or_default()
        } else {
            Vec::new()
        };

        let branch = Branch {
            id: branch_id.clone(),
            frames: current_frames,
            score: 0.0,
        };

        self.branches.insert(branch_id.clone(), branch);
        self.current_branch = Some(branch_id);
        Ok(())
    }

    /// Merge branches, keeping the best scoring one
    pub fn merge(&mut self, branch_ids: Vec<String>) -> Result<String> {
        if branch_ids.is_empty() {
            return Err(anyhow::anyhow!("No branches to merge"));
        }

        // Find best scoring branch
        let best_branch = branch_ids.iter()
            .filter_map(|id| self.branches.get(id))
            .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| anyhow::anyhow!("No valid branches found"))?;

        let merged_id = format!("merged_{}", uuid::Uuid::new_v4());
        let merged_branch = Branch {
            id: merged_id.clone(),
            frames: best_branch.frames.clone(),
            score: best_branch.score,
        };

        self.branches.insert(merged_id.clone(), merged_branch);
        self.current_branch = Some(merged_id.clone());
        Ok(merged_id)
    }

    /// Update branch score
    pub fn update_score(&mut self, branch_id: &str, score: f64) -> Result<()> {
        if let Some(branch) = self.branches.get_mut(branch_id) {
            branch.score = score;
        }
        Ok(())
    }

    /// Get current branch frames
    pub fn get_current_frames(&self) -> Vec<String> {
        if let Some(current_branch) = &self.current_branch {
            self.branches.get(current_branch)
                .map(|b| b.frames.clone())
                .unwrap_or_default()
        } else {
            Vec::new()
        }
    }

    /// Check for symmetry in computation (sign/exponent masks)
    pub fn check_symmetry(&self, frame_id: &str) -> Result<bool> {
        if let Some(frame) = self.frames.get(frame_id) {
            // Simple symmetry check: if step is a mirror operation
            Ok(frame.step.starts_with("mirror."))
        } else {
            Err(anyhow::anyhow!("Frame not found: {}", frame_id))
        }
    }

    /// Get frame by ID
    pub fn get_frame(&self, frame_id: &str) -> Option<&Frame> {
        self.frames.get(frame_id)
    }

    /// Get all branches
    pub fn get_branches(&self) -> Vec<&Branch> {
        self.branches.values().collect()
    }
}
