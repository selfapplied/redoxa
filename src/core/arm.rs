use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use anyhow::Result;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ArmError {
    #[error("Retreat failed: {0}")]
    RetreatFailed(String),
    #[error("QoS violation: {0}")]
    QosViolation(String),
    #[error("Memory pressure critical: {0}")]
    CriticalPressure(String),
}

/// Memory pressure levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryPressure {
    Nominal,
    Soft,
    Hard,
    Critical,
}

/// Retreat step kinds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RetreatKind {
    Precision,      // f64→f32→bf16
    Batch,          // shrink batch size
    Compress,       // toggle compression
    CacheTrim,      // drop non-critical buffers
    AlgoSwap,       // switch to memory-lean algorithm
    LaneShift,      // move to different device lane
}

/// Retreat step with expected outcomes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetreatStep {
    pub kind: RetreatKind,
    pub expected_mem_delta: i64,    // bytes (signed)
    pub expected_qos_delta: f32,    // normalized loss
    pub expected_time_delta: f32,   // slowdown/speedup factor
    pub reversible: bool,
    pub safety_note: String,
}

/// QoS bounds for a program
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QoSBounds {
    pub max_error: f32,
    pub max_latency_ms: u64,
    pub min_throughput: f32,
}

/// Memory profile for a program
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemProfile {
    pub working_set_size: usize,
    pub peak_memory: usize,
    pub burst_threshold: usize,
    pub fragmentation_factor: f32,
}

/// Program profile with retreat capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgramProfile {
    pub id: String,
    pub retreat_ladder: Vec<RetreatStep>,
    pub qos_bounds: QoSBounds,
    pub mem_profile: MemProfile,
    pub current_retreat_level: usize,
    pub retreat_debt: f32,
}

/// Resource allocation plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAllocation {
    pub program_id: String,
    pub allocated_memory: usize,
    pub device_lane: DeviceLane,
    pub preallocated_arenas: Vec<Arena>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum DeviceLane {
    CPU,
    GPU,
    Hybrid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Arena {
    pub size: usize,
    pub alignment: usize,
    pub purpose: String,
}

/// Retreat action result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetreatAction {
    pub program_id: String,
    pub step: RetreatStep,
    pub actual_mem_delta: i64,
    pub actual_qos_delta: f32,
    pub actual_time_delta: f32,
    pub success: bool,
}

/// Memory pressure with forecasting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPressureState {
    pub current_pressure: MemoryPressure,
    pub rss_usage: usize,
    pub vram_usage: usize,
    pub fragmentation: f32,
    pub gc_debt: usize,
    pub ema_forecast: f32,
    pub holt_winters_forecast: f32,
    pub headroom_per_lane: HashMap<DeviceLane, usize>,
}

/// ARM witness entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArmWitness {
    pub timestamp_ms: u64,
    pub program_id: String,
    pub retreat_step: Option<RetreatStep>,
    pub mem_usage: MemoryUsage,
    pub qos_delta: f32,
    pub time_delta: f32,
    pub pressure_level: MemoryPressure,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsage {
    pub rss: usize,
    pub vram: usize,
    pub alloc_churn: usize,
}

/// ARM certificate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArmCertificate {
    pub oom_count: u32,
    pub hard_cap_breaches: u32,
    pub qos_violations: u32,
    pub makespan_ms: u64,
    pub total_retreat_cost: f32,
    pub reversibility_list: Vec<RetreatStep>,
    pub sla_met: bool,
}

/// Program contract for retreatable programs
pub trait Retreatable {
    fn retreat_options(&self) -> Vec<RetreatStep>;
    fn apply_retreat(&mut self, step: &RetreatStep) -> Result<RetreatAction>;
    fn revert_retreat(&mut self, step: &RetreatStep) -> Result<RetreatAction>;
    fn qos_bounds(&self) -> QoSBounds;
    fn mem_profile(&self) -> MemProfile;
    fn current_retreat_level(&self) -> usize;
}

/// ARM system for adaptive resource management
pub struct AdaptiveResourceManager {
    programs: HashMap<String, ProgramProfile>,
    pressure_state: MemoryPressureState,
    soft_cap: usize,
    hard_cap: usize,
    hysteresis_margin: usize,
    witnesses: Vec<ArmWitness>,
    retreat_history: Vec<RetreatAction>,
}

impl AdaptiveResourceManager {
    pub fn new(soft_cap: usize, hard_cap: usize) -> Self {
        Self {
            programs: HashMap::new(),
            pressure_state: MemoryPressureState {
                current_pressure: MemoryPressure::Nominal,
                rss_usage: 0,
                vram_usage: 0,
                fragmentation: 0.0,
                gc_debt: 0,
                ema_forecast: 0.0,
                holt_winters_forecast: 0.0,
                headroom_per_lane: HashMap::new(),
            },
            soft_cap,
            hard_cap,
            hysteresis_margin: soft_cap / 10, // 10% margin
            witnesses: Vec::new(),
            retreat_history: Vec::new(),
        }
    }

    /// Check current memory pressure
    pub fn check_memory_pressure(&mut self) -> MemoryPressure {
        // Update pressure state with current system metrics
        self.update_pressure_state();
        
        // Determine pressure level based on watermarks
        if self.pressure_state.rss_usage >= self.hard_cap {
            self.pressure_state.current_pressure = MemoryPressure::Critical;
        } else if self.pressure_state.rss_usage >= self.soft_cap {
            self.pressure_state.current_pressure = MemoryPressure::Hard;
        } else if self.pressure_state.rss_usage >= self.soft_cap - self.hysteresis_margin {
            self.pressure_state.current_pressure = MemoryPressure::Soft;
        } else {
            self.pressure_state.current_pressure = MemoryPressure::Nominal;
        }
        
        self.pressure_state.current_pressure
    }

    /// Retreat if needed with hysteresis and fairness
    pub fn retreat_if_needed(&mut self, program_id: &str) -> Result<Option<RetreatAction>> {
        let pressure = self.check_memory_pressure();
        
        if pressure == MemoryPressure::Nominal {
            return Ok(None);
        }
        
        // Find program with highest retreat debt (fairness)
        let target_program = if let Some(profile) = self.programs.get(program_id) {
            if profile.retreat_debt > 0.0 && pressure != MemoryPressure::Critical {
                return Ok(None); // This program already has retreat debt
            }
            Some(profile.clone())
        } else {
            return Err(ArmError::RetreatFailed(format!("Program {} not found", program_id)).into());
        };
        
        let profile = target_program.unwrap();
        
        // Find next safe retreat step
        if profile.current_retreat_level >= profile.retreat_ladder.len() {
            return Err(ArmError::CriticalPressure(
                format!("Program {} has no more retreat options", program_id)
            ).into());
        }
        
        let step = &profile.retreat_ladder[profile.current_retreat_level];
        
        // Apply retreat
        let action = RetreatAction {
            program_id: program_id.to_string(),
            step: step.clone(),
            actual_mem_delta: step.expected_mem_delta,
            actual_qos_delta: step.expected_qos_delta,
            actual_time_delta: step.expected_time_delta,
            success: true,
        };
        
        // Update program profile
        if let Some(profile) = self.programs.get_mut(program_id) {
            profile.current_retreat_level += 1;
            profile.retreat_debt += step.expected_qos_delta;
        }
        
        // Record retreat
        self.retreat_history.push(action.clone());
        self.emit_witness(program_id, Some(step.clone()));
        
        Ok(Some(action))
    }

    /// Allocate resources for a program
    pub fn allocate_resources(&mut self, program_id: &str) -> Result<ResourceAllocation> {
        let profile = self.programs.get(program_id)
            .ok_or_else(|| ArmError::RetreatFailed(format!("Program {} not found", program_id)))?;
        
        // Choose device lane based on headroom
        let device_lane = if *self.pressure_state.headroom_per_lane.get(&DeviceLane::GPU)
            .unwrap_or(&0) > profile.mem_profile.working_set_size {
            DeviceLane::GPU
        } else {
            DeviceLane::CPU
        };
        
        // Preallocate arenas to prevent fragmentation
        let arenas = vec![
            Arena {
                size: profile.mem_profile.working_set_size,
                alignment: 64,
                purpose: "working_set".to_string(),
            },
            Arena {
                size: profile.mem_profile.peak_memory / 4,
                alignment: 32,
                purpose: "scratch".to_string(),
            },
        ];
        
        Ok(ResourceAllocation {
            program_id: program_id.to_string(),
            allocated_memory: profile.mem_profile.working_set_size,
            device_lane,
            preallocated_arenas: arenas,
        })
    }

    /// Register a program profile
    pub fn register_program_profile(&mut self, profile: ProgramProfile) {
        self.programs.insert(profile.id.clone(), profile);
    }

    /// Emit witness entry
    pub fn emit_witness(&mut self, program_id: &str, retreat_step: Option<RetreatStep>) {
        let witness = ArmWitness {
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            program_id: program_id.to_string(),
            retreat_step,
            mem_usage: MemoryUsage {
                rss: self.pressure_state.rss_usage,
                vram: self.pressure_state.vram_usage,
                alloc_churn: self.pressure_state.gc_debt,
            },
            qos_delta: 0.0, // Will be filled by caller
            time_delta: 0.0, // Will be filled by caller
            pressure_level: self.pressure_state.current_pressure,
        };
        
        self.witnesses.push(witness);
    }

    /// Generate ARM certificate
    pub fn generate_certificate(&self) -> ArmCertificate {
        let oom_count = if self.pressure_state.current_pressure == MemoryPressure::Critical { 1 } else { 0 };
        let hard_cap_breaches = if self.pressure_state.rss_usage >= self.hard_cap { 1 } else { 0 };
        
        let total_retreat_cost: f32 = self.retreat_history.iter()
            .map(|r| r.actual_qos_delta)
            .sum();
        
        let reversibility_list: Vec<RetreatStep> = self.retreat_history.iter()
            .filter(|r| r.step.reversible)
            .map(|r| r.step.clone())
            .collect();
        
        ArmCertificate {
            oom_count,
            hard_cap_breaches,
            qos_violations: 0, // TODO: track QoS violations
            makespan_ms: 0, // TODO: track makespan
            total_retreat_cost,
            reversibility_list,
            sla_met: oom_count == 0 && hard_cap_breaches == 0,
        }
    }

    /// Update pressure state with system metrics
    fn update_pressure_state(&mut self) {
        // TODO: Use sysinfo to get real system metrics
        // For now, use placeholder values
        self.pressure_state.rss_usage = 1024 * 1024 * 512; // 512MB placeholder
        self.pressure_state.vram_usage = 0;
        self.pressure_state.fragmentation = 0.1;
        self.pressure_state.gc_debt = 1024 * 1024; // 1MB placeholder
        
        // Simple EMA forecast (placeholder)
        self.pressure_state.ema_forecast = self.pressure_state.rss_usage as f32 * 1.1;
        self.pressure_state.holt_winters_forecast = self.pressure_state.rss_usage as f32 * 1.05;
        
        // Update headroom per lane
        self.pressure_state.headroom_per_lane.insert(
            DeviceLane::CPU,
            self.soft_cap - self.pressure_state.rss_usage
        );
        self.pressure_state.headroom_per_lane.insert(
            DeviceLane::GPU,
            1024 * 1024 * 1024 // 1GB GPU placeholder
        );
    }
}
