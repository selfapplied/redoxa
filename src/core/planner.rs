use std::collections::VecDeque;
use anyhow::Result;
use crate::heapstore::HeapStore;
use crate::stackdag::StackDag;
use crate::score::Scorer;
use crate::wasmhost::WasmHost;

/// Execution strategy for operational planning
#[derive(Debug, Clone)]
pub struct ExecutionStrategy {
    pub max_workers: usize,
    pub timeout_seconds: u64,
    pub retry_policy: RetryPolicy,
    pub batching: BatchingStrategy,
    pub device_pinning: DevicePinning,
}

#[derive(Debug, Clone)]
pub enum RetryPolicy {
    None,
    ExponentialBackoff { max_retries: usize, base_delay_ms: u64 },
    LinearBackoff { max_retries: usize, delay_ms: u64 },
}

#[derive(Debug, Clone)]
pub enum BatchingStrategy {
    None,
    ByResource { cpu_only: bool, gpu_only: bool },
    BySize { small: usize, large: usize },
}

#[derive(Debug, Clone)]
pub enum DevicePinning {
    Any,
    CPUOnly,
    GPUOnly,
    Specific { device_id: usize },
}

/// Operational state for CE1 search
#[derive(Debug, Clone)]
pub struct OperationalState {
    pub queue: Vec<DemoTask>,
    pub strategy: ExecutionStrategy,
    pub caps: ResourceCaps,
    pub profile: PerformanceProfile,
    pub ledger: ExecutionLedger,
}

#[derive(Debug, Clone)]
pub struct DemoTask {
    pub id: String,
    pub script_path: String,
    pub estimated_duration_ms: u64,
    pub resource_requirements: ResourceRequirements,
    pub flakiness_score: f64,
}

#[derive(Debug, Clone)]
pub struct ResourceRequirements {
    pub cpu_intensive: bool,
    pub gpu_required: bool,
    pub memory_mb: usize,
    pub io_intensive: bool,
}

#[derive(Debug, Clone)]
pub struct ResourceCaps {
    pub cpu_cores: usize,
    pub memory_mb: usize,
    pub gpu_slots: usize,
    pub io_bandwidth_mbps: f64,
}

#[derive(Debug, Clone)]
pub struct PerformanceProfile {
    pub avg_cpu_usage: f64,
    pub avg_memory_usage: f64,
    pub avg_gpu_usage: f64,
    pub avg_io_wait: f64,
    pub retry_rate: f64,
}

#[derive(Debug, Clone)]
pub struct ExecutionLedger {
    pub witnesses: Vec<ExecutionWitness>,
    pub total_makespan_ms: u64,
    pub total_cpu_usage: f64,
    pub peak_memory_mb: usize,
    pub total_gpu_usage: f64,
    pub total_io_wait_ms: u64,
    pub total_retry_cost: f64,
}

#[derive(Debug, Clone)]
pub struct ExecutionWitness {
    pub demo_id: String,
    pub start_time_ms: u64,
    pub end_time_ms: u64,
    pub cpu_usage: f64,
    pub memory_peak_mb: usize,
    pub gpu_utilization: f64,
    pub io_wait_ms: u64,
    pub exit_code: i32,
    pub retry_count: usize,
}

/// Cost functional weights for operational planning
#[derive(Debug, Clone)]
pub struct CostWeights {
    pub makespan: f64,
    pub cpu_usage: f64,
    pub memory_peak: f64,
    pub gpu_hot: f64,
    pub io_wait: f64,
    pub retry_cost: f64,
}

impl Default for CostWeights {
    fn default() -> Self {
        Self {
            makespan: 1.0,
            cpu_usage: 0.3,
            memory_peak: 0.2,
            gpu_hot: 0.4,
            io_wait: 0.1,
            retry_cost: 0.5,
        }
    }
}

/// A* planner over (Mirrors, Kernels, Boundaries) and operational execution strategies
pub struct Planner {
    available_steps: Vec<String>,
    cost_weights: CostWeights,
}

impl Planner {
    pub fn new() -> Self {
        Planner {
            available_steps: vec![
                "mirror.bitcast64".to_string(),
                "mirror.bitcast32".to_string(),
                "kernel.hilbert_lift".to_string(),
                "kernel.mantissa_quant".to_string(),
                "kernel.stft".to_string(),
                "kernel.istft".to_string(),
            ],
            cost_weights: CostWeights::default(),
        }
    }

    /// Set cost weights for operational planning
    pub fn set_cost_weights(&mut self, weights: CostWeights) {
        self.cost_weights = weights;
    }

    /// Compute cost functional J(σ) for operational state
    pub fn compute_cost(&self, state: &OperationalState) -> f64 {
        let ledger = &state.ledger;
        let caps = &state.caps;
        
        // Normalize metrics against capacity
        let makespan_norm = ledger.total_makespan_ms as f64 / 1000.0; // seconds
        let cpu_norm = ledger.total_cpu_usage / caps.cpu_cores as f64;
        let memory_norm = ledger.peak_memory_mb as f64 / caps.memory_mb as f64;
        let gpu_norm = ledger.total_gpu_usage / caps.gpu_slots as f64;
        let io_norm = ledger.total_io_wait_ms as f64 / 1000.0; // seconds
        let retry_norm = ledger.total_retry_cost;
        
        // Weighted sum
        self.cost_weights.makespan * makespan_norm +
        self.cost_weights.cpu_usage * cpu_norm +
        self.cost_weights.memory_peak * memory_norm +
        self.cost_weights.gpu_hot * gpu_norm +
        self.cost_weights.io_wait * io_norm +
        self.cost_weights.retry_cost * retry_norm
    }

    /// Generate operational planning moves (operators)
    pub fn generate_operational_moves(&self, state: &OperationalState) -> Vec<OperationalMove> {
        let mut moves = Vec::new();
        
        // Scale workers
        if state.strategy.max_workers > 1 {
            moves.push(OperationalMove::ScaleWorkers(state.strategy.max_workers - 1));
        }
        if state.strategy.max_workers < state.caps.cpu_cores {
            moves.push(OperationalMove::ScaleWorkers(state.strategy.max_workers + 1));
        }
        
        // Reorder queue
        moves.push(OperationalMove::ReorderBySize);
        moves.push(OperationalMove::ReorderByCriticalPath);
        moves.push(OperationalMove::ReorderByDeviceAffinity);
        
        // Batch strategies
        moves.push(OperationalMove::BatchByResource);
        moves.push(OperationalMove::BatchBySize);
        
        // Timeout adjustments
        if state.strategy.timeout_seconds > 30 {
            moves.push(OperationalMove::AdjustTimeout(state.strategy.timeout_seconds - 30));
        }
        moves.push(OperationalMove::AdjustTimeout(state.strategy.timeout_seconds + 30));
        
        // Device pinning
        moves.push(OperationalMove::PinToCPU);
        moves.push(OperationalMove::PinToGPU);
        
        // Retry policy changes
        moves.push(OperationalMove::SetRetryPolicy(RetryPolicy::ExponentialBackoff { 
            max_retries: 3, 
            base_delay_ms: 1000 
        }));
        
        // ARM retreat operators for each program in queue
        for task in &state.queue {
            moves.push(OperationalMove::RetreatPrecision(task.id.clone()));
            moves.push(OperationalMove::RetreatBatch(task.id.clone()));
            moves.push(OperationalMove::RetreatCompress(task.id.clone()));
            moves.push(OperationalMove::RetreatCacheTrim(task.id.clone()));
            moves.push(OperationalMove::RetreatAlgoSwap(task.id.clone()));
            moves.push(OperationalMove::RetreatLaneShift(task.id.clone()));
        }
        
        moves
    }

    /// Apply operational move to state
    pub fn apply_operational_move(&self, state: &OperationalState, mv: &OperationalMove) -> OperationalState {
        let mut new_state = state.clone();
        
        match mv {
            OperationalMove::ScaleWorkers(workers) => {
                new_state.strategy.max_workers = *workers;
            }
            OperationalMove::ReorderBySize => {
                new_state.queue.sort_by(|a, b| a.estimated_duration_ms.cmp(&b.estimated_duration_ms));
            }
            OperationalMove::ReorderByCriticalPath => {
                new_state.queue.sort_by(|a, b| b.resource_requirements.memory_mb.cmp(&a.resource_requirements.memory_mb));
            }
            OperationalMove::ReorderByDeviceAffinity => {
                new_state.queue.sort_by(|a, b| {
                    let a_gpu = if a.resource_requirements.gpu_required { 1 } else { 0 };
                    let b_gpu = if b.resource_requirements.gpu_required { 1 } else { 0 };
                    b_gpu.cmp(&a_gpu)
                });
            }
            OperationalMove::BatchByResource => {
                new_state.strategy.batching = BatchingStrategy::ByResource { 
                    cpu_only: true, 
                    gpu_only: false 
                };
            }
            OperationalMove::BatchBySize => {
                new_state.strategy.batching = BatchingStrategy::BySize { 
                    small: 1000, 
                    large: 10000 
                };
            }
            OperationalMove::AdjustTimeout(timeout) => {
                new_state.strategy.timeout_seconds = *timeout;
            }
            OperationalMove::PinToCPU => {
                new_state.strategy.device_pinning = DevicePinning::CPUOnly;
            }
            OperationalMove::PinToGPU => {
                new_state.strategy.device_pinning = DevicePinning::GPUOnly;
            }
            OperationalMove::SetRetryPolicy(policy) => {
                new_state.strategy.retry_policy = policy.clone();
            }
            // ARM retreat operators - these modify the queue tasks' resource requirements
            OperationalMove::RetreatPrecision(program_id) => {
                if let Some(task) = new_state.queue.iter_mut().find(|t| t.id == *program_id) {
                    // Reduce memory requirements by 50% for precision retreat
                    task.resource_requirements.memory_mb = (task.resource_requirements.memory_mb as f64 * 0.5) as usize;
                }
            }
            OperationalMove::RetreatBatch(program_id) => {
                if let Some(task) = new_state.queue.iter_mut().find(|t| t.id == *program_id) {
                    // Reduce memory requirements by 25% for batch retreat
                    task.resource_requirements.memory_mb = (task.resource_requirements.memory_mb as f64 * 0.75) as usize;
                    // Increase estimated duration by 20%
                    task.estimated_duration_ms = (task.estimated_duration_ms as f64 * 1.2) as u64;
                }
            }
            OperationalMove::RetreatCompress(program_id) => {
                if let Some(task) = new_state.queue.iter_mut().find(|t| t.id == *program_id) {
                    // Reduce memory requirements by 12.5% for compression retreat
                    task.resource_requirements.memory_mb = (task.resource_requirements.memory_mb as f64 * 0.875) as usize;
                    // Increase estimated duration by 10%
                    task.estimated_duration_ms = (task.estimated_duration_ms as f64 * 1.1) as u64;
                }
            }
            OperationalMove::RetreatCacheTrim(program_id) => {
                if let Some(task) = new_state.queue.iter_mut().find(|t| t.id == *program_id) {
                    // Reduce memory requirements by 12.5% for cache trim retreat
                    task.resource_requirements.memory_mb = (task.resource_requirements.memory_mb as f64 * 0.875) as usize;
                    // Increase estimated duration by 10%
                    task.estimated_duration_ms = (task.estimated_duration_ms as f64 * 1.1) as u64;
                }
            }
            OperationalMove::RetreatAlgoSwap(program_id) => {
                if let Some(task) = new_state.queue.iter_mut().find(|t| t.id == *program_id) {
                    // Reduce memory requirements by 30% for algorithm swap retreat
                    task.resource_requirements.memory_mb = (task.resource_requirements.memory_mb as f64 * 0.7) as usize;
                    // Increase estimated duration by 50%
                    task.estimated_duration_ms = (task.estimated_duration_ms as f64 * 1.5) as u64;
                }
            }
            OperationalMove::RetreatLaneShift(program_id) => {
                if let Some(task) = new_state.queue.iter_mut().find(|t| t.id == *program_id) {
                    // Switch from GPU to CPU (reduce GPU requirement)
                    task.resource_requirements.gpu_required = false;
                    // Increase estimated duration by 30% for CPU execution
                    task.estimated_duration_ms = (task.estimated_duration_ms as f64 * 1.3) as u64;
                }
            }
        }
        
        new_state
    }

    /// Operational planning search using beam search
    pub fn plan_operational(&self, initial_state: OperationalState, beam_width: usize, max_iterations: usize) -> Result<(OperationalState, f64)> {
        let mut beam = VecDeque::new();
        beam.push_back((initial_state, 0.0));
        
        let mut best_cost = f64::INFINITY;
        let mut best_state = None;
        
        for _ in 0..max_iterations {
            let mut new_beam = VecDeque::new();
            
            for (state, _) in beam.iter() {
                let moves = self.generate_operational_moves(state);
                
                for mv in moves {
                    let new_state = self.apply_operational_move(state, &mv);
                    let cost = self.compute_cost(&new_state);
                    
                    new_beam.push_back((new_state, cost));
                }
            }
            
            // Sort by cost and keep top beam_width
            new_beam.make_contiguous().sort_by(|a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            
            beam.clear();
            for (state, cost) in new_beam.iter().take(beam_width) {
                beam.push_back((state.clone(), *cost));
                
                if *cost < best_cost {
                    best_cost = *cost;
                    best_state = Some(state.clone());
                }
            }
            
            // Early termination if no improvement
            if beam.is_empty() {
                break;
            }
        }
        
        Ok((best_state.unwrap_or_else(|| beam[0].0.clone()), best_cost))
    }

    /// Tick the frontier with beam search
    pub fn tick(
        &self,
        heap: &mut HeapStore,
        _stack: &mut StackDag,
        scorer: &Scorer,
        _wasm_host: &mut WasmHost,
        frontier: Vec<String>,
        beam: usize,
    ) -> Result<Vec<String>> {
        let mut candidates = VecDeque::new();
        
        // Initialize with current frontier
        for cid in frontier {
            candidates.push_back((cid, 0.0)); // (cid, score)
        }

        let mut new_frontier = Vec::new();
        
        // Beam search: expand best candidates
        for _ in 0..beam {
            if let Some((best_cid, _)) = candidates.pop_front() {
                // Try all available steps
                for step in &self.available_steps {
                    if let Ok(outputs) = self.try_step(heap, step, &best_cid) {
                        for output_cid in outputs {
                            // Score the transformation
                            let before_data = heap.view(&best_cid, "raw")?;
                            let after_data = heap.view(&output_cid, "raw")?;
                            let score = scorer.score(&[before_data], &[after_data])?;
                            
                            candidates.push_back((output_cid, score));
                        }
                    }
                }
                
                new_frontier.push(best_cid);
            }
        }

        // Sort by score (best first)
        candidates.make_contiguous().sort_by(|a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Add best candidates to frontier
        for (cid, _) in candidates.iter().take(beam) {
            new_frontier.push(cid.clone());
        }

        Ok(new_frontier)
    }

    /// Try applying a step to a CID
    fn try_step(&self, heap: &mut HeapStore, step: &str, input_cid: &str) -> Result<Vec<String>> {
        let input_data = heap.view(input_cid, "raw")?;
        
        let outputs = match step {
            "mirror.bitcast64" => {
                if input_data.len() % 8 != 0 {
                    return Ok(vec![]);
                }
                // Simple bitcast: u64 -> f64 -> u64 (reversible)
                let mut output = Vec::new();
                for chunk in input_data.chunks(8) {
                    let u64_val = u64::from_le_bytes(chunk.try_into().unwrap());
                    let f64_val = f64::from_bits(u64_val);
                    output.extend_from_slice(&f64_val.to_le_bytes());
                }
                vec![heap.put(&output)?]
            }
            "kernel.hilbert_lift" => {
                if input_data.len() % 8 != 0 {
                    return Ok(vec![]);
                }
                // Simple complex lift: f64 -> c64 (real -> complex)
                let mut output = Vec::new();
                for chunk in input_data.chunks(8) {
                    let real = f64::from_le_bytes(chunk.try_into().unwrap());
                    let imag: f64 = 0.0; // Zero imaginary part
                    output.extend_from_slice(&real.to_le_bytes());
                    output.extend_from_slice(&imag.to_le_bytes());
                }
                vec![heap.put(&output)?]
            }
            "kernel.mantissa_quant" => {
                if input_data.len() % 16 != 0 {
                    return Ok(vec![]);
                }
                // Quantize mantissa bits (simplified)
                let mut output = Vec::new();
                for chunk in input_data.chunks(16) {
                    let real_bytes: [u8; 8] = chunk[0..8].try_into().unwrap();
                    let imag_bytes: [u8; 8] = chunk[8..16].try_into().unwrap();
                    
                    let mut real = f64::from_le_bytes(real_bytes);
                    let mut imag = f64::from_le_bytes(imag_bytes);
                    
                    // Simple quantization: round to nearest power of 2
                    real = real.signum() * 2_f64.powf(real.abs().log2().round());
                    imag = imag.signum() * 2_f64.powf(imag.abs().log2().round());
                    
                    output.extend_from_slice(&real.to_le_bytes());
                    output.extend_from_slice(&imag.to_le_bytes());
                }
                vec![heap.put(&output)?]
            }
            _ => vec![],
        };

        Ok(outputs)
    }

    /// Plan a sequence of steps to reach a goal
    pub fn plan(&self, _start_cids: Vec<String>, goal_description: &str) -> Result<Vec<String>> {
        // Simplified planning: just return a basic sequence
        // In full implementation, this would use A* search
        match goal_description {
            "audio_to_complex" => Ok(vec![
                "mirror.bitcast64".to_string(),
                "kernel.hilbert_lift".to_string(),
            ]),
            "quantize_complex" => Ok(vec![
                "kernel.mantissa_quant".to_string(),
            ]),
            _ => Ok(vec!["mirror.bitcast64".to_string()]),
        }
    }
}

/// Operational planning moves (operators)
#[derive(Debug, Clone)]
pub enum OperationalMove {
    ScaleWorkers(usize),
    ReorderBySize,
    ReorderByCriticalPath,
    ReorderByDeviceAffinity,
    BatchByResource,
    BatchBySize,
    AdjustTimeout(u64),
    PinToCPU,
    PinToGPU,
    SetRetryPolicy(RetryPolicy),
    // ARM retreat operators
    RetreatPrecision(String),  // program_id
    RetreatBatch(String),      // program_id
    RetreatCompress(String),   // program_id
    RetreatCacheTrim(String),  // program_id
    RetreatAlgoSwap(String),   // program_id
    RetreatLaneShift(String),  // program_id
}
