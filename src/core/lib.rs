//! Redoxa Core - Pure Rust implementation
//! 
//! This is the core Rust library with no Python dependencies.
//! Python bindings are provided in the separate `redoxa-py` crate.

mod heapstore;
mod stackdag;
mod score;
mod planner;
mod wasmhost;
mod arm;
mod artifact_model;
mod certificate;

use heapstore::HeapStore;
use stackdag::StackDag;
use score::Scorer;
use planner::Planner;
use wasmhost::WasmHost;
use arm::AdaptiveResourceManager;
use certificate::{CertificateManager, MetricCertificate, ExecutionCertificate, UnifiedCertificate};

/// Main VM struct that coordinates all components
pub struct VM {
    heap: HeapStore,
    stack: StackDag,
    scorer: Scorer,
    planner: Planner,
    wasm_host: WasmHost,
    arm: AdaptiveResourceManager,
}

impl VM {
    /// Create a new VM instance
    pub fn new(db_path: Option<String>) -> Result<Self, Box<dyn std::error::Error>> {
        let heap = HeapStore::new(db_path.unwrap_or_else(|| "vm.db".to_string()))?;
        let stack = StackDag::new();
        let scorer = Scorer::new();
        let planner = Planner::new();
        let wasm_host = WasmHost::new()?;
        
        // Initialize ARM with reasonable defaults (1GB soft, 2GB hard)
        let arm = AdaptiveResourceManager::new(1024 * 1024 * 1024, 2 * 1024 * 1024 * 1024);
        
        Ok(VM {
            heap,
            stack,
            scorer,
            planner,
            wasm_host,
            arm,
        })
    }

    /// Initialize the VM with default settings
    pub fn init(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Initialize all components - for now, just return Ok
        // The individual components will be initialized as needed
        Ok(())
    }

    /// Get the heap store
    pub fn heap(&self) -> &HeapStore {
        &self.heap
    }

    /// Get the stack DAG
    pub fn stack(&self) -> &StackDag {
        &self.stack
    }

    /// Get the scorer
    pub fn scorer(&self) -> &Scorer {
        &self.scorer
    }

    /// Get the planner
    pub fn planner(&self) -> &Planner {
        &self.planner
    }

    /// Get the WASM host
    pub fn wasm_host(&self) -> &WasmHost {
        &self.wasm_host
    }

    /// Get the ARM
    pub fn arm(&self) -> &AdaptiveResourceManager {
        &self.arm
    }

    /// Get mutable references for operations
    pub fn heap_mut(&mut self) -> &mut HeapStore {
        &mut self.heap
    }

    pub fn stack_mut(&mut self) -> &mut StackDag {
        &mut self.stack
    }

    pub fn scorer_mut(&mut self) -> &mut Scorer {
        &mut self.scorer
    }

    pub fn planner_mut(&mut self) -> &mut Planner {
        &mut self.planner
    }

    pub fn wasm_host_mut(&mut self) -> &mut WasmHost {
        &mut self.wasm_host
    }

    pub fn arm_mut(&mut self) -> &mut AdaptiveResourceManager {
        &mut self.arm
    }
}

/// Core initialization function
#[cfg(feature = "standalone")]
pub fn core_init() {
    // Pure Rust initialization
    println!("Redoxa Core initialized in standalone mode");
}

#[cfg(not(feature = "standalone"))]
pub fn core_init() {
    // Same behavior; keep symbol identical
    println!("Redoxa Core initialized");
}

// Re-export key types for use by Python bindings
pub use artifact_model::ArtifactModelManager;