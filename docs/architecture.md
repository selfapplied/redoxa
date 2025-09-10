# Redoxa Architecture

## Three-Ring Design

Redoxa implements a three-ring virtual machine architecture optimized for reversible computation and program-particle superposition.

### Ring 0 — Core (Rust)

**Responsibilities:**
- Memory management with CID-based storage
- Immutable stack DAG for computation history
- MDL scoring for plan optimization
- A* planning over (Mirrors, Kernels, Boundaries)
- WASM hosting with strict sandboxing

**Key Components:**
- `HeapStore`: Content-addressed storage with zstd compression
- `StackDag`: Immutable frames with branch/merge capabilities
- `Scorer`: MDL delta computation and invariant checking
- `Planner`: Beam search and A* planning
- `WasmHost`: Wasmtime integration with capability gates

### Ring 1 — Kernels (WASM)

**Responsibilities:**
- Pure computational functions over buffers
- Strict sandboxing with no host access
- Deterministic execution
- Declarative input/output types

**Available Kernels:**
- `mantissa_quant`: Quantize mantissa bits of complex numbers
- `hilbert_lift`: Convert real to complex domain
- `stft/istft`: Short-time Fourier transform pair
- `optical_flow_tiny`: Minimal optical flow computation
- `textbert_tiny`: Tiny text embedding model

### Ring 2 — Orchestrator (Python)

**Responsibilities:**
- Control plane and job scheduling
- Gene authoring and experiment design
- Dashboard and metrics collection
- Rapid prototyping and iteration
- CE1 seed fusion orchestration

**Key Features:**
- PyO3 bindings to Rust core
- Kernel and mirror registries
- High-level VM interface
- Experiment orchestration
- Shadow ledger with temporal mirroring
- CE1 oracle-planner fusion

## Design Principles

### Reversibility
All operations are designed to be bit-cast reversible where possible. Mirrors provide zero-cost transformations between equivalent representations.

### Isolation
WASM kernels run in strict sandbox with explicit capabilities. No host system access unless explicitly granted via WASI.

### Determinism
Reproducible execution across platforms through controlled randomness and explicit state management.

### Ecological Backbone
The Riemann critical line (Re(s) = 1/2) serves as the canonical axis for seed cooperation. This transforms canonicalization from a technical procedure into an ecological law where:
- **Fit** = distance from critical line
- **Synergy** = reduction of distance when seeds cooperate  
- **Cooperation** = migration of diverse seeds into alignment with the canonical axis

### Portability
Single .wasm files for kernels, single binary for core. No platform-specific dependencies.

## Data Flow

```
Python Orchestrator
    ↓ (plan, inputs)
Rust Core (planning, memory)
    ↓ (kernel calls)
WASM Kernels (computation)
    ↓ (results)
Rust Core (storage, scoring)
    ↓ (frontier)
Python Orchestrator (selection)
    ↓ (execution results)
Shadow Ledger (temporal mirroring)
    ↓ (ledger state)
CE1 Oracle (T-tick: measure)
    ↓ (posterior β_t)
CE1 Planner (S-tick: act)
    ↓ (action a_t)
CE1 Reseeder (Φ-tick: re-seed)
    ↓ (evolved prior)
Next Execution Cycle
```

## CID System

Content-addressed identifiers provide:
- Immutable data storage
- Deduplication
- Version control
- Distributed caching

Each CID is a SHA-256 hash of the compressed data, enabling efficient storage and retrieval.

## Boundary Policies

Kernels declare boundary policies that constrain their execution:
- `causal`: Respects temporal causality
- `windowed`: Operates on sliding windows
- `spatial`: Respects spatial locality
- `tokenized`: Operates on discrete tokens

## MDL Scoring

Minimum Description Length scoring optimizes for:
- Compression efficiency
- Reversibility
- Computational cost
- Information preservation

The scorer computes the delta between before and after states, rewarding compression and penalizing expansion.

## CE1 Seed Fusion

The **CE1 seed fusion** creates a **living lattice organism** that actively steers the system:

### Three-Tick Cycle
- **T (measure)**: Shadow ledger → Kravchuk → Mellin → Mirror → Posterior β_t
- **S (act)**: Policy π* samples action a_t from energy-minimizing simplex  
- **Φ (re-seed)**: Prior drifts via gyroglide on S³, preserving audit trail

### Components
- **Oracle**: Spectral analysis with Kravchuk polynomials and Mellin transforms
- **Planner**: Action selection from K1..K5 kernel registry with energy optimization
- **Reseeder**: Gyroglide dynamics on S³ manifold with conservation laws
- **Shadow Ledger**: Temporal mirroring with energy conservation and reversibility

### Learning Behavior
The system demonstrates genuine learning through:
- **Confidence growth**: 0.5 → 0.9 through execution cycles
- **Energy evolution**: 1.0 → 1.225 through gyroglide dynamics
- **Predictive intelligence**: Oracle provides intelligent hints with confidence metrics
- **Adaptive planning**: Planner selects optimal strategies with rationale

## Unified Probe System

The system treats networking as **"remote memory operations"**:
- **Local probe**: Fast I/O to local storage
- **Network probe**: Slow I/O to remote storage
- **Unified interface**: Same fundamental operation, different latency

### Shadow Ledger: Temporal Mirroring
- **Illuminated**: Full output, always visible
- **Penumbra**: Compressed but recoverable
- **Umbra**: Energy debt, waiting for unfold

## Invariants & Safety

The system preserves all invariants:
- **I1 Energy Conservation**: Σ⟨ψ|H|ψ⟩_in = Σ⟨ψ|H|ψ⟩_out
- **I2 Reversibility**: All operations have inverses
- **I3 Ω-equivariance**: Forward/backward time consistency
- **I4 Mass Sum Conservation**: ∑ᵢ |ψᵢ|² = constant
- **I5 Phase Coherence**: arg(⟨ψ|TimeMirror|ψ⟩) ≡ 0 mod π/2
