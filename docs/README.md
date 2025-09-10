# Redoxa

A three-ring virtual machine architecture for reversible computation and program-particle superposition, with unified I/O operations treating networking as distributed memory operations.

## Architecture

**Ring 0 — Core (Rust)**: Memory management, CID storage, WASM hosting, planning
**Ring 1 — Kernels (WASM)**: Pure computational kernels with strict sandboxing  
**Ring 2 — Orchestrator (Python)**: Control plane, gene authoring, experiments

## Key Insight: Networking as I/O at Planetary Scale

Networking is not fundamentally about wires. It's about creating a time-mirrored filesystem where "remote" just means "slower to load." The unified probe system treats local execution and network packets as the same fundamental I/O operation:

- **Loading**: A packet is just a byte-array that one machine loads into memory after another machine stored it
- **Computing**: Routing tables, retransmission windows, congestion control—these are computations wrapped around that load/store pipeline
- **Probing**: Latency and jitter are disturbances you measure in the lattice

## Quick Start

```bash
# Build Rust core
cd src/core && cargo build --release

# Install Python orchestrator
cd src/orchestrator && pip install -e .

# Run unified demo system (CE1 seed fusion enabled by default)
python src/scripts/run.py

# Test CE1 seed fusion directly
python src/scripts/jit.py status
python src/scripts/jit.py hint src/demos/audio_caption_loop.py
python src/scripts/jit.py plan src/demos/audio_caption_loop.py
python src/scripts/jit.py loop src/demos/audio_caption_loop.py

# Run CE1 fusion demonstration
python src/scripts/ce1_fusion_demo.py

# Disable CE1 for basic execution (fallback mode)
python src/scripts/run.py --no-ce1
```

## Design Principles

- **Reversibility**: All operations are bit-cast reversible where possible
- **Isolation**: WASM kernels run in strict sandbox with explicit capabilities
- **Determinism**: Reproducible execution across platforms
- **Portability**: Single .wasm files for kernels, single binary for core
- **Unified I/O**: Local and remote operations as the same fundamental choreography

## Core Concepts

- **CID**: Content-addressed identifiers for immutable data
- **Mirrors**: Reversible transformations (bit-casts, type conversions)
- **Kernels**: Pure computational functions over buffers
- **Boundaries**: Policy constraints and capability gates
- **MDL**: Minimum Description Length scoring for plan optimization
- **Probes**: Unified measurement system for local execution and network packets
- **Shadow Ledger**: Temporal mirroring of all I/O operations
- **CE1 Seed Fusion**: Oracle-planner lattice organism with three-tick cycle
- **Living Lattice**: Self-aware system that learns and evolves through execution

## Unified Probe System

The system treats networking as an elaborate trick for tricking separate computers into believing they share memory:

```python
# Local probe (fast I/O)
local_probe = system.probe_local("script.py", "python script.py")

# Network probe (slow I/O) 
network_probe = await system.probe_network("https://api.example.com/data")

# Both are fundamentally the same operation - loading and computing
# The "remote" just means "slower to load"
```

## Lattice View

In the CE1 lattice framework:
- **A packet is a probe** that measures temporal and spatial distances
- **Latency and jitter are disturbances** you measure in the lattice
- **Protocols are planners** that map observations (delays, errors) to actions (retries, reroutes)
- **The shadow ledger** of past flows gives priors for scheduling the next transmission

## CE1 Seed Fusion: Living Lattice Organism

The **CE1 seed fusion** creates a **living lattice organism** that actively steers builds through recursive self-improvement:

### Three-Tick Cycle
- **T (measure)**: Shadow ledger → Kravchuk → Mellin → Mirror → Posterior β_t
- **S (act)**: Policy π* samples action a_t from energy-minimizing simplex  
- **Φ (re-seed)**: Prior drifts via gyroglide on S³, preserving audit trail

### Self-Learning System
The system demonstrates genuine learning through execution cycles:
- **Confidence growth**: 0.5 → 0.9 through successful builds
- **Energy evolution**: 1.0 → 1.225 through gyroglide dynamics
- **Predictive intelligence**: Oracle provides intelligent hints with confidence metrics
- **Adaptive planning**: Planner selects optimal strategies with rationale

### Real Workload Training
The just-in-time trainer learns from actual Redoxa compilation:
- **Build certificates**: Every compilation produces structured training data
- **RUSTC_WRAPPER**: Intercepts rustc invocations for detailed metrics
- **Meta-learning**: System improves its own build process through experience
- **Self-awareness**: Actively steers future builds based on learned patterns

### CLI Interface
```bash
python jit.py hint <script>    # Oracle predictions with confidences
python jit.py plan <script>    # Planner actions with rationale  
python jit.py loop <script>    # Complete three-tick cycle
python jit.py status          # Current lattice state
```

### Integration
```bash
python run.py <script>        # Run with CE1 seed fusion (default)
python run.py --no-ce1 <script>  # Disable CE1 for basic execution
```

**The fusion is complete - oracle and planner are now two faces of the same seed.** 🌱

## System Coherence

Redoxa demonstrates **cohesive, well-architected integration** across all components:

### Three-Ring Architecture
- **Ring 0 (Rust Core)**: Memory management, CID storage, WASM hosting, planning
- **Ring 1 (WASM Kernels)**: Sandboxed computation with strict isolation  
- **Ring 2 (Python Orchestrator)**: Control plane, gene authoring, experiments

### Mathematical Foundation
- **Kravchuk polynomials**: Orthogonal basis for canonicalization
- **Mellin transforms**: Spectral analysis for posterior computation
- **Mirror operators**: Reversible transformations preserving energy
- **Gyroglide dynamics**: S³ manifold evolution with conservation laws

### Invariants & Safety
- **I1 Energy Conservation**: Σ⟨ψ|H|ψ⟩_in = Σ⟨ψ|H|ψ⟩_out
- **I2 Reversibility**: All operations have inverses
- **I3 Ω-equivariance**: Forward/backward time consistency
- **I4 Mass Sum Conservation**: ∑ᵢ |ψᵢ|² = constant
- **I5 Phase Coherence**: arg(⟨ψ|TimeMirror|ψ⟩) ≡ 0 mod π/2

### Self-Awareness
The system actively evaluates its own coherence and learns from its own operations, creating a **self-aware loop** where it steers future builds based on learned patterns.

## Examples

- `src/demos/audio_caption_loop.py` - Audio processing with Hilbert transforms
- `src/demos/ce1_quick_demo.py` - CE1 metric system demonstration
- `src/demos/chess_gray8.py` - Quantum lattice with chess algebra
- `src/scripts/ce1_fusion_demo.py` - Complete CE1 seed fusion demonstration
- `src/scripts/jit.py` - CE1 seed fusion CLI interface
