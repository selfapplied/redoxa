# Glossary of Terms for CE1 Virtual Machine

This glossary defines key terms related to the CE1 virtual machine framework, which integrates concepts from complex analysis, signal processing, and information theory to enable reversible operations and efficient data manipulation on 64-bit data and multimodal signals.

---

## Core Concepts

#### **CE1 VM**
A virtual machine framework that uses concepts from complex analysis, information theory, and reversible computing. It operates on 64-bit data and includes components like mirrors, kernels, boundary policies, and a planner for optimizing operations based on minimum description length (MDL).

#### **Analytic Continuation**
A technique in complex analysis to extend the domain of a complex function beyond its initial region of definition. In the CE1 VM, this is analogized to choosing boundary conditions for data types (e.g., integers, floats), where different continuations represent different interpretations of data at boundaries.

#### **Boundary Conditions**
Policies that define how data behaves at the edges or boundaries of its domain. For example, in signal processing, boundary conditions determine how signals are padded beyond known values. In CE1, boundary conditions are typed and applied to ports/heaps (e.g., `int64.boundary` or `float64.boundary`).

#### **MDL (Minimum Description Length)**
A principle from information theory that measures the complexity of data representation, often using compression algorithms like zstd. In CE1, the planner minimizes MDL to choose efficient boundary policies and kernels.

#### **zstd**
A fast, lossless compression algorithm (Zstandard) used throughout the CE1 system for measuring data complexity and implementing MDL scoring. Unlike deflate (used in gzip), zstd offers better compression ratios and faster decompression speeds, making it ideal for real-time applications. The compressed size of data serves as a proxy for its description length, with smaller compressed sizes indicating more efficient representations.

#### **Energy**
In CE1, energy is defined as `Δzstd(u64) + invariant_bonus - boundary_move_cost`, where:
- `Δzstd(u64)` is the change in zstd compression size of the data after operation
- `invariant_bonus` is a reward for satisfying invariants
- `boundary_move_cost` is a cost for changing boundary policies or branch cuts

## Data Structures

#### **Heap and Stack**
Memory structures in the CE1 VM:
- **Heap**: Stores raw data as content IDs (CIDs)
- **Stack**: Stores Merkle DAGs (directed acyclic graphs) with sheet information for tracking branches and transformations

#### **Riemann Surface**
A surface that represents a multi-valued complex function by connecting its different branches (sheets). In CE1, sheet bookkeeping tracks these branches during operations, and branch cuts can be moved via BranchCutMove.

#### **Sheet Bookkeeping**
The process of tracking different sheets (branches) of multi-valued functions during operations. This involves annotating sheet IDs and managing deck transformations when branches are merged or moved.

## Operations and Kernels

#### **Mirror (Π)**
In CE1, mirrors are reversible operations or transformations that can be applied to data without loss of information. They are part of the kernel set and include operations like bit casting with boundary policies.

#### **bit_cast**
An operation that reinterprets the bits of data from one type to another (e.g., from `u64` to `f64`), often with explicit boundary policies (e.g., endianness).

#### **M_if**
A function in CE1 that performs bit casting of 64-bit data with boundary policies, such as endianness canonicalization.

#### **Convolution (Σ)**
A mathematical operation that combines two functions to produce a third function. In CE1, convolution is performed with explicit boundary policies (e.g., via AnalyticPad) to ensure reversibility or minimize artifacts.

#### **K***
A set of continuation kernels in CE1, including:
- `HilbertLift[Β=...]`
- `KKpair[Β=...]`
- `SchwarzReflect[Β=Γ]`
- `EdgeOfWedge[Β=overlap]`
- `BranchCutMove[γ→γ′]`

## Specialized Kernels

#### **HilbertLift**
A kernel that converts a real-valued signal to an analytic signal using the Hilbert transform (via FFT and phase flipping). It is reversible on band-limited windows with declared padding boundaries (e.g., periodic or mirror pad).

#### **K-K Pair**
Refers to Kramers-Kronig relations, which connect the real and imaginary parts of causal functions in the frequency domain. This kernel is used for analytic continuation in causal systems and requires a boundary policy of "causal in the upper half-plane."

#### **SchwarzReflect**
A kernel that reflects an analytic function across a boundary curve Γ using the Schwarz reflection principle. It requires that the function satisfies `f(z̄) = \overline{f(z)}` on the mirror line, which is enforced as an invariant.

#### **Edge-of-Wedge**
A kernel based on the edge-of-wedge theorem, which allows two analytic functions from different wedges (or half-planes) to be merged if they agree on an overlap region. This enables branch collapse and simplifies representations.

#### **BranchCutMove**
A kernel that changes the branch cut of a multi-valued complex function (e.g., logarithm or square root). It involves homotopy (continuous deformation) of the cut path and tracks monodromy (deck transformations) in the stack. This allows for reversible changes in function branches.

#### **MöbiusFlip**
A kernel that performs non-orientable state transport by flipping the handedness of states while preserving analyticity. Unlike orientable kernels (BranchCutMove, SchwarzReflect), MöbiusFlip introduces topological twist, enabling the planner to cross orientation boundaries and explore qualitatively different solution spaces. Incurs a "twist cost" (χ) in energy accounting and requires Orientation Consistency invariant checking to ensure the flipped state can be re-embedded without topological tearing.

## Boundary Policies

#### **AnalyticPad**
A padding method used before operations like FFT or convolution to maintain analyticity or reduce edge artifacts. Options include periodic padding, mirror padding, or anti-periodic padding. The choice of padding is a boundary policy that affects data continuity.

#### **Typed Boundary Policies**
Specific boundary policies applied to different data types in CE1. For example:
- `int64.boundary`: options include `twos_complement`, `saturate`, `wrapmod(2^k)`
- `float64.boundary`: options include `reflect`, `periodic`, `antiperiodic`, `causal`, `analytic_sheet(id)`, `branch_cut(γ)`
- `signal.boundary`: options include `periodize`, `mirror_pad`, `zero_pad`, `min_phase`

#### **Two's-complement**
A boundary convention for representing signed integers in binary, where the most significant bit indicates the sign. In CE1, this is one of several boundary policies for `int64.boundary`, along with options like saturate or wrapmod.

## Invariants and Consistency

#### **Invariants (I)**
Conditions that must be satisfied during operations to ensure correctness. In CE1, these include CR residual, reflection law, dispersion consistency, and sheet bookkeeping. Violations incur penalties in the energy accounting.

#### **CR Residual**
The residual from the Cauchy-Riemann equations, which must be zero for a function to be analytic. In CE1, this is an invariant enforced during operations to ensure analyticity. It is computed as `‖∂f/∂x − i∂f/∂y‖²` within a declared analytic region.

#### **Dispersion Consistency**
A consistency check between the real and imaginary parts of a causal function using Kramers-Kronig relations. In CE1, this invariant ensures that signals satisfy Hilbert transform pairs within a tolerance, which is crucial for causal signal processing.

## Planning and Optimization

#### **Planner**
A component in the CE1 VM that chooses the best combination of kernels and boundary policies to minimize MDL and respect invariants. It searches for paths that reduce description length while maintaining analyticity and other constraints.

#### **Energy Accounting**
A method to measure the cost or benefit of operations in terms of MDL change. It includes `Δzstd(u64)` (change in compression size), bonuses for satisfying invariants, and costs for moving boundary policies or branch cuts. Used by the planner to make decisions.

## QL-Metric Terms

#### **QL (QuantumLattice)**
The fundamental state space in the CE1 framework, representing quantum lattice states in the space 𝔹ᴺ ⊗ ℂᴹ ⊗ ℤₚᴸ. The QuantumLattice serves as the domain for gauge-invariant distance measurements and state comparisons.

#### **QL-Metric**
A gauge-invariant distance metric for comparing CE1 states that accounts for permutation (π), phase (τ), and scale (σ) symmetries. It operates on the orbit space QL/G where G is the gauge group.

#### **Canonicalization (canon!)**
A procedure to normalize any given seed into a canonical representation by applying gauge_fix to remove arbitrary effects of permutation, phase, and scale.

#### **Matching (match!)**
An algorithm to find the optimal gauge transformation (g* = (π*, τ*, σ*)) that best aligns canonicalized representations of two seeds.

#### **Residuals (residuals!)**
Quantifies the mismatch between optimally aligned seeds by calculating differences in energy, phase, frame orientation, and Mellin correlation.

#### **Witnesses (witness.metric!)**
Provides verification of metric properties including non-negativity, symmetry, and identity, with empirical triangle inequality checking.

## Redoxa Architecture Terms

#### **Three-Ring Architecture**
The core design of Redoxa consisting of three concentric rings:
- **Ring 0 (Core)**: Rust-based memory management, CID storage, WASM hosting, planning
- **Ring 1 (Kernels)**: WASM-based pure computational kernels with strict sandboxing
- **Ring 2 (Orchestrator)**: Python-based control plane, gene authoring, experiments

#### **CID (Content-Addressed Identifier)**
A SHA-256 hash-based identifier for immutable data storage. Provides deduplication, version control, and distributed caching capabilities.

#### **HeapStore**
Content-addressed storage system with zstd compression for efficient data management in the Rust core.

#### **StackDag**
Immutable Merkle DAG (directed acyclic graph) for tracking computation history with branch/merge capabilities.

#### **WasmHost**
Wasmtime integration with capability gates for hosting WASM kernels in strict sandboxed environments.

#### **Beam Search**
A* planning algorithm that maintains a beam of best candidates during search, expanding the most promising paths first.

#### **Frontier**
The current set of candidate states in the search space during planning and optimization.

## WASM Kernel Terms

#### **HilbertLift**
A WASM kernel that converts real-valued signals to complex domain using Hilbert transform or zero-padding of imaginary parts.

#### **MantissaQuant**
A WASM kernel that quantizes mantissa bits of complex floating-point numbers to reduce precision while maintaining structure.

#### **STFT/ISTFT**
Short-time Fourier transform pair kernels for time-frequency analysis with windowed boundary policies.

#### **OpticalFlowTiny**
A minimal optical flow computation kernel for spatial boundary policies.

#### **TextBertTiny**
A tiny text embedding model kernel operating on tokenized boundary policies.

## Mirror Terms

#### **Bitcast64/Bitcast32**
Reversible bit-level transformations between integer and floating-point representations (u64↔f64, u32↔f32).

#### **EndianSwap**
Reversible transformation that swaps byte order of multi-byte values.

#### **SignFlip**
Reversible transformation that flips the sign bit of floating-point numbers.

## Quantum Lattice Terms

#### **Metanion (μ)**
A 256-state basis system used in the QuantumLattice for representing states in the space 𝔹ᴺ ⊗ ℂᴹ ⊗ ℤₚᴸ. The metanion basis consists of basis states `|x⟩` for `x ∈ 𝔽₂₈` (the finite field with 2^8 = 256 elements), providing a complete orthonormal basis for the 256-dimensional state space. Topologically modeled as a torus—a closed, orientable surface that guarantees stable recurrence and bounded neutrality, perfectly suited as a neutral buffer with circulation and return paths.

#### **Metacation (κ)**
A dual object to the Metanion (μ). While the metanion is modeled topologically as a torus (an orientable buffer that recirculates energy neutrally across 256 states in 𝔽₂₈), the metacation is modeled as a Möbius strip (a non-orientable carrier with a single continuous boundary). In CE1, the metacation represents twist transport: it couples boundary policies with reversible orientation flips, serving as a charged mediator that can "hand off" state while changing its orientation. Conceptually, κ acts as the Möbius companion to μ's torus, enabling non-orientable continuation in the VM's analytic and reversible kernels.

#### **Monster Subgroup**
A mathematical group structure used for wreath product operations in the QuantumLattice, restricted to 196883-dimensional subspace.

#### **Wreath Product (≀)**
A group-theoretic operation used in Monster adjacency operations for combining group actions.

#### **TimeMirror**
An operation that reflects states across time boundaries: T ↦ -T, Φ ↦ Φ*, S ↦ S.

#### **PK-Diagonal Basis**
Projected Kravchuk basis used for canonicalization, providing orthonormal basis functions for normal form representation.

## Gauge Fixing Terms

#### **π-Gauge Fix**
Algorithm for fixing permutation gauge freedom by finding optimal permutations that minimize energy vector norms.

#### **Kravchuk Polynomials**
Orthogonal polynomials used in the PK-diagonal basis for canonicalization procedures.

#### **Energy Minimization**
The process of finding gauge transformations that minimize the ℓ₂ norm of energy vectors.

#### **Normal Form**
The canonical representation of a state after applying gauge fixing procedures.

## Witness System Terms

#### **Witness Laws**
Executable verification procedures for mathematical invariants with concrete tolerance bounds.

#### **Energy Conservation (I1)**
Invariant ensuring Σ⟨ψ|H|ψ⟩_in = Σ⟨ψ|H|ψ⟩_out within tolerance bounds.

#### **Reversibility (I2)**
Invariant ensuring all operations have inverses: ∀op∈ops, ∃op⁻¹ | op⁻¹ ∘ op = id.

#### **Simultaneity Equivalence (I3)**
Invariant ensuring [Ω, U_∆t] = 0 via commutator testing.

#### **Mass Sum Conservation (I4)**
Invariant ensuring ∑ᵢ |ψᵢ|² = constant.

#### **Phase Coherence (I5)**
Invariant ensuring arg(⟨ψ|TimeMirror|ψ⟩) ≡ 0 mod π/2.

## Boundary Policy Terms

#### **Causal Boundary**
Boundary policy ensuring temporal causality in signal processing operations.

#### **Windowed Boundary**
Boundary policy for operations on sliding windows of data.

#### **Spatial Boundary**
Boundary policy respecting spatial locality constraints.

#### **Tokenized Boundary**
Boundary policy for operations on discrete tokens or symbols.

#### **Periodic/Antiperiodic Boundary**
Boundary policies for maintaining periodicity or anti-periodicity in data.

## Demo and Experiment Terms

#### **Seedstream**
A sequence of states or data points used for testing and demonstration purposes.

#### **Barycenter Computation**
The process of computing Fréchet means or weighted averages in the state space using QL-Metric distances.

#### **Gene Authoring**
The process of designing and creating computational experiments in the orchestrator layer.

#### **Experiment Orchestration**
The coordination and execution of complex multi-step computational experiments.

#### **Ecological Backbone**
The Riemann critical line (Re(s) = 1/2) serves as the canonical axis for seed cooperation. This transforms canonicalization from a technical procedure into an ecological law where fit is measured as distance from the critical line, and synergy emerges when seeds cooperate to reduce their collective distance-to-line.

#### **Critical Line Operator**
A CE1 operator that maps "fit" to "distance from critical line" where the critical line serves as the trust surface for seed cooperation. Implements the principle that cooperation is the migration of diverse seeds into alignment with the canonical axis.

## CE1 Seed Fusion Terms

#### **CE1 Seed Fusion**
A living lattice organism that unifies oracle and planner into a single self-contained, reversible system. Implements the three-tick cycle: T (measure), S (act), Φ (re-seed).

#### **Three-Tick Cycle**
The fundamental operation cycle of the CE1 seed fusion:
- **T (measure)**: Shadow ledger → Kravchuk → Mellin → Mirror → Posterior β_t
- **S (act)**: Policy π* samples action a_t from energy-minimizing simplex  
- **Φ (re-seed)**: Prior drifts via gyroglide on S³, preserving audit trail

#### **Oracle (T-tick)**
The measurement component that extracts ledger state and applies spectral transforms to generate posterior probability distributions with confidence metrics.

#### **Planner (S-tick)**
The action component that maps posterior distributions to action probabilities and samples optimal strategies from the K1..K5 kernel registry.

#### **Reseeder (Φ-tick)**
The evolution component that applies gyroglide dynamics to update the prior state on the S³ manifold while preserving audit trail for reversibility.

#### **Living Lattice**
A self-aware system that learns and evolves through execution cycles, demonstrating genuine learning behavior with growing confidence and energy evolution.

#### **Gyroglide Dynamics**
The evolution mechanism for the prior state on the S³ manifold, preserving energy conservation while updating angles through gyroglide vectors.

#### **Energy-Minimizing Simplex**
The action selection mechanism that maps posterior distributions to action probabilities, optimizing for energy efficiency while respecting system constraints.

#### **Audit Trail**
The complete record of all three-tick cycles, preserving full reversibility by maintaining the history of prior evolution, posterior measurements, and action selections.

#### **K1..K5 Kernels**
The action registry containing five specialized kernels:
- **K1**: Compile optimization
- **K2**: Runtime optimization  
- **K3**: Retry with exponential backoff
- **K4**: Parallel execution
- **K5**: Adaptive strategy

#### **Posterior β_t**
The probability distribution generated by the oracle through spectral analysis, representing the system's current understanding of the execution state.

#### **Prior State**
The S³ manifold state (θ, φ, ψ, energy, confidence) that evolves through gyroglide dynamics, serving as the system's learned knowledge base.

#### **Mirror Operator (Ω)**
The reversible transformation that preserves energy while inverting phase, creating the mirror state from Mellin coefficients.

#### **Kravchuk Transform**
The spectral analysis step that applies Kravchuk polynomials to extract features from ledger state for posterior computation.

#### **Mellin Transform**
The spectral analysis step that applies Mellin integrals to Kravchuk coefficients, creating the foundation for mirror state computation.

---

*This glossary is part of the CE1 VM documentation and should be updated as the framework evolves.*
