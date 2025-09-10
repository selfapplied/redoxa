# QL-Metric: Gauge-Invariant Distance for CE1 States

The QL-Metric is a distance metric designed for comparing two canonicalized "seeds" or states within the CE1 framework, here denoted as A and B. It's a "quotient metric" because it measures the distance between the equivalence classes of states, accounting for the gauge symmetries of permutation (π), phase (τ), and scale (σ).

## Key Components

### Gauge Group (G)
The group of symmetries (⟨π, τ, σ⟩) that leaves the fundamental physics of the CE1 system unchanged. The metric must be invariant under these gauge transformations.

### Canonicalization (canon!)
A procedure to normalize any given seed into a canonical representation. This involves applying a gauge_fix to remove the arbitrary effects of permutation, phase, and scale, resulting in:
- **Canonical state representation** (ψ̂): State vector in PK-diagonal basis
- **Sorted energy-vector** (Ê): Eigenvalues in canonical order  
- **Frame** (Û): PK-basis transformation matrix

This process is essential for ensuring that different but equivalent representations of a state are treated identically.

### Matching (match!)
An algorithm to find the optimal gauge transformation (g* = (π*, τ*, σ*)) that best aligns the canonicalized representations of seeds A and B. This is broken down into three steps:

#### Permutation Matching (π*)
Uses a Hungarian algorithm to find the optimal permutation that minimizes the difference between the sorted energy vectors of A and B. This is based on the intuition that similar states should have similar energy spectra.

#### Phase Matching (τ*)
Finds the phase shift (τ) that maximizes the overlap between the states' wavefunctions, using a closed-form solution.

#### Scale Matching (σ*)
Finds the optimal scaling (σ) by maximizing the Mellin correlation of the wavefunctions. The Mellin transform is used because of its scale-invariance property, which is ideal for this purpose.

### Residuals (residuals!)
Quantifies the mismatch between the optimally aligned seeds by calculating:
- **ΔE**: Energy vector mismatch
- **Δφ**: State overlap deficit  
- **ΔΩ**: Commutator consistency
- **Δℳ**: Mellin correlation deficit

### Metric (metric!)
Defines the scalar distance (d²(A,B)) as a weighted sum of the squared residuals:

```
d²(A,B) = w_E ‖ΔE‖₂² + w_φ (Δφ)² + w_Ω (ΔΩ)² + w_ℳ (Δℳ)²
```

This allows for tuning the sensitivity of the metric to different types of mismatches.

### Witnesses (witness.metric!)
Provides a way to verify the properties of the metric, including non-negativity, symmetry, and identity. The triangle inequality is checked empirically, with the results logged as a certificate.

### Kernel (kernel!)
A kernel function based on the distance metric, which can be used for tasks like spectral clustering or kernel-based machine learning.

### Solver (solver!)
Outlines the computational strategy, starting with an efficient approximation and including an optional refinement step for higher precision.

### Diagnostics (diagnostics!)
Defines the output format for the metric calculation, including a certificate of the optimal gauge transformation and the residual terms, along with a check for invariant preservation.

## Implementation

The QL-Metric is implemented in `seed_metric/ce1_seed_metric.py` with the following key classes:

- **`CE1SeedMetric`**: Main metric implementation
- **`CanonicalForm`**: Result from gauge fixing procedure
- **`AlignmentResult`**: Optimal gauge transformation parameters
- **`MetricCertificate`**: Complete verification certificate

### Usage Example

```python
from seed_metric.ce1_seed_metric import CE1SeedMetric

# Initialize metric with custom weights
metric = CE1SeedMetric(weights={
    "w_E": 1.0,      # Energy vector weight
    "w_phi": 1.0,    # State overlap weight  
    "w_Omega": 1.0,  # Commutator weight
    "w_M": 1.0       # Mellin correlation weight
})

# Canonicalize states A and B
canonical_A = metric.canonicalize(state_A)
canonical_B = metric.canonicalize(state_B)

# Find optimal alignment
alignment = metric.align(canonical_A, canonical_B)

# Compute distance metric
result = metric.compute_metric(canonical_A, canonical_B, alignment)
distance = result["distance"]
```

## Analogy to Physical Systems

The QL-Metric can be understood through an analogy to physics, where comparing two systems involves more than just a simple difference calculation. For instance, comparing two molecules requires accounting for their rotational and translational symmetries. The QL-Metric generalizes this idea to the abstract, thermodynamic CE1 system by:

### Defining Symmetries
The gauge group (G) of permutations, phase shifts, and scaling acts like the rotational and translational symmetries of a molecule.

### Finding Optimal Alignment
The match! procedure is analogous to optimally aligning two molecules to compare their structural differences. The Hungarian algorithm matches energy levels, while the phase and Mellin correlations align the finer details.

### Measuring Residuals
The residuals measure the mismatch after optimal alignment, which is a robust indicator of the fundamental difference between the systems, independent of their arbitrary orientation.

### Ensuring Consistency
The invariants check confirms that the comparison respects the fundamental laws of the system.

## Integration with Redoxa

The QL-Metric integrates with the Redoxa architecture in several ways:

1. **Planner Integration**: Can be used to guide the A* search by measuring distances between states in the frontier
2. **Scoring Enhancement**: Complements the existing MDL scoring with gauge-invariant distance measures
3. **Kernel Operations**: Provides kernel functions for downstream machine learning tasks
4. **Certificate System**: Ensures verifiable computation with full transparency

## Mathematical Foundation

The QL-Metric operates on the orbit space QL/G, where QL represents the space of QuantumLattice states and G is the gauge group. The metric satisfies:

- **Non-negativity**: d(A,B) ≥ 0
- **Identity**: d(A,A) = 0  
- **Symmetry**: d(A,B) = d(B,A)
- **Triangle Inequality**: d(A,C) ≤ d(A,B) + d(B,C)

The implementation includes empirical verification of these properties through the witness system.

## Future Directions

The QL-Metric provides a robust, gauge-invariant method for quantifying the difference between two CE1 states, laying the groundwork for more advanced operations like:

- **Averaging**: Computing Fréchet means in the state space
- **Clustering**: Spectral clustering based on QL-Metric distances
- **Navigation**: Efficient exploration of the state space
- **Optimization**: Gradient-based optimization respecting gauge symmetries

This foundation enables the development of sophisticated thermodynamic learning systems that can reason about state equivalence and similarity in a principled, mathematically rigorous way.
