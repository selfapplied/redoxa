"""
CE1 Planner Shim: Metric-Guided Search Integration

Integrates CE1 seed-metric with the planner system for metric-guided exploration.
Provides pure functions for distance comparison with witness emission.

CE1{
  seed: QL-MetricOp
  goal: given {Q_target}, return argmin_Q' d(Q', Q_target) subject to constraints
  invariants: I1..I5
  operators: { propose(), align(), certify(), accept_if Δd<0 }
  witness: store ζ(certificate)
}
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass
import time
import hashlib
from ce1_seed_metric import CE1SeedMetric, MetricCertificate, CanonicalForm
from ce1_kernel_atlas import CE1KernelAtlas
from ce1_certificate_cache import CE1CertificateCache

@dataclass
class PlannerState:
    """State in the planner search"""
    seedstream: np.ndarray
    canonical_form: CanonicalForm
    distance_to_target: float
    certificate: MetricCertificate
    ecological_fit: float = 0.0  # Distance from critical line
    parent_state: Optional['PlannerState'] = None
    generation: int = 0

@dataclass
class PlannerResult:
    """Result of planner search"""
    path: List[PlannerState]
    target_reached: bool
    final_distance: float
    total_certificates: int
    execution_time: float
    witness_log: List[Dict[str, Any]]

class CE1PlannerShim:
    """
    CE1 Planner Shim
    
    Provides metric-guided search capabilities for the planner system.
    Integrates with existing planner to enable distance-based exploration.
    """
    
    def __init__(self, metric: Optional[CE1SeedMetric] = None, 
                 atlas: Optional[CE1KernelAtlas] = None,
                 cache: Optional[CE1CertificateCache] = None):
        """
        Initialize planner shim
        
        Args:
            metric: CE1 seed metric instance
            atlas: Optional kernel atlas for navigation
            cache: Optional certificate cache
        """
        self.metric = metric or CE1SeedMetric()
        self.atlas = atlas
        self.cache = cache or CE1CertificateCache(max_size=1000)
        self.witness_log = []
        
    def metric_compare(self, seedstream_A: np.ndarray, seedstream_B: np.ndarray,
                      hamiltonian: Optional[np.ndarray] = None) -> Tuple[float, MetricCertificate]:
        """
        Pure function for metric comparison with witness emission
        
        CE1{
          metric_compare: (A, B) -> (d, cert)
          side_effects: emit ζ(certificate) to witness log
          invariants: preserve I1-I5
        }
        
        Returns:
            Tuple of (distance, certificate)
        """
        # Check cache first
        cached_cert = self.cache.get(seedstream_A, seedstream_B)
        if cached_cert:
            return cached_cert.distance, cached_cert
        
        # Compute distance
        cert = self.metric.compute_distance(seedstream_A, seedstream_B, hamiltonian)
        
        # Cache result
        cache_key = self.cache.put(seedstream_A, seedstream_B, cert, run_seed=42)
        
        # Emit witness
        witness_entry = {
            "timestamp": time.time(),
            "operation": "metric_compare",
            "certificate": {
                "distance": cert.distance,
                "pi_star": cert.pi_star,
                "tau_star": cert.tau_star,
                "sigma_star": cert.sigma_star,
                "residuals": cert.residuals,
                "execution_time": cert.execution_time,
                "invariants_preserved": cert.invariants_preserved
            },
            "cache_key": cache_key
        }
        self.witness_log.append(witness_entry)
        
        return cert.distance, cert
    
    
    def propose_candidates(self, current_state: PlannerState, target_state: PlannerState,
                          n_candidates: int = 5) -> List[PlannerState]:
        """
        Propose candidate states for exploration
        
        CE1{
          propose: generate {Q'} candidates
          method: use atlas navigation or random perturbations
          constraint: maintain gauge equivalence
        }
        """
        candidates = []
        
        if self.atlas is not None:
            # Use atlas for intelligent navigation
            candidates.extend(self._propose_from_atlas(current_state, target_state, n_candidates))
        else:
            # Use random perturbations
            candidates.extend(self._propose_random_perturbations(current_state, n_candidates))
        
        return candidates
    
    def _propose_from_atlas(self, current_state: PlannerState, target_state: PlannerState,
                           n_candidates: int) -> List[PlannerState]:
        """Propose candidates using atlas navigation"""
        candidates = []
        
        # Get current navigation coordinates
        current_coords = self.atlas.get_navigation_coords(current_state.seedstream)
        target_coords = self.atlas.get_navigation_coords(target_state.seedstream)
        
        # Generate interpolated candidates
        for i in range(n_candidates):
            # Interpolate between current and target
            alpha = (i + 1) / (n_candidates + 1)
            interpolated_coords = (1 - alpha) * current_coords + alpha * target_coords
            
            # Find nearest atlas point
            # This is simplified - full implementation would use inverse mapping
            candidate_seedstream = self._interpolate_seedstream(
                current_state.seedstream, target_state.seedstream, alpha
            )
            
            # Create candidate state
            canonical_form = self.metric.canonicalize(candidate_seedstream)
            distance, cert = self.metric_compare(candidate_seedstream, target_state.seedstream)
            
            candidate = PlannerState(
                seedstream=candidate_seedstream,
                canonical_form=canonical_form,
                distance_to_target=distance,
                certificate=cert,
                parent_state=current_state,
                generation=current_state.generation + 1
            )
            candidates.append(candidate)
        
        return candidates
    
    def _propose_random_perturbations(self, current_state: PlannerState, 
                                    n_candidates: int) -> List[PlannerState]:
        """Propose candidates using random gauge perturbations"""
        candidates = []
        n = len(current_state.seedstream)
        
        for i in range(n_candidates):
            # Apply small random gauge transformation
            random_perm = np.random.permutation(n)
            random_phase = np.random.uniform(-0.1, 0.1)  # Small phase change
            random_scale = np.random.uniform(0.95, 1.05)  # Small scale change
            
            candidate_seedstream = (current_state.seedstream[random_perm] * 
                                  np.exp(1j * random_phase) * random_scale)
            candidate_seedstream = candidate_seedstream / np.linalg.norm(candidate_seedstream)
            
            # Create candidate state
            canonical_form = self.metric.canonicalize(candidate_seedstream)
            distance, cert = self.metric_compare(candidate_seedstream, current_state.seedstream)
            
            candidate = PlannerState(
                seedstream=candidate_seedstream,
                canonical_form=canonical_form,
                distance_to_target=distance,
                certificate=cert,
                parent_state=current_state,
                generation=current_state.generation + 1
            )
            candidates.append(candidate)
        
        return candidates
    
    def _interpolate_seedstream(self, seedstream_A: np.ndarray, seedstream_B: np.ndarray,
                               alpha: float) -> np.ndarray:
        """Interpolate between two seedstreams"""
        # Simple linear interpolation in complex space
        interpolated = (1 - alpha) * seedstream_A + alpha * seedstream_B
        return interpolated / np.linalg.norm(interpolated)
    
    def greedy_search(self, initial_seedstream: np.ndarray, target_seedstream: np.ndarray,
                     max_iterations: int = 10, tolerance: float = 1e-6,
                     hamiltonian: Optional[np.ndarray] = None) -> PlannerResult:
        """
        Greedy metric-guided search
        
        CE1{
          greedy_search: pick next state that reduces metric to goal set
          method: propose -> align -> certify -> accept_if Δd<0
          stop: when d(Q, target) < tolerance or max_iterations
        }
        """
        start_time = time.time()
        
        # Initialize states
        initial_canonical = self.metric.canonicalize(initial_seedstream, hamiltonian)
        initial_distance, initial_cert = self.metric_compare(initial_seedstream, target_seedstream, hamiltonian)
        
        initial_state = PlannerState(
            seedstream=initial_seedstream,
            canonical_form=initial_canonical,
            distance_to_target=initial_distance,
            certificate=initial_cert,
            generation=0
        )
        
        target_canonical = self.metric.canonicalize(target_seedstream, hamiltonian)
        target_distance, target_cert = self.metric_compare(target_seedstream, target_seedstream, hamiltonian)
        
        target_state = PlannerState(
            seedstream=target_seedstream,
            canonical_form=target_canonical,
            distance_to_target=target_distance,
            certificate=target_cert,
            generation=0
        )
        
        # Search loop
        current_state = initial_state
        path = [current_state]
        
        for iteration in range(max_iterations):
            # Check if target reached
            if current_state.distance_to_target < tolerance:
                break
            
            # Propose candidates
            candidates = self.propose_candidates(current_state, target_state, n_candidates=5)
            
            # Find best candidate (lowest distance to target)
            best_candidate = min(candidates, key=lambda c: c.distance_to_target)
            
            # Accept if improvement
            if best_candidate.distance_to_target < current_state.distance_to_target:
                current_state = best_candidate
                path.append(current_state)
            else:
                # No improvement found
                break
        
        execution_time = time.time() - start_time
        
        return PlannerResult(
            path=path,
            target_reached=current_state.distance_to_target < tolerance,
            final_distance=current_state.distance_to_target,
            total_certificates=len(self.witness_log),
            execution_time=execution_time,
            witness_log=self.witness_log.copy()
        )
    
    def get_witness_summary(self) -> Dict[str, Any]:
        """Get summary of witness log"""
        if not self.witness_log:
            return {"total_witnesses": 0}
        
        total_witnesses = len(self.witness_log)
        total_execution_time = sum(w["certificate"]["execution_time"] for w in self.witness_log)
        avg_distance = np.mean([w["certificate"]["distance"] for w in self.witness_log])
        
        return {
            "total_witnesses": total_witnesses,
            "total_execution_time": total_execution_time,
            "avg_distance": avg_distance,
            "cache_size": self.cache.get_stats()["cache_size"],
            "invariants_preserved": all(w["certificate"]["invariants_preserved"] for w in self.witness_log)
        }
    
    def measure_ecological_fit(self, seedstream: np.ndarray, critical_line: float = 0.5) -> float:
        """
        Measure ecological fit as distance from critical line
        
        CE1{
          ecological_fit: distance from critical line (Re(s) = 1/2)
          interpretation: smaller distance = better fit = more cooperative
          principle: canonical line = ecological backbone
        }
        """
        return self.metric.measure_ecological_fit(seedstream, critical_line)
    
    def measure_cooperation_synergy(self, seedstreams: List[np.ndarray], critical_line: float = 0.5) -> Dict[str, float]:
        """
        Measure cooperation synergy as reduction of distance-to-line when seeds cooperate
        
        CE1{
          synergy: reduction of distance when seeds act together
          cooperation: migration of diverse seeds into alignment with canonical axis
          principle: synergy = reduction of distance-to-line
        }
        """
        return self.metric.measure_cooperation_synergy(seedstreams, critical_line)
    
    def create_planner_state_with_ecological_fit(self, seedstream: np.ndarray, target: np.ndarray, 
                                                parent_state: Optional[PlannerState] = None) -> PlannerState:
        """
        Create planner state with ecological fit measurement
        
        CE1{
          state: (seedstream, canonical_form, distance_to_target, ecological_fit)
          ecological_fit: distance from critical line
          generation: inherited from parent or 0
        }
        """
        canonical_form = self.metric.canonicalize(seedstream)
        certificate = self.metric.compute_distance(seedstream, target)
        ecological_fit = self.measure_ecological_fit(seedstream)
        
        generation = 0
        if parent_state is not None:
            generation = parent_state.generation + 1
        
        return PlannerState(
            seedstream=seedstream,
            canonical_form=canonical_form,
            distance_to_target=certificate.distance,
            certificate=certificate,
            ecological_fit=ecological_fit,
            parent_state=parent_state,
            generation=generation
        )

def demo_planner_shim():
    """Demonstrate CE1 planner shim"""
    print("=== CE1 Planner Shim Demonstration ===")
    
    # Create test seedstreams
    n = 8
    initial_seedstream = np.random.randn(n) + 1j * np.random.randn(n)
    initial_seedstream = initial_seedstream / np.linalg.norm(initial_seedstream)
    
    # Create target (gauge-equivalent variant)
    random_perm = np.random.permutation(n)
    random_phase = np.random.uniform(0, 2*np.pi)
    random_scale = np.random.uniform(0.5, 2.0)
    
    target_seedstream = initial_seedstream[random_perm] * np.exp(1j * random_phase) * random_scale
    target_seedstream = target_seedstream / np.linalg.norm(target_seedstream)
    
    print(f"Initial seedstream shape: {initial_seedstream.shape}")
    print(f"Target seedstream shape: {target_seedstream.shape}")
    print(f"Applied transformation: perm={random_perm}, phase={random_phase:.3f}, scale={random_scale:.3f}")
    
    # Initialize planner shim
    planner = CE1PlannerShim()
    
    # Test metric comparison
    print("\nTesting metric comparison...")
    distance, cert = planner.metric_compare(initial_seedstream, target_seedstream)
    print(f"Distance: {distance:.6f}")
    print(f"Certificate execution time: {cert.execution_time:.4f}s")
    
    # Run greedy search
    print("\nRunning greedy search...")
    result = planner.greedy_search(initial_seedstream, target_seedstream, max_iterations=5)
    
    print(f"Search results:")
    print(f"  Target reached: {result.target_reached}")
    print(f"  Final distance: {result.final_distance:.6f}")
    print(f"  Path length: {len(result.path)}")
    print(f"  Total certificates: {result.total_certificates}")
    print(f"  Execution time: {result.execution_time:.4f}s")
    
    # Show path progression
    print("\nPath progression:")
    for i, state in enumerate(result.path):
        print(f"  Step {i}: distance = {state.distance_to_target:.6f}, generation = {state.generation}")
    
    # Get witness summary
    witness_summary = planner.get_witness_summary()
    print(f"\nWitness summary:")
    for key, value in witness_summary.items():
        print(f"  {key}: {value}")
    
    print("\n✓ CE1 planner shim demonstration completed!")
    print("The shim provides metric-guided search with witness emission.")

if __name__ == "__main__":
    demo_planner_shim()
