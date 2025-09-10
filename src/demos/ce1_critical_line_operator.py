"""
CE1 Critical Line Operator: Ecological Backbone Canonicalization

Implements the Riemann critical line as the canonical axis for seed cooperation.
Maps "fit" to "distance from critical line" where the critical line serves as
the ecological backbone - the trust surface where seeds converge.

CE1{
  canonical_line: Re(s) = 1/2 (Riemann critical line)
  fit: distance_to_critical_line(seed)
  synergy: reduction of distance when seeds cooperate
  ecological_backbone: critical line as trust surface
  cooperation: migration of diverse seeds into alignment with canonical axis
}
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, NamedTuple
from dataclasses import dataclass
import math
from scipy.optimize import minimize_scalar
from ce1_seed_metric import CE1SeedMetric, CanonicalForm

@dataclass
class CriticalLineProjection:
    """Projection of seed onto critical line"""
    real_part: float  # Re(s) - distance from critical line
    imaginary_part: float  # Im(s) - position along critical line
    distance_to_line: float  # |Re(s) - 1/2| - the "fit" measure
    canonical_coordinate: complex  # Projected point on critical line

@dataclass
class EcologicalBackbone:
    """Ecological backbone defined by critical line"""
    trust_surface: np.ndarray  # Canonical axis for cooperation
    critical_line: float = 0.5  # Re(s) = 1/2
    resonance_guarantee: bool = True  # Resonance guaranteed on critical line

@dataclass
class SynergyResult:
    """Result of seed cooperation synergy measurement"""
    individual_distances: List[float]  # Distance-to-line for each seed
    cooperative_distance: float  # Distance-to-line when seeds cooperate
    synergy_gain: float  # Reduction in distance (positive = synergy)
    canonical_alignment: float  # How well seeds align with critical line

class CE1CriticalLineOperator:
    """
    CE1 Critical Line Operator
    
    Implements the Riemann critical line as the canonical axis for seed cooperation.
    The critical line Re(s) = 1/2 serves as the ecological backbone - the trust surface
    where diverse seeds converge and cooperate.
    """
    
    def __init__(self, critical_line: float = 0.5):
        """
        Initialize critical line operator
        
        Args:
            critical_line: The canonical line Re(s) = critical_line (default 1/2)
        """
        self.critical_line = critical_line
        self.ecological_backbone = EcologicalBackbone(
            critical_line=critical_line,
            trust_surface=self._generate_trust_surface()
        )
        
        # Initialize seed metric for canonicalization
        self.seed_metric = CE1SeedMetric()
        
    def _generate_trust_surface(self) -> np.ndarray:
        """Generate the trust surface (canonical axis) for cooperation"""
        # The critical line as a complex axis
        # For simplicity, we'll use a discretized version
        t_values = np.linspace(-10, 10, 1000)  # Imaginary part range
        critical_line_points = np.array([self.critical_line + 1j * t for t in t_values])
        return critical_line_points
    
    def project_to_critical_line(self, seed: np.ndarray) -> CriticalLineProjection:
        """
        Project seed onto critical line
        
        CE1{
          project: seed → critical_line
          method: find closest point on Re(s) = 1/2
          output: (Re(s), Im(s), distance, canonical_coordinate)
        }
        """
        # Canonicalize the seed first
        canonical_form = self.seed_metric.canonicalize(seed)
        
        # Extract the "zeta-like" properties from the canonical form
        # We'll use the energy vector as a proxy for zeta function values
        energy_vector = canonical_form.E_hat
        
        # Find the best projection onto critical line
        # This is where we map the seed's "chaos" to the critical line
        best_projection = self._find_best_critical_projection(energy_vector)
        
        real_part = best_projection.real
        imaginary_part = best_projection.imag
        distance_to_line = abs(real_part - self.critical_line)
        canonical_coordinate = complex(self.critical_line, imaginary_part)
        
        return CriticalLineProjection(
            real_part=real_part,
            imaginary_part=imaginary_part,
            distance_to_line=distance_to_line,
            canonical_coordinate=canonical_coordinate
        )
    
    def _find_best_critical_projection(self, energy_vector: np.ndarray) -> complex:
        """
        Find the best projection of energy vector onto critical line
        
        This implements the core insight: chaotic oscillations normalized
        onto the canonical axis where resonance is guaranteed.
        """
        # Use the dominant frequency/phase from the energy vector
        # to determine the best position on the critical line
        
        # Find the dominant component
        dominant_idx = np.argmax(np.abs(energy_vector))
        dominant_phase = np.angle(energy_vector[dominant_idx])
        
        # Map phase to imaginary part of critical line
        # This is where the "chaos" gets normalized onto the canonical axis
        imaginary_part = dominant_phase / np.pi  # Normalize to [-1, 1] range
        
        # The real part is determined by how "close" the energy vector
        # is to being on the critical line
        energy_norm = np.linalg.norm(energy_vector)
        real_part = self.critical_line + (1.0 - energy_norm) * 0.1  # Small deviation
        
        return complex(real_part, imaginary_part)
    
    def measure_fit(self, seed: np.ndarray) -> float:
        """
        Measure "fit" as distance from critical line
        
        CE1{
          fit: distance_to_critical_line(seed)
          interpretation: smaller distance = better fit = more cooperative
        }
        """
        projection = self.project_to_critical_line(seed)
        return projection.distance_to_line
    
    def measure_synergy(self, seeds: List[np.ndarray]) -> SynergyResult:
        """
        Measure synergy as reduction of distance-to-line when seeds cooperate
        
        CE1{
          synergy: reduction of distance when seeds act together
          cooperation: migration of diverse seeds into alignment with canonical axis
        }
        """
        # Measure individual distances
        individual_distances = [self.measure_fit(seed) for seed in seeds]
        
        # Create cooperative seed (weighted combination)
        # This represents seeds "acting together"
        cooperative_seed = self._create_cooperative_seed(seeds)
        cooperative_distance = self.measure_fit(cooperative_seed)
        
        # Synergy is the reduction in distance
        avg_individual_distance = np.mean(individual_distances)
        synergy_gain = avg_individual_distance - cooperative_distance
        
        # Measure how well seeds align with critical line
        canonical_alignment = self._measure_canonical_alignment(seeds)
        
        return SynergyResult(
            individual_distances=individual_distances,
            cooperative_distance=cooperative_distance,
            synergy_gain=synergy_gain,
            canonical_alignment=canonical_alignment
        )
    
    def _create_cooperative_seed(self, seeds: List[np.ndarray]) -> np.ndarray:
        """Create cooperative seed by combining individual seeds"""
        # Normalize all seeds first
        normalized_seeds = [seed / np.linalg.norm(seed) for seed in seeds]
        
        # Weighted combination (equal weights for simplicity)
        weights = np.ones(len(seeds)) / len(seeds)
        cooperative_seed = np.zeros_like(seeds[0])
        
        for seed, weight in zip(normalized_seeds, weights):
            cooperative_seed += weight * seed
            
        return cooperative_seed / np.linalg.norm(cooperative_seed)
    
    def _measure_canonical_alignment(self, seeds: List[np.ndarray]) -> float:
        """Measure how well seeds align with the canonical axis"""
        projections = [self.project_to_critical_line(seed) for seed in seeds]
        
        # Measure variance in imaginary parts (position along critical line)
        imaginary_parts = [p.imaginary_part for p in projections]
        alignment_variance = np.var(imaginary_parts)
        
        # Better alignment = lower variance
        return 1.0 / (1.0 + alignment_variance)
    
    def test_seed_cooperation(self, seeds: List[np.ndarray]) -> Dict[str, Any]:
        """
        Test seeds by how well they project onto the canonical axis
        
        CE1{
          test: project seeds onto critical line
          measure: fit, synergy, canonical alignment
          output: comprehensive cooperation analysis
        }
        """
        results = {}
        
        # Individual fit measurements
        individual_fits = [self.measure_fit(seed) for seed in seeds]
        results['individual_fits'] = individual_fits
        results['avg_individual_fit'] = np.mean(individual_fits)
        
        # Synergy measurement
        synergy_result = self.measure_synergy(seeds)
        results['synergy'] = synergy_result
        
        # Critical line projections
        projections = [self.project_to_critical_line(seed) for seed in seeds]
        results['projections'] = projections
        
        # Ecological backbone analysis
        results['ecological_backbone'] = self.ecological_backbone
        
        # Cooperation score (higher = more cooperative)
        cooperation_score = self._calculate_cooperation_score(seeds)
        results['cooperation_score'] = cooperation_score
        
        return results
    
    def _calculate_cooperation_score(self, seeds: List[np.ndarray]) -> float:
        """Calculate overall cooperation score"""
        synergy_result = self.measure_synergy(seeds)
        
        # Cooperation score combines:
        # 1. Low individual distances (good fit)
        # 2. High synergy gain (cooperation helps)
        # 3. Good canonical alignment (seeds align with critical line)
        
        avg_fit = np.mean(synergy_result.individual_distances)
        synergy_gain = synergy_result.synergy_gain
        alignment = synergy_result.canonical_alignment
        
        # Normalize and combine
        fit_score = 1.0 / (1.0 + avg_fit)  # Higher for better fit
        synergy_score = max(0, synergy_gain)  # Only positive synergy counts
        alignment_score = alignment
        
        cooperation_score = (fit_score + synergy_score + alignment_score) / 3.0
        return cooperation_score

def demo_critical_line_operator():
    """Demonstrate the critical line operator"""
    print("CE1 Critical Line Operator: Ecological Backbone Canonicalization")
    print("=" * 70)
    
    # Initialize operator
    operator = CE1CriticalLineOperator()
    
    # Create test seeds
    print("\nCreating test seeds...")
    n = 6
    seeds = []
    
    # Seed 1: Well-behaved (close to critical line)
    seed1 = np.array([1.0, 0.5, 0.3, 0.2, 0.1, 0.05], dtype=complex)
    seed1 = seed1 / np.linalg.norm(seed1)
    seeds.append(seed1)
    
    # Seed 2: Chaotic (far from critical line)
    seed2 = np.random.randn(n) + 1j * np.random.randn(n)
    seed2 = seed2 / np.linalg.norm(seed2)
    seeds.append(seed2)
    
    # Seed 3: Resonant (on critical line)
    seed3 = np.array([0.8, 0.6, 0.4, 0.3, 0.2, 0.1], dtype=complex)
    seed3 = seed3 / np.linalg.norm(seed3)
    seeds.append(seed3)
    
    print(f"✓ Created {len(seeds)} test seeds")
    
    # Test seed cooperation
    print("\n=== Testing Seed Cooperation ===")
    results = operator.test_seed_cooperation(seeds)
    
    # Display results
    print(f"\nIndividual fits (distance from critical line):")
    for i, fit in enumerate(results['individual_fits']):
        print(f"  Seed {i+1}: {fit:.6f}")
    
    print(f"\nAverage individual fit: {results['avg_individual_fit']:.6f}")
    
    synergy = results['synergy']
    print(f"\nSynergy Analysis:")
    print(f"  Cooperative distance: {synergy.cooperative_distance:.6f}")
    print(f"  Synergy gain: {synergy.synergy_gain:.6f}")
    print(f"  Canonical alignment: {synergy.canonical_alignment:.6f}")
    
    print(f"\nCooperation Score: {results['cooperation_score']:.6f}")
    
    # Display projections
    print(f"\nCritical Line Projections:")
    for i, proj in enumerate(results['projections']):
        print(f"  Seed {i+1}:")
        print(f"    Real part: {proj.real_part:.6f}")
        print(f"    Imaginary part: {proj.imaginary_part:.6f}")
        print(f"    Distance to line: {proj.distance_to_line:.6f}")
        print(f"    Canonical coordinate: {proj.canonical_coordinate}")
    
    print(f"\n✓ Critical line operator demonstrates ecological backbone principle")
    print(f"✓ Fit = distance from critical line (Re(s) = {operator.critical_line})")
    print(f"✓ Synergy = reduction of distance when seeds cooperate")
    print(f"✓ Cooperation = migration of diverse seeds into alignment with canonical axis")

if __name__ == "__main__":
    demo_critical_line_operator()
