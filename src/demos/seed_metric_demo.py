#!/usr/bin/env python3
"""
CE1 Seed-Metric Demo

Demonstrates the complete CE1 seed-metric implementation with:
- Canonicalization procedure
- Alignment algorithm with Hungarian algorithm
- Metric calculation with residuals
- Witness properties verification
- Certificate system
"""

import sys
import os
import numpy as np

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'seed_metric'))

from ce1_seed_metric import CE1SeedMetric, MetricCertificate

def create_test_seedstreams():
    """Create test seedstreams for demonstration"""
    print("Creating test seedstreams...")
    
    # Create base seedstream
    n = 6  # Small for demo
    base_seedstream = np.array([1.0, 0.5, 0.3, 0.2, 0.1, 0.05], dtype=complex)
    base_seedstream = base_seedstream / np.linalg.norm(base_seedstream)
    
    # Create equivalent seedstreams by applying gauge transformations
    seedstreams = []
    
    # Original
    seedstreams.append(("Original", base_seedstream))
    
    # Permutation only
    perm = [2, 0, 1, 4, 3, 5]
    permuted = base_seedstream[perm]
    seedstreams.append(("Permuted", permuted))
    
    # Phase shift only
    phase_shifted = base_seedstream * np.exp(1j * np.pi / 4)
    seedstreams.append(("Phase shifted", phase_shifted))
    
    # Scale transformation only
    scaled = base_seedstream * 1.5
    scaled = scaled / np.linalg.norm(scaled)
    seedstreams.append(("Scaled", scaled))
    
    # Combined transformation
    combined = base_seedstream[perm] * np.exp(1j * np.pi / 3) * 0.8
    combined = combined / np.linalg.norm(combined)
    seedstreams.append(("Combined", combined))
    
    # Different seedstream (not equivalent)
    different = np.random.randn(n) + 1j * np.random.randn(n)
    different = different / np.linalg.norm(different)
    seedstreams.append(("Different", different))
    
    print(f"✓ Created {len(seedstreams)} test seedstreams")
    
    return seedstreams

def demonstrate_canonicalization(metric: CE1SeedMetric, seedstreams):
    """Demonstrate canonicalization procedure"""
    print("\n=== Canonicalization Procedure ===")
    
    for name, seedstream in seedstreams[:3]:  # Test first 3
        print(f"\nCanonicalizing {name}:")
        print(f"  Original: {seedstream[:3]}...")
        
        canonical = metric.canonicalize(seedstream)
        
        print(f"  Canonical state: {canonical.psi_hat[:3]}...")
        print(f"  Energy vector: {canonical.E_hat[:3]}...")
        print(f"  PK basis shape: {canonical.U_hat.shape}")

def demonstrate_alignment(metric: CE1SeedMetric, seedstreams):
    """Demonstrate alignment procedure"""
    print("\n=== Alignment Procedure ===")
    
    # Test alignment between original and transformed
    original = seedstreams[0][1]  # Original
    transformed = seedstreams[4][1]  # Combined transformation
    
    print(f"Aligning original vs combined transformation:")
    
    # Canonicalize both
    canonical_A = metric.canonicalize(original)
    canonical_B = metric.canonicalize(transformed)
    
    # Align
    alignment = metric.align(canonical_A, canonical_B)
    
    print(f"  Optimal permutation: {alignment.pi_star}")
    print(f"  Optimal phase: {alignment.tau_star:.6f}")
    print(f"  Optimal scale: {alignment.sigma_star:.6f}")
    print(f"  Alignment cost: {alignment.cost:.6f}")

def demonstrate_metric_calculation(metric: CE1SeedMetric, seedstreams):
    """Demonstrate metric calculation"""
    print("\n=== Metric Calculation ===")
    
    # Test various pairs
    test_pairs = [
        (0, 1, "Original vs Permuted"),
        (0, 2, "Original vs Phase shifted"),
        (0, 3, "Original vs Scaled"),
        (0, 4, "Original vs Combined"),
        (0, 5, "Original vs Different")
    ]
    
    certificates = []
    
    for i, j, description in test_pairs:
        print(f"\n{description}:")
        
        seedstream_A = seedstreams[i][1]
        seedstream_B = seedstreams[j][1]
        
        certificate = metric.compute_distance(seedstream_A, seedstream_B)
        certificates.append(certificate)
        
        print(f"  Distance: {certificate.distance:.6f}")
        print(f"  Execution time: {certificate.execution_time:.4f}s")
        print(f"  Residuals:")
        for key, value in certificate.residuals.items():
            print(f"    {key}: {value:.6f}")
    
    return certificates

def demonstrate_witness_properties(metric: CE1SeedMetric, seedstreams):
    """Demonstrate witness properties verification"""
    print("\n=== Witness Properties Verification ===")
    
    # Extract just the seedstreams
    test_seedstreams = [s[1] for s in seedstreams]
    
    properties = metric.verify_metric_properties(test_seedstreams)
    
    print("Metric properties:")
    for prop, result in properties.items():
        if isinstance(result, bool):
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"  {prop}: {status}")
        else:
            print(f"  {prop}: {result}")

def demonstrate_kernel_computation(metric: CE1SeedMetric, seedstreams):
    """Demonstrate kernel computation"""
    print("\n=== Kernel Computation ===")
    
    # Test kernel between original and transformed
    original = seedstreams[0][1]
    transformed = seedstreams[4][1]
    
    kernel_value = metric.compute_kernel(original, transformed)
    
    print(f"Kernel K(original, transformed): {kernel_value:.6f}")
    
    # Test kernel between original and different
    different = seedstreams[5][1]
    kernel_value_2 = metric.compute_kernel(original, different)
    
    print(f"Kernel K(original, different): {kernel_value_2:.6f}")
    
    print(f"Kernel difference: {abs(kernel_value - kernel_value_2):.6f}")

def demonstrate_certificate_system(certificates):
    """Demonstrate certificate system"""
    print("\n=== Certificate System ===")
    
    print("Certificate summary:")
    for i, cert in enumerate(certificates):
        print(f"  Certificate {i+1}:")
        print(f"    Distance: {cert.distance:.6f}")
        print(f"    Optimal permutation: {cert.pi_star}")
        print(f"    Optimal phase: {cert.tau_star:.6f}")
        print(f"    Optimal scale: {cert.sigma_star:.6f}")
        print(f"    Invariants preserved: {cert.invariants_preserved}")
        print(f"    Execution time: {cert.execution_time:.4f}s")

def demonstrate_barycenter_computation(metric: CE1SeedMetric, seedstreams):
    """Demonstrate barycenter computation (Fréchet mean)"""
    print("\n=== Barycenter Computation ===")
    
    # Select equivalent seedstreams for barycenter
    equivalent_seedstreams = [s[1] for s in seedstreams[:5]]  # First 5 are equivalent
    
    print(f"Computing barycenter of {len(equivalent_seedstreams)} equivalent seedstreams...")
    
    # Simple barycenter computation (in practice, would use iterative Fréchet mean)
    # For demo, just average the canonical forms
    canonical_forms = [metric.canonicalize(s) for s in equivalent_seedstreams]
    
    # Average the canonical states
    avg_psi_hat = np.mean([cf.psi_hat for cf in canonical_forms], axis=0)
    avg_psi_hat = avg_psi_hat / np.linalg.norm(avg_psi_hat)
    
    print(f"  Barycenter state: {avg_psi_hat[:3]}...")
    
    # Compute distances from barycenter
    barycenter_canonical = metric.canonicalize(avg_psi_hat)
    
    print("  Distances from barycenter:")
    for i, seedstream in enumerate(equivalent_seedstreams):
        cert = metric.compute_distance(avg_psi_hat, seedstream)
        print(f"    Seedstream {i+1}: {cert.distance:.6f}")

def main():
    """Main demonstration"""
    print("CE1 Seed-Metric: Gauge-Invariant Distance for QuantumLattice Passports")
    print("=" * 70)
    
    # Create test data
    seedstreams = create_test_seedstreams()
    
    # Initialize metric
    print("\nInitializing CE1 seed-metric...")
    metric = CE1SeedMetric()
    print("✓ Metric initialized with default weights")
    
    # Demonstrate canonicalization
    demonstrate_canonicalization(metric, seedstreams)
    
    # Demonstrate alignment
    demonstrate_alignment(metric, seedstreams)
    
    # Demonstrate metric calculation
    certificates = demonstrate_metric_calculation(metric, seedstreams)
    
    # Demonstrate witness properties
    demonstrate_witness_properties(metric, seedstreams)
    
    # Demonstrate kernel computation
    demonstrate_kernel_computation(metric, seedstreams)
    
    # Demonstrate certificate system
    demonstrate_certificate_system(certificates)
    
    # Demonstrate barycenter computation
    demonstrate_barycenter_computation(metric, seedstreams)
    
    # Summary
    print("\n=== Summary ===")
    print("✓ CE1 seed-metric implemented with full specification")
    print("✓ Canonicalization procedure with PK-diagonal basis")
    print("✓ Alignment algorithm with Hungarian algorithm")
    print("✓ Metric calculation with residuals and weights")
    print("✓ Witness properties verification")
    print("✓ Certificate system for transparency")
    print("✓ Kernel computation for downstream use")
    print("✓ Barycenter computation for atlas-level averaging")
    
    print("\nThe CE1 seed-metric provides:")
    print("  • Gauge-invariant distance measurement")
    print("  • Full certificate system for verification")
    print("  • Respect for π, τ, σ group actions")
    print("  • Enables continuum-level analysis")
    print("  • Ready for Fréchet mean computation")
    
    print("\n🎉 The metric is now ready to be called by the planner")
    print("   to guide the search for efficient continuations!")

if __name__ == "__main__":
    main()
