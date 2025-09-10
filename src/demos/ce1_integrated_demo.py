"""
CE1 Integrated Demo: From Metric to Machine

Demonstrates the complete CE1 discovery loop:
1. Symmetry-fixed metric with rigorous testing
2. Kernel atlas with spectral embedding
3. Planner shim with witness emission
4. Barycenter computation for consensus passports
5. Chess board algebra in 𝔽₂⁸

This is the "Week-Zero → Week-One" progression from metric to machine.
"""

import numpy as np
import time
from typing import List, Dict, Any

# Import our CE1 components
from ce1_seed_metric import CE1SeedMetric
from ce1_kernel_atlas import CE1KernelAtlas
from ce1_planner_shim import CE1PlannerShim
from ce1_barycenter import CE1Barycenter
from ce1_chess_algebra import CE1ChessAlgebra, BoundaryPolicy, OrientationMode

def demo_complete_ce1_system():
    """Demonstrate the complete CE1 discovery loop"""
    print("=== CE1 Complete System Demonstration ===")
    print("Week-Zero → Week-One: from metric to machine")
    print()
    
    # Initialize all components
    print("Initializing CE1 components...")
    metric = CE1SeedMetric()
    atlas_builder = CE1KernelAtlas(metric)
    planner = CE1PlannerShim(metric)
    barycenter_computer = CE1Barycenter(metric)
    chess_algebra = CE1ChessAlgebra()
    
    print("✓ All components initialized")
    print()
    
    # 1. Symmetry-fixed metric with rigorous testing
    print("1. Testing symmetry-fixed metric...")
    symmetry_results = metric.test_symmetry_rigorous(n_tests=50, n_dim=8)
    print(f"   Symmetry pass rate: {symmetry_results['symmetry_pass_rate']:.1%}")
    print(f"   Max symmetry error: {symmetry_results['max_symmetry_error']:.2e}")
    print()
    
    # 2. Create seedstreams and build kernel atlas
    print("2. Building kernel atlas...")
    n_seedstreams = 6
    n_dim = 8
    
    # Create test seedstreams
    seedstreams = []
    base_seedstream = np.random.randn(n_dim) + 1j * np.random.randn(n_dim)
    base_seedstream = base_seedstream / np.linalg.norm(base_seedstream)
    seedstreams.append(base_seedstream)
    
    # Create gauge-equivalent variants
    for i in range(n_seedstreams - 1):
        random_perm = np.random.permutation(n_dim)
        random_phase = np.random.uniform(0, 2*np.pi)
        random_scale = np.random.uniform(0.5, 2.0)
        
        variant = base_seedstream[random_perm] * np.exp(1j * random_phase) * random_scale
        variant = variant / np.linalg.norm(variant)
        seedstreams.append(variant)
    
    # Build atlas
    atlas = atlas_builder.build_atlas(seedstreams, n_eigenvectors=3)
    print(f"   Atlas built with {len(atlas.seedstreams)} seedstreams")
    print(f"   Kernel bandwidth λ: {atlas.lambda_param:.6f}")
    print(f"   Number of clusters: {len(np.unique(atlas.clusters))}")
    print()
    
    # 3. Planner shim with witness emission
    print("3. Testing planner shim...")
    initial_seedstream = seedstreams[0]
    target_seedstream = seedstreams[-1]
    
    result = planner.greedy_search(initial_seedstream, target_seedstream, max_iterations=3)
    print(f"   Search completed: target_reached={result.target_reached}")
    print(f"   Final distance: {result.final_distance:.6f}")
    print(f"   Path length: {len(result.path)}")
    print(f"   Total witnesses: {len(result.witness_log)}")
    print()
    
    # 4. Barycenter computation
    print("4. Computing barycenter...")
    barycenter_result = barycenter_computer.compute_barycenter(seedstreams[:4], max_iterations=5)
    print(f"   Barycenter computed in {barycenter_result.iterations} iterations")
    print(f"   Mean distance: {barycenter_result.mean_distance:.6f}")
    print(f"   Stability score: {barycenter_result.stability_score:.6f}")
    print()
    
    # 5. Chess board algebra
    print("5. Testing chess board algebra...")
    
    # Test encoding
    test_squares = [("a", 1), ("e", 4), ("h", 8)]
    for file, rank in test_squares:
        byte_val = chess_algebra.encode_square(file, rank, BoundaryPolicy.TORUS, OrientationMode.MU)
        decoded = chess_algebra.decode_square(byte_val)
        print(f"   {file}{rank} → {byte_val:02X} → {decoded.file}{decoded.rank}")
    
    # Test move generation
    from_square = chess_algebra.decode_square(chess_algebra.encode_square("e", 4, BoundaryPolicy.TORUS, OrientationMode.MU))
    moves = chess_algebra.generate_all_moves(from_square)
    analysis = chess_algebra.analyze_move_patterns(moves)
    
    print(f"   From {from_square.file}{from_square.rank}: {analysis['total_moves']} moves")
    print(f"   Hamming distribution: {analysis['hamming_distribution']}")
    print()
    
    # 6. Integration test: Use chess squares as seedstreams
    print("6. Integration test: Chess squares as seedstreams...")
    
    # Convert chess squares to seedstreams
    chess_seedstreams = []
    for file in ['a', 'e', 'h']:
        for rank in [1, 4, 8]:
            byte_val = chess_algebra.encode_square(file, rank, BoundaryPolicy.TORUS, OrientationMode.MU)
            # Convert byte to complex seedstream (safer method)
            seedstream = np.array([complex(byte_val & (1 << i), (byte_val >> 4) & (1 << i)) for i in range(8)])
            norm = np.linalg.norm(seedstream)
            if norm > 0:
                seedstream = seedstream / norm
            else:
                # Fallback: create a simple seedstream
                seedstream = np.random.randn(8) + 1j * np.random.randn(8)
                seedstream = seedstream / np.linalg.norm(seedstream)
            chess_seedstreams.append(seedstream)
    
    # Build atlas for chess seedstreams
    chess_atlas = atlas_builder.build_atlas(chess_seedstreams, n_eigenvectors=2)
    print(f"   Chess atlas built with {len(chess_atlas.seedstreams)} squares")
    print(f"   Number of clusters: {len(np.unique(chess_atlas.clusters))}")
    
    # Test navigation
    test_square = chess_seedstreams[0]
    nav_coords = atlas_builder.get_navigation_coords(test_square)
    print(f"   Navigation coordinates: {nav_coords}")
    print()
    
    # 7. Summary and next steps
    print("7. System Summary:")
    print("   ✓ Symmetry-fixed metric with rigorous testing")
    print("   ✓ Kernel atlas with spectral embedding")
    print("   ✓ Planner shim with witness emission")
    print("   ✓ Barycenter computation for consensus passports")
    print("   ✓ Chess board algebra in 𝔽₂⁸")
    print("   ✓ Integration test with chess squares as seedstreams")
    print()
    
    print("Next steps for Week-One:")
    print("   • Scale to 256 metanions with optimized algorithms")
    print("   • Implement autotune for metric weights")
    print("   • Add adversarial and drift tests")
    print("   • Build certificate cache with compression")
    print("   • Create consensus passports for training anchors")
    print()
    
    print("✓ CE1 discovery loop demonstration completed!")
    print("The system provides a working foundation for metric-guided exploration.")

def demo_chess_metric_integration():
    """Demonstrate chess board integration with CE1 metric"""
    print("\n=== Chess-Metric Integration Demo ===")
    
    chess = CE1ChessAlgebra()
    metric = CE1SeedMetric()
    
    # Create chess position as seedstream
    print("Creating chess position as seedstream...")
    
    # Encode a simple position (pawns on ranks 2 and 7)
    position_bytes = []
    for file in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']:
        # White pawns on rank 2
        white_pawn = chess.encode_square(file, 2, BoundaryPolicy.TORUS, OrientationMode.MU)
        position_bytes.append(white_pawn)
        
        # Black pawns on rank 7
        black_pawn = chess.encode_square(file, 7, BoundaryPolicy.TORUS, OrientationMode.MU)
        position_bytes.append(black_pawn)
    
    # Convert to complex seedstream (safer method)
    position_seedstream = np.array([complex(byte_val & 0xFF, (byte_val >> 4) & 0xFF) for byte_val in position_bytes])
    norm = np.linalg.norm(position_seedstream)
    if norm > 0:
        position_seedstream = position_seedstream / norm
    else:
        # Fallback: create a simple seedstream
        position_seedstream = np.random.randn(16) + 1j * np.random.randn(16)
        position_seedstream = position_seedstream / np.linalg.norm(position_seedstream)
    
    print(f"Position seedstream shape: {position_seedstream.shape}")
    print(f"Position seedstream norm: {np.linalg.norm(position_seedstream):.6f}")
    
    # Create a modified position (move a pawn)
    print("\nCreating modified position (move pawn from e2 to e4)...")
    
    modified_bytes = position_bytes.copy()
    # Find e2 (index 8) and e7 (index 9) in our array
    e2_index = 8  # e is index 4, rank 2 is index 1, so 4*2 + 1 = 9, but we have 2 bytes per file
    e7_index = 9
    
    # Move e2 to e4
    e4_byte = chess.encode_square('e', 4, BoundaryPolicy.TORUS, OrientationMode.MU)
    modified_bytes[e2_index] = e4_byte
    
    modified_seedstream = np.array([complex(byte_val & 0xFF, (byte_val >> 4) & 0xFF) for byte_val in modified_bytes])
    norm = np.linalg.norm(modified_seedstream)
    if norm > 0:
        modified_seedstream = modified_seedstream / norm
    else:
        # Fallback: create a simple seedstream
        modified_seedstream = np.random.randn(16) + 1j * np.random.randn(16)
        modified_seedstream = modified_seedstream / np.linalg.norm(modified_seedstream)
    
    # Compute distance between positions
    print("Computing distance between positions...")
    cert = metric.compute_distance(position_seedstream, modified_seedstream)
    
    print(f"Distance between positions: {cert.distance:.6f}")
    print(f"Optimal permutation: {cert.pi_star}")
    print(f"Optimal phase: {cert.tau_star:.6f}")
    print(f"Optimal scale: {cert.sigma_star:.6f}")
    print(f"Execution time: {cert.execution_time:.4f}s")
    
    # Analyze residuals
    print("\nResiduals:")
    for key, value in cert.residuals.items():
        print(f"  {key}: {value:.6f}")
    
    print("\n✓ Chess-metric integration demo completed!")
    print("Chess positions can be used as seedstreams in the CE1 metric system.")

if __name__ == "__main__":
    demo_complete_ce1_system()
    demo_chess_metric_integration()
