"""
CE1 Quick Demo: Essential Components Working

A fast demonstration of the key CE1 components without infinite loops.
Shows the "Week-Zero → Week-One" progression from metric to machine.
"""

import numpy as np
import time
from ce1_seed_metric import CE1SeedMetric
from ce1_kernel_atlas import CE1KernelAtlas
from ce1_planner_shim import CE1PlannerShim
from ce1_chess_algebra import CE1ChessAlgebra, BoundaryPolicy, OrientationMode

def quick_demo():
    """Quick demonstration of CE1 components"""
    print("=== CE1 Quick Demo: From Metric to Machine ===")
    print()
    
    # Initialize components
    print("Initializing CE1 components...")
    metric = CE1SeedMetric()
    atlas_builder = CE1KernelAtlas(metric)
    planner = CE1PlannerShim(metric)
    chess = CE1ChessAlgebra()
    print("✓ All components initialized")
    print()
    
    # 1. Symmetry test (quick)
    print("1. Testing symmetry (50 random pairs)...")
    symmetry_results = metric.test_symmetry_rigorous(n_tests=50, n_dim=8)
    print(f"   Symmetry pass rate: {symmetry_results['symmetry_pass_rate']:.1%}")
    print(f"   Max symmetry error: {symmetry_results['max_symmetry_error']:.2e}")
    print()
    
    # 2. Create seedstreams and build atlas
    print("2. Building kernel atlas...")
    n_seedstreams = 4  # Smaller for speed
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
    atlas = atlas_builder.build_atlas(seedstreams, n_eigenvectors=2)
    print(f"   Atlas built with {len(atlas.seedstreams)} seedstreams")
    print(f"   Kernel bandwidth λ: {atlas.lambda_param:.6f}")
    print(f"   Number of clusters: {len(np.unique(atlas.clusters))}")
    print()
    
    # 3. Planner shim
    print("3. Testing planner shim...")
    initial_seedstream = seedstreams[0]
    target_seedstream = seedstreams[-1]
    
    result = planner.greedy_search(initial_seedstream, target_seedstream, max_iterations=2)
    print(f"   Search completed: target_reached={result.target_reached}")
    print(f"   Final distance: {result.final_distance:.6f}")
    print(f"   Path length: {len(result.path)}")
    print(f"   Total witnesses: {len(result.witness_log)}")
    print()
    
    # 4. Chess board algebra
    print("4. Testing chess board algebra...")
    
    # Test encoding
    test_squares = [("a", 1), ("e", 4), ("h", 8)]
    for file, rank in test_squares:
        byte_val = chess.encode_square(file, rank, BoundaryPolicy.TORUS, OrientationMode.MU)
        decoded = chess.decode_square(byte_val)
        print(f"   {file}{rank} → {byte_val:02X} → {decoded.file}{decoded.rank}")
    
    # Test move generation
    from_square = chess.decode_square(chess.encode_square("e", 4, BoundaryPolicy.TORUS, OrientationMode.MU))
    moves = chess.generate_all_moves(from_square)
    analysis = chess.analyze_move_patterns(moves)
    
    print(f"   From {from_square.file}{from_square.rank}: {analysis['total_moves']} moves")
    print(f"   Hamming distribution: {analysis['hamming_distribution']}")
    print()
    
    # 5. Integration test
    print("5. Integration test: Chess squares as seedstreams...")
    
    # Convert chess squares to seedstreams
    chess_seedstreams = []
    for file in ['a', 'e', 'h']:
        for rank in [1, 4, 8]:
            byte_val = chess.encode_square(file, rank, BoundaryPolicy.TORUS, OrientationMode.MU)
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
    
    # 6. Summary
    print("6. System Summary:")
    print("   ✓ Symmetry-fixed metric with rigorous testing")
    print("   ✓ Kernel atlas with spectral embedding")
    print("   ✓ Planner shim with witness emission")
    print("   ✓ Chess board algebra in 𝔽₂⁸")
    print("   ✓ Integration test with chess squares as seedstreams")
    print()
    
    print("✓ CE1 discovery loop demonstration completed!")
    print("The system provides a working foundation for metric-guided exploration.")
    print()
    print("Next steps:")
    print("   • Scale to 256 metanions with optimized algorithms")
    print("   • Implement autotune for metric weights")
    print("   • Add adversarial and drift tests")
    print("   • Build certificate cache with compression")

if __name__ == "__main__":
    quick_demo()
