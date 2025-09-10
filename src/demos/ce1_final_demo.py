"""
CE1 Final Demo: Complete Discovery Loop

Demonstrates the complete CE1 system with all components integrated:
1. Symmetry-fixed metric with rigorous testing
2. Kernel atlas with spectral embedding
3. Planner shim with witness emission and certificate cache
4. Barycenter computation for consensus passports
5. Chess board algebra in 𝔽₂⁸
6. Certificate cache with compression and reciprocity rules

This is the complete "Week-Zero → Week-One" progression from metric to machine.
"""

import numpy as np
import time
from ce1_seed_metric import CE1SeedMetric
from ce1_kernel_atlas import CE1KernelAtlas
from ce1_planner_shim import CE1PlannerShim
from ce1_barycenter import CE1Barycenter
from ce1_chess_algebra import CE1ChessAlgebra, BoundaryPolicy, OrientationMode
from ce1_certificate_cache import CE1CertificateCache

def final_demo():
    """Final comprehensive demonstration of CE1 system"""
    print("=== CE1 Final Demo: Complete Discovery Loop ===")
    print("Week-Zero → Week-One: from metric to machine")
    print()
    
    # Initialize all components
    print("Initializing CE1 components...")
    metric = CE1SeedMetric()
    cache = CE1CertificateCache(max_size=500, compression_level=6)
    atlas_builder = CE1KernelAtlas(metric)
    planner = CE1PlannerShim(metric, cache=cache)
    barycenter_computer = CE1Barycenter(metric)
    chess_algebra = CE1ChessAlgebra()
    
    print("✓ All components initialized")
    print()
    
    # 1. Symmetry-fixed metric with rigorous testing
    print("1. Testing symmetry-fixed metric...")
    symmetry_results = metric.test_symmetry_rigorous(n_tests=100, n_dim=8)
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
    
    # 3. Planner shim with certificate cache
    print("3. Testing planner shim with certificate cache...")
    initial_seedstream = seedstreams[0]
    target_seedstream = seedstreams[-1]
    
    # First search (will populate cache)
    result1 = planner.greedy_search(initial_seedstream, target_seedstream, max_iterations=3)
    print(f"   First search: target_reached={result1.target_reached}, final_distance={result1.final_distance:.6f}")
    
    # Second search (should use cache)
    result2 = planner.greedy_search(initial_seedstream, target_seedstream, max_iterations=3)
    print(f"   Second search: target_reached={result2.target_reached}, final_distance={result2.final_distance:.6f}")
    
    # Show cache statistics
    cache_stats = cache.get_stats()
    print(f"   Cache hit rate: {cache_stats['hit_rate']:.1%}")
    print(f"   Total bytes saved: {cache_stats['total_bytes_saved']:,}")
    print(f"   Average compression ratio: {cache_stats['avg_compression_ratio']:.2f}x")
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
    
    # 6. Integration test: Chess squares as seedstreams
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
    
    # 7. Certificate cache demonstration
    print("7. Certificate cache demonstration...")
    
    # Show cache compression report
    compression_report = cache.get_compression_report()
    print(compression_report)
    
    # Test cache with chess squares
    print("   Testing cache with chess squares...")
    chess_planner = CE1PlannerShim(metric, cache=cache)
    
    # Compute distance between chess squares
    cert = chess_planner.metric_compare(chess_seedstreams[0], chess_seedstreams[1])
    print(f"   Distance between chess squares: {cert[0]:.6f}")
    
    # Show final cache statistics
    final_cache_stats = cache.get_stats()
    print(f"   Final cache size: {final_cache_stats['cache_size']}")
    print(f"   Final hit rate: {final_cache_stats['hit_rate']:.1%}")
    print(f"   Total bytes saved: {final_cache_stats['total_bytes_saved']:,}")
    print()
    
    # 8. System Summary
    print("8. System Summary:")
    print("   ✓ Symmetry-fixed metric with rigorous testing")
    print("   ✓ Kernel atlas with spectral embedding")
    print("   ✓ Planner shim with witness emission")
    print("   ✓ Barycenter computation for consensus passports")
    print("   ✓ Chess board algebra in 𝔽₂⁸")
    print("   ✓ Certificate cache with compression and reciprocity rules")
    print("   ✓ Integration test with chess squares as seedstreams")
    print()
    
    print("✓ CE1 discovery loop demonstration completed!")
    print("The system provides a complete foundation for metric-guided exploration.")
    print()
    print("Next steps for Week-One:")
    print("   • Scale to 256 metanions with optimized algorithms")
    print("   • Implement autotune for metric weights")
    print("   • Add adversarial and drift tests")
    print("   • Build consensus passports for training anchors")
    print("   • Create navigable manifold for exploration and memory retrieval")

if __name__ == "__main__":
    final_demo()
