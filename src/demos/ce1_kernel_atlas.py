"""
CE1 Kernel Atlas: Self-Tuning Spectral Embedding with Diffusion-Time Scaling

Builds self-tuning kernel matrix with local bandwidths and diffusion-time scaling
for robust "local gossip → global myth" navigation.

CE1{
  seed: QL-KernelAtlas
  lens: PK-diag + self-tuning + diffusion-time
  goal: build K with local λᵢ via k-NN; diffusion map with time t
  output: {K, coords, clusters, t, k_local}
}
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
import time
from numpy.linalg import eigh
from sklearn.cluster import KMeans
from ce1_seed_metric import CE1SeedMetric, MetricCertificate

@dataclass
class KernelAtlas:
    """Kernel atlas with spectral coordinates and diffusion-time scaling"""
    K: np.ndarray                    # Self-tuning kernel matrix
    coords: np.ndarray              # Spectral coordinates (top eigenvectors)
    clusters: np.ndarray            # Cluster assignments
    lambda_param: float             # Legacy: median bandwidth (for compatibility)
    eigenvalues: np.ndarray         # Eigenvalues
    seedstreams: List[np.ndarray]   # Original seedstreams
    certificates: List[MetricCertificate]  # Distance certificates
    # New self-tuning parameters
    k_local: int                    # Local neighborhood size
    t: float                        # Diffusion time parameter
    mode: str                       # "diffusion" or "spectral"
    sigma: np.ndarray               # Local bandwidths

# Core self-tuning kernel functions

def pairwise_sq_dists(X: np.ndarray) -> np.ndarray:
    """Compute pairwise squared distances efficiently"""
    # X: (n, d)
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    n = X.shape[0]
    norms = np.sum(X**2, axis=1, keepdims=True)
    D2 = norms + norms.T - 2 * (X @ X.T)
    np.fill_diagonal(D2, 0.0)
    # numerical cleanup
    D2[D2 < 0] = 0.0
    return D2

def local_bandwidths(D2: np.ndarray, k: int) -> np.ndarray:
    """Compute local bandwidths using k-NN distances"""
    # sigma_i = distance to k-th nearest neighbor of i
    n = D2.shape[0]
    # sort each row; skip self at 0
    idx = np.argpartition(D2, kth=min(k, n-1), axis=1)
    kth = min(k, n-1)
    # gather k-th smallest positive distance (sqrt at end)
    sigma2 = D2[np.arange(n), idx[:, kth]]
    sigma = np.sqrt(np.maximum(sigma2, 1e-18))
    return sigma

def self_tuning_kernel(D2: np.ndarray, sigma: np.ndarray, eps: float = 1e-18) -> np.ndarray:
    """Build self-tuning kernel with local bandwidths"""
    # K_ij = exp( - d2_ij / (sigma_i * sigma_j + eps) )
    denom = sigma[:, None] * sigma[None, :] + eps
    K = np.exp(-D2 / denom)
    # safety: zero numerical garbage on diagonal; keep self-loops = 1.0
    np.fill_diagonal(K, 1.0)
    return K

def diffusion_embedding(K: np.ndarray, n_components: int = 3, t: float = 1.0):
    """Diffusion map embedding with time scaling"""
    # Build random-walk P = D^{-1} K
    d = K.sum(axis=1)
    # guard isolated points
    d = np.where(d <= 1e-18, 1e-18, d)
    P = K / d[:, None]

    # For stable real symmetric eigen-decomp, use similarity S = D^{1/2} P D^{-1/2} = D^{-1/2} K D^{-1/2}
    Dm12 = 1.0 / np.sqrt(d)
    S = (Dm12[:, None] * K) * Dm12[None, :]

    # Symmetric eigendecomposition
    # eigenvalues are in ascending order from eigh; take largest
    vals, vecs = eigh(S)
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]

    # Map back to right eigenvectors of P: psi = D^{-1/2} * v
    psi = (vecs * Dm12[:, None])

    # Drop the trivial stationary component if desired; here we keep top n_components
    m = min(n_components, psi.shape[1])
    lambdas = np.clip(vals[:m], 0.0, 1.0)  # numerical guard
    # Diffusion time scaling
    lambdas_t = lambdas**t
    coords = psi[:, :m] * lambdas_t  # scale columns

    return coords, lambdas, P

def spectral_clustering_embedding(K: np.ndarray, n_components: int = 3):
    """Spectral clustering embedding (Ng-Jordan-Weiss)"""
    # Symmetric normalized Laplacian approach via S = D^{-1/2} K D^{-1/2}
    d = K.sum(axis=1)
    d = np.where(d <= 1e-18, 1e-18, d)
    Dm12 = 1.0 / np.sqrt(d)
    S = (Dm12[:, None] * K) * Dm12[None, :]
    vals, vecs = eigh(S)
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]
    U = vecs[:, :n_components]
    # Row-normalize U before k-means (Ng–Jordan–Weiss)
    row_norm = np.linalg.norm(U, axis=1, keepdims=True)
    row_norm[row_norm == 0] = 1.0
    Y = U / row_norm
    return Y, vals

def build_ce1_kernel_atlas(
    X: np.ndarray,
    k: int = 7,
    t: float = 1.0,
    n_components: int = 3,
    n_clusters: Optional[int] = None,
    mode: str = "diffusion",  # "diffusion" | "spectral"
):
    """
    Build self-tuning kernel atlas with diffusion-time scaling
    
    Args:
        X: (n, d) seedstream coordinates or features
        k: neighbors for local bandwidth; use max(2, min(k, n-1))
        t: diffusion time (>=0). t=0 -> identity embedding; t>1 smooths/zooms out.
        n_components: embedding dimension
        n_clusters: if None, pick via simple eigengap; else KMeans with given k
        mode: "diffusion" (diffusion maps) or "spectral" (Ng–Jordan–Weiss)
    """
    n = X.shape[0]
    if n <= 1:
        raise ValueError("Need at least 2 points for an atlas.")

    k_eff = max(2, min(k, n - 1))
    D2 = pairwise_sq_dists(X)
    sigma = local_bandwidths(D2, k=k_eff)
    K = self_tuning_kernel(D2, sigma)

    if mode == "diffusion":
        coords, evals, _P = diffusion_embedding(K, n_components=n_components, t=t)
        eigvals_for_k = evals
    elif mode == "spectral":
        coords, evals = spectral_clustering_embedding(K, n_components=n_components)
        eigvals_for_k = evals
    else:
        raise ValueError("mode must be 'diffusion' or 'spectral'")

    # crude eigengap heuristic if clusters not provided
    if n_clusters is None:
        # look at top 10 or n-1 eigenvalues; pick gap
        m = min(10, len(eigvals_for_k) - 1)
        lam = eigvals_for_k[: m + 1]
        gaps = lam[:-1] - lam[1:]
        k_hat = int(np.argmax(gaps[1:]) + 1) if m >= 2 else 2  # skip the trivial first gap
        n_clusters = max(2, k_hat)

    labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit_predict(coords)

    atlas = {
        "kernel": K,
        "coords": coords,
        "eigvals": eigvals_for_k,
        "labels": labels,
        "k_local": k_eff,
        "t": float(t),
        "mode": mode,
        "sigma": sigma,
    }
    return atlas

class CE1KernelAtlas:
    """
    CE1 Kernel Atlas implementation
    
    Builds navigable manifold for exploration and memory retrieval
    using spectral embedding of gauge-invariant distances.
    """
    
    def __init__(self, metric: Optional[CE1SeedMetric] = None):
        """
        Initialize kernel atlas
        
        Args:
            metric: CE1 seed metric instance
        """
        self.metric = metric or CE1SeedMetric()
        self.atlas = None
        
    def build_atlas(self, seedstreams: List[np.ndarray], 
                   k: int = 7,
                   t: float = 1.0,
                   n_eigenvectors: int = 3,
                   n_clusters: Optional[int] = None,
                   mode: str = "diffusion",
                   hamiltonian: Optional[np.ndarray] = None) -> KernelAtlas:
        """
        Build self-tuning kernel atlas from seedstreams
        
        Args:
            seedstreams: List of seedstreams to embed
            k: Local neighborhood size for bandwidth computation
            t: Diffusion time parameter (t=0 identity, t>1 smooths)
            n_eigenvectors: Number of top eigenvectors for embedding
            n_clusters: Number of clusters (if None, auto-detect via eigengap)
            mode: "diffusion" or "spectral" embedding mode
            hamiltonian: Hamiltonian for metric computation
            
        Returns:
            KernelAtlas with self-tuning spectral coordinates and clusters
        """
        start_time = time.time()
        n = len(seedstreams)
        
        print(f"Building self-tuning kernel atlas for {n} seedstreams...")
        print(f"Parameters: k={k}, t={t}, mode={mode}")
        
        # Step 1: Compute pairwise distances using CE1 metric
        print("Computing pairwise distances...")
        D = np.zeros((n, n))
        certificates = []
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    D[i, j] = 0.0
                    cert = MetricCertificate(
                        pi_star=[], tau_star=0.0, sigma_star=1.0,
                        distance=0.0, residuals={}, execution_time=0.0,
                        invariants_preserved=True
                    )
                else:
                    cert = self.metric.compute_distance(seedstreams[i], seedstreams[j], hamiltonian)
                    D[i, j] = cert.distance
                    D[j, i] = cert.distance  # Symmetry
                
                certificates.append((i, j, cert))
        
        # Step 2: Build self-tuning kernel atlas
        print("Building self-tuning kernel...")
        atlas_dict = build_ce1_kernel_atlas(
            X=D,  # Use distance matrix as features
            k=k,
            t=t,
            n_components=n_eigenvectors,
            n_clusters=n_clusters,
            mode=mode
        )
        
        # Legacy compatibility: compute median bandwidth
        upper_tri = D[np.triu_indices_from(D, k=1)]
        lambda_param = np.median(upper_tri)
        
        execution_time = time.time() - start_time
        
        # Create atlas with new parameters
        atlas = KernelAtlas(
            K=atlas_dict["kernel"],
            coords=atlas_dict["coords"],
            clusters=atlas_dict["labels"],
            lambda_param=lambda_param,  # Legacy compatibility
            eigenvalues=atlas_dict["eigvals"],
            seedstreams=seedstreams,
            certificates=certificates,
            # New self-tuning parameters
            k_local=atlas_dict["k_local"],
            t=atlas_dict["t"],
            mode=atlas_dict["mode"],
            sigma=atlas_dict["sigma"]
        )
        
        self.atlas = atlas
        
        print(f"Atlas built in {execution_time:.4f}s")
        print(f"Kernel matrix shape: {atlas_dict['kernel'].shape}")
        print(f"Spectral coordinates shape: {atlas_dict['coords'].shape}")
        print(f"Number of clusters: {len(np.unique(atlas_dict['labels']))}")
        print(f"Top eigenvalues: {atlas_dict['eigvals'][:3]}")
        print(f"Local bandwidths (σ): min={atlas_dict['sigma'].min():.4f}, max={atlas_dict['sigma'].max():.4f}")
        
        return atlas
    
    def get_navigation_coords(self, seedstream: np.ndarray, 
                            hamiltonian: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get navigation coordinates for a new seedstream
        
        Projects new seedstream into the atlas coordinate system
        """
        if self.atlas is None:
            raise ValueError("Atlas not built yet. Call build_atlas() first.")
        
        # Compute distances to all atlas seedstreams
        distances = []
        for atlas_seedstream in self.atlas.seedstreams:
            cert = self.metric.compute_distance(seedstream, atlas_seedstream, hamiltonian)
            distances.append(cert.distance)
        
        distances = np.array(distances)
        
        # Compute kernel values
        kernel_values = np.exp(-distances**2 / self.atlas.lambda_param**2)
        
        # Project onto spectral coordinates
        coords = self.atlas.coords.T @ kernel_values
        
        return coords
    
    def find_nearest_neighbors(self, seedstream: np.ndarray, k: int = 5,
                             hamiltonian: Optional[np.ndarray] = None) -> List[Tuple[int, float]]:
        """
        Find k nearest neighbors in the atlas
        
        Returns list of (index, distance) tuples
        """
        if self.atlas is None:
            raise ValueError("Atlas not built yet. Call build_atlas() first.")
        
        # Compute distances to all atlas seedstreams
        distances = []
        for i, atlas_seedstream in enumerate(self.atlas.seedstreams):
            cert = self.metric.compute_distance(seedstream, atlas_seedstream, hamiltonian)
            distances.append((i, cert.distance))
        
        # Sort by distance
        distances.sort(key=lambda x: x[1])
        
        return distances[:k]
    
    def get_cluster_info(self) -> Dict[int, Dict[str, Any]]:
        """
        Get information about each cluster
        
        Returns dictionary with cluster statistics
        """
        if self.atlas is None:
            raise ValueError("Atlas not built yet. Call build_atlas() first.")
        
        cluster_info = {}
        unique_clusters = np.unique(self.atlas.clusters)
        
        for cluster_id in unique_clusters:
            mask = self.atlas.clusters == cluster_id
            cluster_coords = self.atlas.coords[mask]
            
            # Compute cluster statistics
            center = np.mean(cluster_coords, axis=0)
            spread = np.std(cluster_coords, axis=0)
            
            cluster_info[cluster_id] = {
                "size": np.sum(mask),
                "center": center,
                "spread": spread,
                "indices": np.where(mask)[0].tolist()
            }
        
        return cluster_info
    
    def visualize_atlas(self) -> str:
        """
        Generate ASCII visualization of the atlas
        
        Returns ASCII representation of the spectral embedding
        """
        if self.atlas is None:
            return "Atlas not built yet."
        
        coords = self.atlas.coords
        clusters = self.atlas.clusters
        
        # Create simple 2D visualization (using first 2 coordinates)
        if coords.shape[1] < 2:
            return "Need at least 2 spectral coordinates for visualization."
        
        # Normalize coordinates to [0, 1]
        x_coords = (coords[:, 0] - coords[:, 0].min()) / (coords[:, 0].max() - coords[:, 0].min())
        y_coords = (coords[:, 1] - coords[:, 1].min()) / (coords[:, 1].max() - coords[:, 1].min())
        
        # Create grid
        grid_size = 20
        grid = [[' ' for _ in range(grid_size)] for _ in range(grid_size)]
        
        # Place points on grid
        for i, (x, y, cluster) in enumerate(zip(x_coords, y_coords, clusters)):
            grid_x = int(x * (grid_size - 1))
            grid_y = int(y * (grid_size - 1))
            
            # Use different symbols for different clusters
            symbols = ['●', '▲', '■', '◆', '★']
            symbol = symbols[cluster % len(symbols)]
            grid[grid_y][grid_x] = symbol
        
        # Create visualization
        lines = []
        lines.append("CE1 Kernel Atlas - Spectral Embedding")
        lines.append("=" * 40)
        lines.append("")
        
        for row in reversed(grid):  # Reverse to match coordinate system
            lines.append(''.join(row))
        
        lines.append("")
        lines.append("Legend:")
        unique_clusters = np.unique(clusters)
        for i, cluster_id in enumerate(unique_clusters):
            symbol = symbols[cluster_id % len(symbols)]
            size = np.sum(clusters == cluster_id)
            lines.append(f"  {symbol} Cluster {cluster_id}: {size} points")
        
        return '\n'.join(lines)

def demo_kernel_atlas():
    """Demonstrate CE1 kernel atlas"""
    print("=== CE1 Kernel Atlas Demonstration ===")
    
    # Create 6 test seedstreams
    n_dim = 8
    n_seedstreams = 6
    
    print(f"Creating {n_seedstreams} test seedstreams...")
    seedstreams = []
    
    # Create base seedstream
    base_seedstream = np.random.randn(n_dim) + 1j * np.random.randn(n_dim)
    base_seedstream = base_seedstream / np.linalg.norm(base_seedstream)
    seedstreams.append(base_seedstream)
    
    # Create gauge-equivalent variants
    for i in range(n_seedstreams - 1):
        # Apply random gauge transformation
        random_perm = np.random.permutation(n_dim)
        random_phase = np.random.uniform(0, 2*np.pi)
        random_scale = np.random.uniform(0.5, 2.0)
        
        variant = base_seedstream[random_perm] * np.exp(1j * random_phase) * random_scale
        variant = variant / np.linalg.norm(variant)
        seedstreams.append(variant)
    
    # Build atlas with self-tuning
    atlas_builder = CE1KernelAtlas()
    atlas = atlas_builder.build_atlas(
        seedstreams, 
        k=3,  # Local neighborhood
        t=1.0,  # Diffusion time
        n_eigenvectors=3,
        mode="diffusion"
    )
    
    # Display atlas information
    print("\nAtlas Information:")
    print(f"  Kernel matrix shape: {atlas.K.shape}")
    print(f"  Spectral coordinates shape: {atlas.coords.shape}")
    print(f"  Number of clusters: {len(np.unique(atlas.clusters))}")
    print(f"  Local bandwidths (σ): min={atlas.sigma.min():.4f}, max={atlas.sigma.max():.4f}")
    print(f"  Diffusion time t: {atlas.t}")
    print(f"  Mode: {atlas.mode}")
    print(f"  Legacy bandwidth λ: {atlas.lambda_param:.6f}")
    
    # Show cluster information
    cluster_info = atlas_builder.get_cluster_info()
    print("\nCluster Information:")
    for cluster_id, info in cluster_info.items():
        print(f"  Cluster {cluster_id}: {info['size']} points")
        print(f"    Center: {info['center']}")
        print(f"    Spread: {info['spread']}")
    
    # Visualize atlas
    print("\nAtlas Visualization:")
    print(atlas_builder.visualize_atlas())
    
    # Test navigation
    print("\nTesting navigation...")
    test_seedstream = np.random.randn(n_dim) + 1j * np.random.randn(n_dim)
    test_seedstream = test_seedstream / np.linalg.norm(test_seedstream)
    
    nav_coords = atlas_builder.get_navigation_coords(test_seedstream)
    print(f"Navigation coordinates: {nav_coords}")
    
    nearest = atlas_builder.find_nearest_neighbors(test_seedstream, k=3)
    print("Nearest neighbors:")
    for idx, dist in nearest:
        print(f"  Seedstream {idx}: distance = {dist:.6f}")
    
    print("\n✓ CE1 kernel atlas demonstration completed!")
    print("The atlas provides navigable manifold for exploration and memory retrieval.")

if __name__ == "__main__":
    demo_kernel_atlas()
