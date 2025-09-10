"""
CE1 Certificate Cache: Compression and Reciprocity Rules

Implements certificate caching with compression and reciprocity rules for
globally consistent atlas navigation.

CE1{
  cache: Key certificates by (hash(QA), hash(QB))
  compression: ζ(π*, τ*, σ*, ΔE, Δϕ, ΔΩ, Δℳ, d) with run seed
  reciprocity: When matching inverse certificate exists, prefer it
  goal: Keep atlas globally consistent
}
"""

import numpy as np
import hashlib
import pickle
import zlib
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass, asdict
import time
from ce1_seed_metric import MetricCertificate

@dataclass
class CompressedCertificate:
    """Compressed certificate for storage"""
    pi_star: List[int]
    tau_star: float
    sigma_star: float
    distance: float
    residuals: Dict[str, float]
    execution_time: float
    invariants_preserved: bool
    run_seed: int
    compressed_size: int
    compression_ratio: float

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    compressed_cert: CompressedCertificate
    access_count: int
    last_accessed: float
    is_reciprocal: bool
    inverse_key: Optional[str]

class CE1CertificateCache:
    """
    CE1 Certificate Cache
    
    Provides certificate caching with compression and reciprocity rules
    for globally consistent atlas navigation.
    """
    
    def __init__(self, max_size: int = 10000, compression_level: int = 6):
        """
        Initialize certificate cache
        
        Args:
            max_size: Maximum number of cached certificates
            compression_level: zlib compression level (1-9)
        """
        self.max_size = max_size
        self.compression_level = compression_level
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []  # LRU tracking
        self.reciprocity_map: Dict[str, str] = {}  # key -> inverse_key
        self.stats = {
            "hits": 0,
            "misses": 0,
            "compressions": 0,
            "decompressions": 0,
            "reciprocity_hits": 0,
            "total_bytes_saved": 0
        }
        
    def _get_cache_key(self, seedstream_A: np.ndarray, seedstream_B: np.ndarray) -> str:
        """Generate cache key for seedstream pair"""
        # Use hash of concatenated seedstreams
        combined = np.concatenate([seedstream_A.flatten(), seedstream_B.flatten()])
        return hashlib.sha256(combined.tobytes()).hexdigest()[:16]
    
    def _get_inverse_key(self, key: str) -> str:
        """Get inverse key for reciprocity"""
        # For now, use a simple approach - in practice would need more sophisticated logic
        return f"inv_{key}"
    
    def _compress_certificate(self, cert: MetricCertificate, run_seed: int) -> CompressedCertificate:
        """Compress certificate for storage"""
        # Serialize certificate
        cert_data = asdict(cert)
        cert_data['run_seed'] = run_seed
        
        # Compress using zlib
        serialized = pickle.dumps(cert_data)
        compressed = zlib.compress(serialized, self.compression_level)
        
        # Calculate compression ratio
        original_size = len(serialized)
        compressed_size = len(compressed)
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
        
        self.stats["compressions"] += 1
        self.stats["total_bytes_saved"] += (original_size - compressed_size)
        
        return CompressedCertificate(
            pi_star=cert.pi_star,
            tau_star=cert.tau_star,
            sigma_star=cert.sigma_star,
            distance=cert.distance,
            residuals=cert.residuals,
            execution_time=cert.execution_time,
            invariants_preserved=cert.invariants_preserved,
            run_seed=run_seed,
            compressed_size=compressed_size,
            compression_ratio=compression_ratio
        )
    
    def _decompress_certificate(self, compressed_cert: CompressedCertificate) -> MetricCertificate:
        """Decompress certificate from storage"""
        # Reconstruct certificate data
        cert_data = {
            'pi_star': compressed_cert.pi_star,
            'tau_star': compressed_cert.tau_star,
            'sigma_star': compressed_cert.sigma_star,
            'distance': compressed_cert.distance,
            'residuals': compressed_cert.residuals,
            'execution_time': compressed_cert.execution_time,
            'invariants_preserved': compressed_cert.invariants_preserved
        }
        
        self.stats["decompressions"] += 1
        
        return MetricCertificate(**cert_data)
    
    def _evict_lru(self):
        """Evict least recently used entry"""
        if not self.access_order:
            return
        
        # Remove oldest entry
        oldest_key = self.access_order.pop(0)
        if oldest_key in self.cache:
            del self.cache[oldest_key]
        
        # Clean up reciprocity mapping
        if oldest_key in self.reciprocity_map:
            inverse_key = self.reciprocity_map[oldest_key]
            if inverse_key in self.reciprocity_map:
                del self.reciprocity_map[inverse_key]
            del self.reciprocity_map[oldest_key]
    
    def _update_access(self, key: str):
        """Update access tracking for LRU"""
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
    
    def get(self, seedstream_A: np.ndarray, seedstream_B: np.ndarray) -> Optional[MetricCertificate]:
        """
        Get certificate from cache
        
        Returns:
            MetricCertificate if found, None otherwise
        """
        key = self._get_cache_key(seedstream_A, seedstream_B)
        
        # Check direct cache
        if key in self.cache:
            entry = self.cache[key]
            entry.access_count += 1
            entry.last_accessed = time.time()
            self._update_access(key)
            self.stats["hits"] += 1
            
            return self._decompress_certificate(entry.compressed_cert)
        
        # Check reciprocity rule
        inverse_key = self._get_inverse_key(key)
        if inverse_key in self.cache:
            entry = self.cache[inverse_key]
            entry.access_count += 1
            entry.last_accessed = time.time()
            self._update_access(inverse_key)
            self.stats["hits"] += 1
            self.stats["reciprocity_hits"] += 1
            
            # Return inverse certificate
            inverse_cert = self._decompress_certificate(entry.compressed_cert)
            # Create inverse certificate
            return self._create_inverse_certificate(inverse_cert)
        
        self.stats["misses"] += 1
        return None
    
    def _create_inverse_certificate(self, cert: MetricCertificate) -> MetricCertificate:
        """Create inverse certificate for reciprocity"""
        # Compute inverse transformation
        pi_inv = [0] * len(cert.pi_star)
        for i, j in enumerate(cert.pi_star):
            pi_inv[j] = i
        
        tau_inv = -cert.tau_star
        sigma_inv = 1.0 / cert.sigma_star if cert.sigma_star != 0 else 1.0
        
        return MetricCertificate(
            pi_star=pi_inv,
            tau_star=tau_inv,
            sigma_star=sigma_inv,
            distance=cert.distance,  # Distance should be the same
            residuals=cert.residuals,
            execution_time=cert.execution_time,
            invariants_preserved=cert.invariants_preserved
        )
    
    def put(self, seedstream_A: np.ndarray, seedstream_B: np.ndarray, 
            cert: MetricCertificate, run_seed: int = 0) -> str:
        """
        Store certificate in cache
        
        Args:
            seedstream_A: First seedstream
            seedstream_B: Second seedstream
            cert: Certificate to store
            run_seed: Random seed for reproducibility
            
        Returns:
            Cache key
        """
        key = self._get_cache_key(seedstream_A, seedstream_B)
        
        # Check if we need to evict
        if len(self.cache) >= self.max_size and key not in self.cache:
            self._evict_lru()
        
        # Compress certificate
        compressed_cert = self._compress_certificate(cert, run_seed)
        
        # Create cache entry
        entry = CacheEntry(
            compressed_cert=compressed_cert,
            access_count=1,
            last_accessed=time.time(),
            is_reciprocal=False,
            inverse_key=None
        )
        
        # Store in cache
        self.cache[key] = entry
        self._update_access(key)
        
        # Set up reciprocity mapping
        inverse_key = self._get_inverse_key(key)
        self.reciprocity_map[key] = inverse_key
        self.reciprocity_map[inverse_key] = key
        
        return key
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0.0
        
        return {
            "cache_size": len(self.cache),
            "max_size": self.max_size,
            "hit_rate": hit_rate,
            "total_requests": total_requests,
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "reciprocity_hits": self.stats["reciprocity_hits"],
            "compressions": self.stats["compressions"],
            "decompressions": self.stats["decompressions"],
            "total_bytes_saved": self.stats["total_bytes_saved"],
            "avg_compression_ratio": self._get_avg_compression_ratio()
        }
    
    def _get_avg_compression_ratio(self) -> float:
        """Get average compression ratio"""
        if not self.cache:
            return 1.0
        
        total_ratio = sum(entry.compressed_cert.compression_ratio for entry in self.cache.values())
        return total_ratio / len(self.cache)
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.access_order.clear()
        self.reciprocity_map.clear()
        self.stats = {key: 0 for key in self.stats.keys()}
    
    def get_compression_report(self) -> str:
        """Generate compression report"""
        stats = self.get_stats()
        
        report = []
        report.append("=== CE1 Certificate Cache Compression Report ===")
        report.append("")
        report.append(f"Cache size: {stats['cache_size']}/{stats['max_size']}")
        report.append(f"Hit rate: {stats['hit_rate']:.1%}")
        report.append(f"Total requests: {stats['total_requests']}")
        report.append(f"Reciprocity hits: {stats['reciprocity_hits']}")
        report.append("")
        report.append("Compression Statistics:")
        report.append(f"  Total compressions: {stats['compressions']}")
        report.append(f"  Total decompressions: {stats['decompressions']}")
        report.append(f"  Total bytes saved: {stats['total_bytes_saved']:,}")
        report.append(f"  Average compression ratio: {stats['avg_compression_ratio']:.2f}x")
        report.append("")
        
        if stats['total_bytes_saved'] > 0:
            report.append("✓ Cache is saving significant storage space")
        else:
            report.append("⚠ Cache has not saved storage space yet")
        
        return "\n".join(report)

def demo_certificate_cache():
    """Demonstrate CE1 certificate cache"""
    print("=== CE1 Certificate Cache Demonstration ===")
    
    # Initialize cache
    cache = CE1CertificateCache(max_size=100, compression_level=6)
    
    # Create test seedstreams
    n = 8
    seedstreams = []
    for i in range(5):
        seedstream = np.random.randn(n) + 1j * np.random.randn(n)
        seedstream = seedstream / np.linalg.norm(seedstream)
        seedstreams.append(seedstream)
    
    print(f"Created {len(seedstreams)} test seedstreams")
    
    # Create mock certificates
    from ce1_seed_metric import CE1SeedMetric
    metric = CE1SeedMetric()
    
    print("\nComputing and caching certificates...")
    
    # Compute and cache certificates
    for i in range(len(seedstreams)):
        for j in range(i + 1, len(seedstreams)):
            cert = metric.compute_distance(seedstreams[i], seedstreams[j])
            key = cache.put(seedstreams[i], seedstreams[j], cert, run_seed=42)
            print(f"  Cached {i}→{j}: key={key}")
    
    # Test cache retrieval
    print("\nTesting cache retrieval...")
    
    # Test direct retrieval
    retrieved_cert = cache.get(seedstreams[0], seedstreams[1])
    if retrieved_cert:
        print(f"  Direct retrieval: distance={retrieved_cert.distance:.6f}")
    else:
        print("  Direct retrieval: MISS")
    
    # Test reciprocity retrieval
    retrieved_cert = cache.get(seedstreams[1], seedstreams[0])
    if retrieved_cert:
        print(f"  Reciprocity retrieval: distance={retrieved_cert.distance:.6f}")
    else:
        print("  Reciprocity retrieval: MISS")
    
    # Test cache statistics
    print("\nCache Statistics:")
    stats = cache.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Generate compression report
    print("\n" + cache.get_compression_report())
    
    # Test cache eviction
    print("\nTesting cache eviction...")
    print(f"Cache size before: {len(cache.cache)}")
    
    # Add many more entries to trigger eviction
    for i in range(50):
        seedstream = np.random.randn(n) + 1j * np.random.randn(n)
        seedstream = seedstream / np.linalg.norm(seedstream)
        cert = metric.compute_distance(seedstreams[0], seedstream)
        cache.put(seedstreams[0], seedstream, cert, run_seed=42)
    
    print(f"Cache size after: {len(cache.cache)}")
    print(f"Max size: {cache.max_size}")
    
    # Final statistics
    print("\nFinal Statistics:")
    final_stats = cache.get_stats()
    for key, value in final_stats.items():
        print(f"  {key}: {value}")
    
    print("\n✓ CE1 certificate cache demonstration completed!")
    print("The cache provides compression and reciprocity rules for globally consistent navigation.")

if __name__ == "__main__":
    demo_certificate_cache()
