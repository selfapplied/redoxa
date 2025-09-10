"""
CE1 Backbone Report Harmonizer

Treats each report as a local canonical line (guild backbone), while maintaining 
a reversible projection to the global canonical line ℒ_global = ℜ = ½.

This enables synergy to have teeth system-wide while preserving local guild truth.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import hashlib

class GaugeType(Enum):
    """Permutation gauge types"""
    IDENTITY = "π_id"
    OPTIMIZED = "π_opt"
    DEFAULT = "π0"

@dataclass
class CanonicalLine:
    """A canonical line with gauge information"""
    alpha: float
    beta: float
    gauge: GaugeType
    gauge_permutation: Optional[List[int]] = None

@dataclass
class FitReport:
    """Dual fit report with local and global metrics"""
    line: Dict[str, CanonicalLine]  # local, global
    distance: Dict[str, float]      # local, global
    fit: Dict[str, float]           # local, global
    witnesses: Dict[str, Dict[str, Any]]  # local, global witness data
    flags: Dict[str, bool]          # strained_projection, migrated, etc.

@dataclass
class SynergyReport:
    """Dual synergy report with local and global metrics"""
    synergy: Dict[str, float]       # local, global
    distances: Dict[str, float]     # a_local, b_local, pair_local, a_global, b_global, pair_global
    braid_energy: float
    witnesses: Dict[str, Any]       # local, global CE1 certificates

class BackboneHarmonizer:
    """
    CE1 Backbone Report Harmonizer - v0.1 Spectral Fit
    
    Real math kernel that breaks flatlines using discrete spectral probes.
    No mysticism, just minimal NumPy math that respects CE1 vibe.
    """
    
    def __init__(self, global_line: CanonicalLine = None):
        # Global canonical line: ℒ_global = ℜ = ½
        self.global_line = global_line or CanonicalLine(
            alpha=0.5,
            beta=0.0,  # Default beta
            gauge=GaugeType.DEFAULT
        )
        
        # Energy threshold for strained projections
        self.energy_threshold = 0.1
        
        # Huber norm threshold for gauge optimization
        self.huber_threshold = 0.01
        
        # Spectral fit parameters
        self.delta = 0.1  # Huber threshold
        self.min_length = 64  # Minimum vector length for FFT
    
    def detect_local_line(self, report_data: Dict[str, Any]) -> CanonicalLine:
        """Detect local canonical line from report data"""
        if "line" in report_data and "local" in report_data["line"]:
            line_data = report_data["line"]["local"]
            return CanonicalLine(
                alpha=line_data.get("alpha", 0.5),
                beta=line_data.get("beta", 0.0),
                gauge=GaugeType(line_data.get("gauge", "π_id")),
                gauge_permutation=line_data.get("gauge_permutation")
            )
        else:
            # Infer from existing report config or use defaults
            return CanonicalLine(
                alpha=0.5,  # Default alpha
                beta=0.0,   # Default beta
                gauge=GaugeType.IDENTITY
            )
    
    def build_projection(self, local_line: CanonicalLine, state: np.ndarray) -> Tuple[np.ndarray, GaugeType, List[int]]:
        """
        Build reversible projection Π_{loc→glob}
        
        Solves for affine real shift to ℜ=½ + best permutation π (min Huber norm on zeros)
        """
        # Compute affine shift to align with global line
        alpha_shift = self.global_line.alpha - local_line.alpha
        beta_shift = self.global_line.beta - local_line.beta
        
        # Apply affine transformation
        projected_state = state * np.exp(1j * alpha_shift) + beta_shift
        
        # Find best permutation gauge
        best_permutation, best_gauge = self._optimize_permutation_gauge(projected_state)
        
        return projected_state, best_gauge, best_permutation
    
    def _optimize_permutation_gauge(self, state: np.ndarray) -> Tuple[List[int], GaugeType]:
        """Optimize permutation gauge to minimize Huber norm on zeros"""
        n = len(state)
        
        # Try identity permutation first
        identity_perm = list(range(n))
        identity_loss = self._compute_huber_loss(state, identity_perm)
        
        # Try a few simple permutations
        best_perm = identity_perm
        best_loss = identity_loss
        best_gauge = GaugeType.IDENTITY
        
        # Try reverse permutation
        reverse_perm = list(reversed(range(n)))
        reverse_loss = self._compute_huber_loss(state, reverse_perm)
        
        if reverse_loss < best_loss:
            best_perm = reverse_perm
            best_loss = reverse_loss
            best_gauge = GaugeType.OPTIMIZED
        
        # Try random permutations (simplified - in practice would use Hungarian algorithm)
        for _ in range(min(10, n)):
            perm = np.random.permutation(n).tolist()
            loss = self._compute_huber_loss(state, perm)
            if loss < best_loss:
                best_perm = perm
                best_loss = loss
                best_gauge = GaugeType.OPTIMIZED
        
        return best_perm, best_gauge
    
    def seed_to_vec(self, seed: str) -> np.ndarray:
        """Convert seed to real vector in [-1, 1]"""
        # Deterministic adapter: bytes → float32
        seed_bytes = seed.encode('utf-8')
        hash_obj = hashlib.sha256(seed_bytes)
        hash_bytes = hash_obj.digest()
        
        # Convert to float32 array
        vec = np.frombuffer(hash_bytes, dtype=np.uint8).astype(np.float32)
        vec = (vec - 127.5) / 127.5  # Normalize to [-1, 1]
        
        # Pad to minimum length for FFT
        if len(vec) < self.min_length:
            vec = np.tile(vec, (self.min_length // len(vec)) + 1)
        
        return vec[:self.min_length]
    
    def analytic_signal(self, x: np.ndarray) -> np.ndarray:
        """Hilbert transform via FFT (zero-out negative freqs)"""
        fft_x = np.fft.fft(x)
        n = len(fft_x)
        
        # Zero out negative frequencies
        fft_x[n//2+1:] = 0
        if n % 2 == 0:
            fft_x[n//2] = fft_x[n//2] / 2
        
        # IFFT to get analytic signal
        x_a = np.fft.ifft(fft_x)
        return x_a
    
    def spectral_modes(self, x_a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute spectral modes z_k and weights w_k"""
        # FFT of analytic signal
        S = np.fft.fft(x_a)
        K = len(S)
        
        # Compute magnitudes and phases
        magnitudes = np.abs(S)
        phases = np.angle(S)
        
        # Energy-rank: r_k = rank(|S_k|) / (K-1)
        rank_indices = np.argsort(magnitudes)
        ranks = np.zeros_like(rank_indices)
        ranks[rank_indices] = np.arange(K)
        r_k = ranks / (K - 1) if K > 1 else np.zeros(K)
        
        # Phase: φ_k = ∠S_k / π
        phi_k = phases / np.pi
        
        # Complex modes: z_k = r_k + i*φ_k
        z_k = r_k + 1j * phi_k
        
        # Weights: w_k = |S_k| / Σ|S_j|
        w_k = magnitudes / np.sum(magnitudes) if np.sum(magnitudes) > 0 else np.ones(K) / K
        
        return z_k, w_k
    
    def distance_to_line(self, z: np.ndarray, w: np.ndarray, alpha: float, delta: float = None) -> float:
        """Weighted Huber distance to canonical line ℜ(z) = α"""
        if delta is None:
            delta = self.delta
        
        # Real parts: ℜ(z_k) - α
        real_parts = np.real(z) - alpha
        
        # Huber loss: ρ_δ(u)
        def huber_loss(u):
            abs_u = np.abs(u)
            return np.where(abs_u <= delta, 
                           0.5 * u**2, 
                           delta * (abs_u - 0.5 * delta))
        
        huber_values = huber_loss(real_parts)
        
        # Weighted average
        weighted_distance = np.sum(w * huber_values) / np.sum(w) if np.sum(w) > 0 else 0.0
        
        return weighted_distance
    
    def braid(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """Circular convolution braid: Braid = circ_conv(x₁, x₂)"""
        # Ensure same length
        min_len = min(len(x1), len(x2))
        x1_trunc = x1[:min_len]
        x2_trunc = x2[:min_len]
        
        # Circular convolution via FFT
        fft1 = np.fft.fft(x1_trunc)
        fft2 = np.fft.fft(x2_trunc)
        braid_fft = fft1 * fft2
        braid = np.fft.ifft(braid_fft)
        
        return braid
    
    def compute_real_fit_metrics(self, seed: str, alpha_local: float = None) -> Tuple[float, float, float]:
        """
        Compute real fit metrics using spectral analysis
        
        Returns: (local_fit, global_fit, local_alpha)
        """
        # Convert seed to vector
        x = self.seed_to_vec(seed)
        
        # Get analytic signal
        x_a = self.analytic_signal(x)
        
        # Compute spectral modes
        z, w = self.spectral_modes(x_a)
        
        # Determine local alpha (median of real parts)
        if alpha_local is None:
            alpha_local = np.median(np.real(z))
        
        # Compute distances
        local_distance = self.distance_to_line(z, w, alpha_local)
        global_distance = self.distance_to_line(z, w, 0.5)
        
        # Compute fits
        local_fit = 1.0 / (1.0 + local_distance)
        global_fit = 1.0 / (1.0 + global_distance)
        
        return local_fit, global_fit, alpha_local
    
    def compute_real_synergy(self, seed1: str, seed2: str) -> Tuple[float, float, float]:
        """
        Compute real synergy using spectral braid analysis
        
        Returns: (local_synergy, global_synergy, braid_energy)
        """
        # Convert seeds to vectors
        x1 = self.seed_to_vec(seed1)
        x2 = self.seed_to_vec(seed2)
        
        # Compute individual fits
        local_fit1, global_fit1, alpha1 = self.compute_real_fit_metrics(seed1)
        local_fit2, global_fit2, alpha2 = self.compute_real_fit_metrics(seed2)
        
        # Compute braid
        braid = self.braid(x1, x2)
        braid_a = self.analytic_signal(braid)
        z_braid, w_braid = self.spectral_modes(braid_a)
        
        # Compute braid distances
        alpha_local = (alpha1 + alpha2) / 2  # Average local alpha
        braid_local_distance = self.distance_to_line(z_braid, w_braid, alpha_local)
        braid_global_distance = self.distance_to_line(z_braid, w_braid, 0.5)
        
        # Compute synergies
        local_synergy = (1.0 / (1.0 + local_fit1) + 1.0 / (1.0 + local_fit2) - 
                        braid_local_distance)
        global_synergy = (1.0 / (1.0 + global_fit1) + 1.0 / (1.0 + global_fit2) - 
                         braid_global_distance)
        
        # Compute braid energy
        braid_energy = np.linalg.norm(x1) * np.linalg.norm(x2) / np.linalg.norm(braid)
        
        # Soft-clip if braid energy too high
        if braid_energy > 10.0:  # Arbitrary threshold
            local_synergy = max(local_synergy, 0.0)
            global_synergy = max(global_synergy, 0.0)
        
        return local_synergy, global_synergy, braid_energy
    
    def _compute_huber_loss(self, state: np.ndarray, permutation: List[int]) -> float:
        """Compute Huber loss for permutation gauge optimization"""
        permuted_state = state[permutation]
        
        # Huber loss on zeros (simplified)
        zeros = np.abs(permuted_state) < 1e-10
        if np.any(zeros):
            return np.sum(np.abs(permuted_state[zeros]))
        else:
            return 0.0
    
    def compute_fit_metrics(self, state: np.ndarray, line: CanonicalLine) -> Tuple[float, float]:
        """
        Compute fit metrics for a state against a canonical line
        
        Returns: (distance, fit)
        """
        # Compute distance to canonical line (simplified)
        # In practice, this would use the full CE1 distance metric
        distance = np.linalg.norm(state - line.alpha)
        
        # Fit = 1/(1 + distance)
        fit = 1.0 / (1.0 + distance)
        
        return distance, fit
    
    def compute_synergy_metrics(self, state_a: np.ndarray, state_b: np.ndarray, 
                               line: CanonicalLine) -> Tuple[float, float]:
        """
        Compute synergy metrics for a pair of states
        
        Returns: (synergy, braid_energy)
        """
        # Compute individual distances
        dist_a, _ = self.compute_fit_metrics(state_a, line)
        dist_b, _ = self.compute_fit_metrics(state_b, line)
        
        # Compute braid (simplified)
        braid_state = (state_a + state_b) / 2.0
        dist_braid, _ = self.compute_fit_metrics(braid_state, line)
        
        # Synergy = d(a) + d(b) - d(Braid(a,b))
        synergy = dist_a + dist_b - dist_braid
        
        # Braid energy (simplified)
        braid_energy = np.linalg.norm(braid_state)
        
        return synergy, braid_energy
    
    def harmonize_report(self, report_data: Dict[str, Any], state: np.ndarray) -> FitReport:
        """
        Harmonize a report with dual local/global metrics
        
        Returns a FitReport with both local and global fit information
        """
        # Detect local line
        local_line = self.detect_local_line(report_data)
        
        # Build projection to global line
        projected_state, global_gauge, global_perm = self.build_projection(local_line, state)
        
        # Compute local metrics
        local_distance, local_fit = self.compute_fit_metrics(state, local_line)
        
        # Compute global metrics
        global_distance, global_fit = self.compute_fit_metrics(projected_state, self.global_line)
        
        # Check for strained projection
        energy_delta = abs(global_fit - local_fit)
        strained_projection = energy_delta > self.energy_threshold
        
        # Create witnesses
        local_witnesses = {
            "modes_aligned": list(range(len(state))),
            "braid_energy": np.linalg.norm(state),
            "mirror": True,
            "gauge": local_line.gauge.value
        }
        
        global_witnesses = {
            "modes_aligned": global_perm,
            "braid_energy": np.linalg.norm(projected_state),
            "mirror": True,
            "gauge": global_gauge.value
        }
        
        # Create fit report
        fit_report = FitReport(
            line={
                "local": local_line,
                "global": CanonicalLine(
                    alpha=self.global_line.alpha,
                    beta=self.global_line.beta,
                    gauge=global_gauge,
                    gauge_permutation=global_perm
                )
            },
            distance={
                "local": local_distance,
                "global": global_distance
            },
            fit={
                "local": local_fit,
                "global": global_fit
            },
            witnesses={
                "local": local_witnesses,
                "global": global_witnesses
            },
            flags={
                "strained_projection": strained_projection,
                "migrated": "line" not in report_data or "local" not in report_data.get("line", {})
            }
        )
        
        return fit_report
    
    def harmonize_synergy(self, state_a: np.ndarray, state_b: np.ndarray, 
                         local_line: CanonicalLine) -> SynergyReport:
        """
        Harmonize synergy metrics for a pair of states
        
        Returns a SynergyReport with both local and global synergy information
        """
        # Build projections
        projected_a, _, _ = self.build_projection(local_line, state_a)
        projected_b, _, _ = self.build_projection(local_line, state_b)
        
        # Compute local synergy
        local_synergy, local_braid_energy = self.compute_synergy_metrics(state_a, state_b, local_line)
        
        # Compute global synergy
        global_synergy, global_braid_energy = self.compute_synergy_metrics(projected_a, projected_b, self.global_line)
        
        # Compute individual distances
        local_dist_a, _ = self.compute_fit_metrics(state_a, local_line)
        local_dist_b, _ = self.compute_fit_metrics(state_b, local_line)
        local_dist_pair, _ = self.compute_fit_metrics((state_a + state_b) / 2.0, local_line)
        
        global_dist_a, _ = self.compute_fit_metrics(projected_a, self.global_line)
        global_dist_b, _ = self.compute_fit_metrics(projected_b, self.global_line)
        global_dist_pair, _ = self.compute_fit_metrics((projected_a + projected_b) / 2.0, self.global_line)
        
        # Create synergy report
        synergy_report = SynergyReport(
            synergy={
                "local": local_synergy,
                "global": global_synergy
            },
            distances={
                "a_local": local_dist_a,
                "b_local": local_dist_b,
                "pair_local": local_dist_pair,
                "a_global": global_dist_a,
                "b_global": global_dist_b,
                "pair_global": global_dist_pair
            },
            braid_energy=local_braid_energy,
            witnesses={
                "local": f"CE1c_local_{hash(tuple(state_a))}_{hash(tuple(state_b))}",
                "global": f"CE1c_global_{hash(tuple(projected_a))}_{hash(tuple(projected_b))}"
            }
        )
        
        return synergy_report
    
    def compute_planner_score(self, fit_report: FitReport, synergy_reports: List[SynergyReport],
                             alpha: float = 0.4, beta: float = 0.3, beta_loc: float = 0.2, 
                             gamma: float = 0.1) -> float:
        """
        Compute planner score with dual local/global metrics
        
        score = α·task + β·mean(Synergy_glob) + β_loc·mean(Synergy_loc) + γ·diversity
        """
        # Task score (simplified)
        task_score = fit_report.fit["local"]
        
        # Global synergy score
        if synergy_reports:
            global_synergies = [sr.synergy["global"] for sr in synergy_reports]
            global_synergy_score = np.mean(global_synergies)
        else:
            global_synergy_score = 0.0
        
        # Local synergy score
        if synergy_reports:
            local_synergies = [sr.synergy["local"] for sr in synergy_reports]
            local_synergy_score = np.mean(local_synergies)
        else:
            local_synergy_score = 0.0
        
        # Diversity score (simplified)
        diversity_score = 1.0 - abs(fit_report.fit["local"] - fit_report.fit["global"])
        
        # Compute final score
        score = (alpha * task_score + 
                beta * global_synergy_score + 
                beta_loc * local_synergy_score + 
                gamma * diversity_score)
        
        return score
    
    def generate_harmonized_report(self, fit_report: FitReport, synergy_reports: List[SynergyReport]) -> str:
        """Generate a harmonized report with dual metrics"""
        report = f"""# Harmonized Report

## Local vs Global Metrics

| Metric | Local | Global | Delta |
|--------|-------|--------|-------|
| Fit | {fit_report.fit['local']:.4f} | {fit_report.fit['global']:.4f} | {fit_report.fit['global'] - fit_report.fit['local']:.4f} |
| Distance | {fit_report.distance['local']:.4f} | {fit_report.distance['global']:.4f} | {fit_report.distance['global'] - fit_report.distance['local']:.4f} |

## Line Configuration

**Local Line**: α={fit_report.line['local'].alpha:.3f}, β={fit_report.line['local'].beta:.3f}, gauge={fit_report.line['local'].gauge.value}
**Global Line**: α={fit_report.line['global'].alpha:.3f}, β={fit_report.line['global'].beta:.3f}, gauge={fit_report.line['global'].gauge.value}

## Synergy Metrics

"""
        
        if synergy_reports:
            for i, sr in enumerate(synergy_reports):
                report += f"""**Pair {i+1}**:
- Local Synergy: {sr.synergy['local']:.4f}
- Global Synergy: {sr.synergy['global']:.4f}
- Braid Energy: {sr.braid_energy:.4f}

"""
        else:
            report += "No synergy data available.\n"
        
        # Add flags
        if fit_report.flags.get("strained_projection"):
            report += "\n⚠️ **Strained Projection**: Energy delta exceeds threshold\n"
        
        if fit_report.flags.get("migrated"):
            report += "\n🔄 **Migrated**: Report migrated from legacy format\n"
        
        return report
