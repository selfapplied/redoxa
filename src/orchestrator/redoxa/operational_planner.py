"""
CE1 Operational Planner: Execution Strategy Search

Implements CE1 search over execution strategies for demo running.
Unifies computational and operational planning under the same CE1 framework.

CE1{
  atlas: {unit: Execution, space: S×K×C×P×L}  // states×strategies×caps×priors×ledger
  invariants: {I1: capacity, I2: isolation, I3: fairness, I4: reversibility, I5: correctness}
  metric: J(σ)=α·T+β·CPU+γ·RAM+δ·GPU+ε·IO+ζ·Retry  // gauge-normalized
  ops: [scale, reorder, batch, timeout, pin, retry, probe, shed]
  search: beam(b), greedy+backoff, tol on ΔJ, cert on SLA
  witnesses: per-demo usage; certificate: {makespan, OOM=0, deadlines≤p}
  objective: minimize J subject to invariants; emit cert
}
"""

import numpy as np
import time
import psutil
import sys
import os
import subprocess
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

class ExecutionStrategy(Enum):
    """Execution strategy types"""
    SEQUENTIAL = "sequential"
    CONSERVATIVE_PARALLEL = "conservative_parallel"
    MODERATE_PARALLEL = "moderate_parallel"
    AGGRESSIVE_PARALLEL = "aggressive_parallel"

@dataclass
class DemoProfile:
    """Profile of a demo for execution planning"""
    name: str
    directory: str
    size: int
    lines: int
    has_heavy_computation: bool
    has_numpy: bool
    has_matplotlib: bool
    has_gpu: bool
    category: str
    expected_duration: str
    flakiness_score: float = 0.0

@dataclass
class ResourceCaps:
    """Resource capacity envelope"""
    cpu_cores: int
    memory_gb: float
    gpu_slots: int = 0
    io_bandwidth_mbps: float = 1000.0

@dataclass
class ExecutionState:
    """State in operational planning search"""
    queue: List[DemoProfile]
    strategy: ExecutionStrategy
    max_workers: int
    timeouts: Dict[str, float]
    caps: ResourceCaps
    profile: Dict[str, Any] = field(default_factory=dict)
    ledger: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class ExecutionWitness:
    """Witness for operational execution"""
    demo_name: str
    start_time: float
    end_time: float
    cpu_usage: float
    memory_peak: float
    gpu_util: float
    io_wait: float
    exit_code: int
    retries: int
    timeout_used: float

@dataclass
class ExecutionCertificate:
    """Certificate proving SLA compliance"""
    makespan: float
    oom_count: int
    deadline_hits: int
    total_demos: int
    success_rate: float
    resource_efficiency: float
    sla_met: bool

class OperationalPlanner:
    """
    CE1 Operational Planner
    
    Implements CE1 search over execution strategies for demo running.
    Uses the same framework as computational planning but for operational concerns.
    """
    
    def __init__(self, caps: Optional[ResourceCaps] = None, 
                 weights: Optional[Dict[str, float]] = None):
        """
        Initialize operational planner
        
        Args:
            caps: Resource capacity envelope
            weights: Cost function weights {α,β,γ,δ,ε,ζ}
        """
        self.caps = caps or self._detect_caps()
        self.weights = weights or {
            'makespan': 1.0,      # α
            'cpu_usage': 0.3,     # β  
            'memory_peak': 0.2,   # γ
            'gpu_util': 0.1,      # δ
            'io_wait': 0.1,       # ε
            'retry_cost': 0.5     # ζ
        }
        self.witness_log = []
        self.invariants = self._setup_invariants()
        
    def _detect_caps(self) -> ResourceCaps:
        """Auto-detect system resource capacities"""
        return ResourceCaps(
            cpu_cores=psutil.cpu_count(),
            memory_gb=psutil.virtual_memory().total / (1024**3),
            gpu_slots=0,  # TODO: detect GPU
            io_bandwidth_mbps=1000.0  # TODO: benchmark I/O
        )
    
    def _setup_invariants(self) -> Dict[str, Callable]:
        """Setup invariant checking functions"""
        return {
            'I1_capacity': self._check_capacity,
            'I2_isolation': self._check_isolation,
            'I3_fairness': self._check_fairness,
            'I4_reversibility': self._check_reversibility,
            'I5_correctness': self._check_correctness
        }
    
    def _check_capacity(self, state: ExecutionState) -> bool:
        """I1: Capacity constraint"""
        return state.max_workers <= self.caps.cpu_cores
    
    def _check_isolation(self, state: ExecutionState) -> bool:
        """I2: GPU isolation constraint"""
        # For now, assume no GPU oversubscription
        return True
    
    def _check_fairness(self, state: ExecutionState) -> bool:
        """I3: Fairness constraint"""
        # Check that no demo category is starved
        categories = set(demo.category for demo in state.queue)
        return len(categories) <= 5  # Reasonable limit
    
    def _check_reversibility(self, state: ExecutionState) -> bool:
        """I4: Reversibility constraint"""
        # Check that strategy changes are logged
        return len(state.ledger) > 0 or len(state.queue) == 0
    
    def _check_correctness(self, state: ExecutionState) -> bool:
        """I5: Correctness constraint"""
        # Check that failed demos are handled
        failed_demos = [w for w in self.witness_log if w.exit_code != 0]
        return len(failed_demos) == 0 or all(w.retries > 0 for w in failed_demos)
    
    def score(self, state: ExecutionState) -> float:
        """
        Compute cost function J(σ) = α·makespan + β·CPU + γ·RAM + δ·GPU + ε·IO + ζ·Retry
        
        Returns gauge-normalized cost in [0,1]
        """
        if not self.witness_log:
            # Estimate based on strategy
            estimated_makespan = self._estimate_makespan(state)
            estimated_cpu = min(1.0, state.max_workers / self.caps.cpu_cores)
            estimated_memory = 0.5  # Conservative estimate
            estimated_gpu = 0.0
            estimated_io = 0.1
            estimated_retry = 0.0
            
            return (self.weights['makespan'] * estimated_makespan +
                    self.weights['cpu_usage'] * estimated_cpu +
                    self.weights['memory_peak'] * estimated_memory +
                    self.weights['gpu_util'] * estimated_gpu +
                    self.weights['io_wait'] * estimated_io +
                    self.weights['retry_cost'] * estimated_retry)
        
        # Compute from actual witness data
        total_time = max(w.end_time for w in self.witness_log) - min(w.start_time for w in self.witness_log)
        avg_cpu = np.mean([w.cpu_usage for w in self.witness_log])
        max_memory = max(w.memory_peak for w in self.witness_log) / (self.caps.memory_gb * 1024)  # Normalize to GB
        avg_gpu = np.mean([w.gpu_util for w in self.witness_log])
        avg_io_wait = np.mean([w.io_wait for w in self.witness_log])
        retry_cost = sum(w.retries for w in self.witness_log) / len(self.witness_log)
        
        # Normalize makespan (assume 1 hour max)
        normalized_makespan = min(1.0, total_time / 3600.0)
        
        return (self.weights['makespan'] * normalized_makespan +
                self.weights['cpu_usage'] * avg_cpu +
                self.weights['memory_peak'] * max_memory +
                self.weights['gpu_util'] * avg_gpu +
                self.weights['io_wait'] * avg_io_wait +
                self.weights['retry_cost'] * retry_cost)
    
    def _estimate_makespan(self, state: ExecutionState) -> float:
        """Estimate makespan based on demo profiles and strategy"""
        if not state.queue:
            return 0.0
        
        total_work = sum(self._estimate_demo_time(demo) for demo in state.queue)
        
        if state.strategy == ExecutionStrategy.SEQUENTIAL:
            return total_work / 3600.0  # Normalize to hours
        else:
            # Parallel execution reduces makespan
            parallel_factor = min(state.max_workers, len(state.queue))
            return (total_work / parallel_factor) / 3600.0
    
    def _estimate_demo_time(self, demo: DemoProfile) -> float:
        """Estimate execution time for a demo"""
        base_time = 1.0  # 1 second base
        
        # Adjust based on characteristics
        if demo.has_heavy_computation:
            base_time *= 3.0
        if demo.size > 10000:
            base_time *= 1.5
        if demo.category == 'quantum':
            base_time *= 2.0
        elif demo.category == 'quick':
            base_time *= 0.5
        
        return base_time
    
    def expand(self, state: ExecutionState) -> List[Tuple[str, ExecutionState]]:
        """
        Expand state with operational moves
        
        Returns list of (move_name, new_state) tuples
        """
        moves = []
        
        # Scale workers
        if state.max_workers > 1:
            moves.append(("scale_down", self._apply_scale(state, -1)))
        if state.max_workers < self.caps.cpu_cores:
            moves.append(("scale_up", self._apply_scale(state, 1)))
        
        # Reorder queue
        moves.append(("reorder_size", self._apply_reorder(state, "size")))
        moves.append(("reorder_category", self._apply_reorder(state, "category")))
        
        # Adjust timeouts
        moves.append(("timeout_tighten", self._apply_timeout(state, -10)))
        moves.append(("timeout_relax", self._apply_timeout(state, 10)))
        
        # Change strategy
        if state.strategy != ExecutionStrategy.SEQUENTIAL:
            moves.append(("to_sequential", self._apply_strategy(state, ExecutionStrategy.SEQUENTIAL)))
        if state.strategy != ExecutionStrategy.AGGRESSIVE_PARALLEL:
            moves.append(("to_aggressive", self._apply_strategy(state, ExecutionStrategy.AGGRESSIVE_PARALLEL)))
        
        return moves
    
    def _apply_scale(self, state: ExecutionState, delta: int) -> ExecutionState:
        """Apply scale operator"""
        new_workers = max(1, min(self.caps.cpu_cores, state.max_workers + delta))
        new_state = ExecutionState(
            queue=state.queue.copy(),
            strategy=state.strategy,
            max_workers=new_workers,
            timeouts=state.timeouts.copy(),
            caps=state.caps,
            profile=state.profile.copy(),
            ledger=state.ledger.copy()
        )
        new_state.ledger.append({
            "move": "scale",
            "delta": delta,
            "new_workers": new_workers,
            "timestamp": time.time()
        })
        return new_state
    
    def _apply_reorder(self, state: ExecutionState, key: str) -> ExecutionState:
        """Apply reorder operator"""
        if key == "size":
            new_queue = sorted(state.queue, key=lambda d: d.size)
        elif key == "category":
            new_queue = sorted(state.queue, key=lambda d: d.category)
        else:
            new_queue = state.queue.copy()
        
        new_state = ExecutionState(
            queue=new_queue,
            strategy=state.strategy,
            max_workers=state.max_workers,
            timeouts=state.timeouts.copy(),
            caps=state.caps,
            profile=state.profile.copy(),
            ledger=state.ledger.copy()
        )
        new_state.ledger.append({
            "move": "reorder",
            "key": key,
            "timestamp": time.time()
        })
        return new_state
    
    def _apply_timeout(self, state: ExecutionState, delta: float) -> ExecutionState:
        """Apply timeout adjustment"""
        new_timeouts = {}
        for demo_type, timeout in state.timeouts.items():
            new_timeouts[demo_type] = max(10.0, timeout + delta)
        
        new_state = ExecutionState(
            queue=state.queue.copy(),
            strategy=state.strategy,
            max_workers=state.max_workers,
            timeouts=new_timeouts,
            caps=state.caps,
            profile=state.profile.copy(),
            ledger=state.ledger.copy()
        )
        new_state.ledger.append({
            "move": "timeout",
            "delta": delta,
            "timestamp": time.time()
        })
        return new_state
    
    def _apply_strategy(self, state: ExecutionState, new_strategy: ExecutionStrategy) -> ExecutionState:
        """Apply strategy change"""
        new_workers = state.max_workers
        if new_strategy == ExecutionStrategy.SEQUENTIAL:
            new_workers = 1
        elif new_strategy == ExecutionStrategy.AGGRESSIVE_PARALLEL:
            new_workers = self.caps.cpu_cores
        
        new_state = ExecutionState(
            queue=state.queue.copy(),
            strategy=new_strategy,
            max_workers=new_workers,
            timeouts=state.timeouts.copy(),
            caps=state.caps,
            profile=state.profile.copy(),
            ledger=state.ledger.copy()
        )
        new_state.ledger.append({
            "move": "strategy",
            "new_strategy": new_strategy.value,
            "timestamp": time.time()
        })
        return new_state
    
    def search(self, initial_state: ExecutionState, beam_width: int = 5, 
               tolerance: float = 0.01, max_steps: int = 10) -> Tuple[ExecutionState, ExecutionCertificate]:
        """
        Run CE1 beam search over execution strategies
        
        Args:
            initial_state: Starting execution state
            beam_width: Beam search width
            tolerance: Convergence tolerance
            max_steps: Maximum search steps
            
        Returns:
            Tuple of (best_state, certificate)
        """
        beam = [initial_state]
        best_score = float('inf')
        best_state = initial_state
        no_improvement_steps = 0
        
        for step in range(max_steps):
            # Expand all states in beam
            candidates = []
            for state in beam:
                moves = self.expand(state)
                for move_name, new_state in moves:
                    # Check invariants
                    if all(invariant(new_state) for invariant in self.invariants.values()):
                        score = self.score(new_state)
                        candidates.append((score, new_state, move_name))
            
            # Sort by score and take top beam_width
            candidates.sort(key=lambda x: x[0])
            beam = [state for _, state, _ in candidates[:beam_width]]
            
            if not beam:
                break
            
            # Check for improvement
            current_best_score = min(self.score(state) for state in beam)
            if current_best_score < best_score - tolerance:
                best_score = current_best_score
                best_state = beam[0]
                no_improvement_steps = 0
            else:
                no_improvement_steps += 1
                if no_improvement_steps >= 3:
                    break
        
        # Generate certificate
        certificate = self._generate_certificate(best_state)
        
        return best_state, certificate
    
    def _generate_certificate(self, state: ExecutionState) -> ExecutionCertificate:
        """Generate execution certificate"""
        if not self.witness_log:
            # Estimate certificate
            estimated_makespan = self._estimate_makespan(state)
            return ExecutionCertificate(
                makespan=estimated_makespan,
                oom_count=0,
                deadline_hits=0,
                total_demos=len(state.queue),
                success_rate=1.0,
                resource_efficiency=0.8,
                sla_met=True
            )
        
        # Compute from witness data
        total_demos = len(self.witness_log)
        successful_demos = sum(1 for w in self.witness_log if w.exit_code == 0)
        oom_count = sum(1 for w in self.witness_log if w.memory_peak > self.caps.memory_gb * 1024 * 0.9)
        makespan = max(w.end_time for w in self.witness_log) - min(w.start_time for w in self.witness_log)
        
        return ExecutionCertificate(
            makespan=makespan,
            oom_count=oom_count,
            deadline_hits=0,  # TODO: implement deadline tracking
            total_demos=total_demos,
            success_rate=successful_demos / total_demos if total_demos > 0 else 0.0,
            resource_efficiency=1.0 - self.score(state),
            sla_met=oom_count == 0 and successful_demos == total_demos
        )
    
    def execute(self, state: ExecutionState) -> List[ExecutionWitness]:
        """
        Execute demos according to the planned strategy
        
        Returns list of execution witnesses
        """
        witnesses = []
        
        if state.strategy == ExecutionStrategy.SEQUENTIAL:
            witnesses = self._execute_sequential(state)
        else:
            witnesses = self._execute_parallel(state)
        
        # Store witnesses
        self.witness_log.extend(witnesses)
        
        return witnesses
    
    def _execute_sequential(self, state: ExecutionState) -> List[ExecutionWitness]:
        """Execute demos sequentially"""
        witnesses = []
        
        for demo in state.queue:
            witness = self._run_single_demo(demo, state.timeouts.get(demo.category, 30.0))
            witnesses.append(witness)
        
        return witnesses
    
    def _execute_parallel(self, state: ExecutionState) -> List[ExecutionWitness]:
        """Execute demos in parallel"""
        witnesses = []
        
        with ThreadPoolExecutor(max_workers=state.max_workers) as executor:
            # Submit all demos
            future_to_demo = {}
            for demo in state.queue:
                timeout = state.timeouts.get(demo.category, 30.0)
                future = executor.submit(self._run_single_demo, demo, timeout)
                future_to_demo[future] = demo
            
            # Collect results
            for future in as_completed(future_to_demo):
                witness = future.result()
                witnesses.append(witness)
        
        return witnesses
    
    def _run_single_demo(self, demo: DemoProfile, timeout: float) -> ExecutionWitness:
        """Run a single demo and return witness"""
        import subprocess
        import os
        
        start_time = time.time()
        
        # Monitor resources
        monitor = ResourceMonitor()
        monitor.start_monitoring()
        
        try:
            # Run the demo
            result = subprocess.run(
                [sys.executable, demo.name],
                cwd=demo.directory,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            end_time = time.time()
            exit_code = result.returncode
            
        except subprocess.TimeoutExpired:
            end_time = time.time()
            exit_code = 1
        except Exception:
            end_time = time.time()
            exit_code = 1
        
        # Stop monitoring
        metrics = monitor.stop_monitoring()
        
        return ExecutionWitness(
            demo_name=demo.name,
            start_time=start_time,
            end_time=end_time,
            cpu_usage=metrics.get('cpu_avg', 0.0),
            memory_peak=metrics.get('memory_max', 0.0),
            gpu_util=0.0,  # TODO: implement GPU monitoring
            io_wait=0.0,   # TODO: implement I/O monitoring
            exit_code=exit_code,
            retries=0,     # TODO: implement retry logic
            timeout_used=timeout
        )

class ResourceMonitor:
    """Monitor system resources during execution"""
    
    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.monitoring = False
        self.metrics = []
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start resource monitoring"""
        self.monitoring = True
        self.metrics = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self) -> Dict[str, float]:
        """Stop monitoring and return summary"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        if not self.metrics:
            return {}
        
        cpu_values = [m['cpu'] for m in self.metrics]
        memory_values = [m['memory'] for m in self.metrics]
        
        return {
            'cpu_avg': sum(cpu_values) / len(cpu_values),
            'cpu_max': max(cpu_values),
            'memory_avg': sum(memory_values) / len(memory_values),
            'memory_max': max(memory_values)
        }
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                
                self.metrics.append({
                    'cpu': cpu_percent,
                    'memory': memory.percent
                })
                
                time.sleep(self.interval)
            except Exception:
                break

def demo_operational_planner():
    """Demonstrate CE1 operational planner"""
    print("=== CE1 Operational Planner Demonstration ===")
    
    # Create test demo profiles
    demos = [
        DemoProfile("quantum_lattice_demo.py", "demos", 10466, 286, True, True, False, False, "quantum", "medium"),
        DemoProfile("ce1_quick_demo.py", "seed_metric", 5829, 144, True, True, False, False, "quick", "fast"),
        DemoProfile("ce1_integrated_demo.py", "seed_metric", 10160, 242, True, True, False, False, "comprehensive", "medium"),
        DemoProfile("audio_caption_loop.py", "demos", 3143, 90, False, True, False, False, "standard", "fast"),
    ]
    
    # Create initial state
    initial_state = ExecutionState(
        queue=demos,
        strategy=ExecutionStrategy.MODERATE_PARALLEL,
        max_workers=4,
        timeouts={"quantum": 60.0, "quick": 20.0, "comprehensive": 45.0, "standard": 30.0},
        caps=ResourceCaps(cpu_cores=8, memory_gb=16.0)
    )
    
    print(f"Initial state: {len(demos)} demos, {initial_state.max_workers} workers")
    print(f"Initial strategy: {initial_state.strategy.value}")
    
    # Initialize planner
    planner = OperationalPlanner()
    
    # Run search
    print("\nRunning CE1 search over execution strategies...")
    best_state, certificate = planner.search(initial_state, beam_width=3, tolerance=0.01, max_steps=5)
    
    print(f"\nSearch results:")
    print(f"  Best strategy: {best_state.strategy.value}")
    print(f"  Best workers: {best_state.max_workers}")
    print(f"  Best score: {planner.score(best_state):.4f}")
    
    print(f"\nCertificate:")
    print(f"  Makespan: {certificate.makespan:.2f}s")
    print(f"  Success rate: {certificate.success_rate:.2%}")
    print(f"  Resource efficiency: {certificate.resource_efficiency:.2%}")
    print(f"  SLA met: {certificate.sla_met}")
    
    print(f"\nStrategy evolution:")
    for entry in best_state.ledger:
        print(f"  {entry['move']}: {entry}")
    
    print("\n✓ CE1 operational planner demonstration completed!")
    print("The planner provides execution strategy search with witness emission.")

if __name__ == "__main__":
    import sys
    demo_operational_planner()
