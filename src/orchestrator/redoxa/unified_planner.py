"""
Unified Planner Interface

Bridges computational and operational planning under a single CE1 framework.
Provides unified API for both computational state search and execution strategy search.

Unifying interface (one trait/type across both worlds):

Planner API (shared):
  • plan.init(goal, scorer, invariants)
  • plan.expand(state) -> Iterable[move, state′, witness]
  • plan.score(state) -> float
  • plan.certificate(state) -> Cert
  • plan.search(beam=b, tol, max_steps) -> Best
"""

from typing import Dict, List, Tuple, Any, Optional, Union, Protocol
from dataclasses import dataclass
from abc import ABC, abstractmethod
import time

# Import existing planners
from .operational_planner import OperationalPlanner, ExecutionState, ExecutionCertificate
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from seed_metric.ce1_planner_shim import CE1PlannerShim, PlannerState, PlannerResult

@dataclass
class UnifiedGoal:
    """Unified goal specification"""
    goal_type: str  # "computational" or "operational"
    description: str
    constraints: Dict[str, Any] = None
    weights: Dict[str, float] = None

@dataclass
class UnifiedWitness:
    """Unified witness format"""
    operation: str
    timestamp: float
    state_id: str
    move_applied: str
    metrics: Dict[str, Any]
    invariants_preserved: bool

@dataclass
class UnifiedCertificate:
    """Unified certificate format"""
    goal_type: str
    objective_met: bool
    efficiency_score: float
    invariants_preserved: bool
    execution_time: float
    witness_count: int
    details: Dict[str, Any]

class PlannerProtocol(Protocol):
    """Protocol for unified planner interface"""
    
    def init(self, goal: UnifiedGoal) -> None:
        """Initialize planner with goal"""
        ...
    
    def expand(self, state: Any) -> List[Tuple[str, Any]]:
        """Expand state with available moves"""
        ...
    
    def score(self, state: Any) -> float:
        """Score state (lower is better)"""
        ...
    
    def certificate(self, state: Any) -> UnifiedCertificate:
        """Generate certificate for state"""
        ...
    
    def search(self, initial_state: Any, beam_width: int = 5, 
               tolerance: float = 0.01, max_steps: int = 10) -> Tuple[Any, UnifiedCertificate]:
        """Run beam search"""
        ...

class UnifiedPlanner:
    """
    Unified Planner
    
    Provides single interface for both computational and operational planning.
    Routes to appropriate planner based on goal type.
    """
    
    def __init__(self):
        self.computational_planner = CE1PlannerShim()
        self.operational_planner = OperationalPlanner()
        self.witness_log = []
    
    def init(self, goal: UnifiedGoal) -> None:
        """Initialize planner with unified goal"""
        if goal.goal_type == "computational":
            # Initialize computational planner
            pass  # CE1PlannerShim doesn't need explicit init
        elif goal.goal_type == "operational":
            # Initialize operational planner with weights
            if goal.weights:
                self.operational_planner.weights.update(goal.weights)
        else:
            raise ValueError(f"Unknown goal type: {goal.goal_type}")
    
    def expand(self, state: Union[PlannerState, ExecutionState]) -> List[Tuple[str, Any]]:
        """Expand state with available moves"""
        if isinstance(state, PlannerState):
            # Computational expansion
            return self._expand_computational(state)
        elif isinstance(state, ExecutionState):
            # Operational expansion
            return self._expand_operational(state)
        else:
            raise ValueError(f"Unknown state type: {type(state)}")
    
    def _expand_computational(self, state: PlannerState) -> List[Tuple[str, Any]]:
        """Expand computational state"""
        # Use CE1 planner shim to propose candidates
        candidates = self.computational_planner.propose_candidates(state, state, n_candidates=5)
        return [("propose", candidate) for candidate in candidates]
    
    def _expand_operational(self, state: ExecutionState) -> List[Tuple[str, Any]]:
        """Expand operational state"""
        return self.operational_planner.expand(state)
    
    def score(self, state: Union[PlannerState, ExecutionState]) -> float:
        """Score state (lower is better)"""
        if isinstance(state, PlannerState):
            # Computational scoring (distance to target)
            return state.distance_to_target
        elif isinstance(state, ExecutionState):
            # Operational scoring (cost function)
            return self.operational_planner.score(state)
        else:
            raise ValueError(f"Unknown state type: {type(state)}")
    
    def certificate(self, state: Union[PlannerState, ExecutionState]) -> UnifiedCertificate:
        """Generate unified certificate"""
        if isinstance(state, PlannerState):
            # Computational certificate
            return UnifiedCertificate(
                goal_type="computational",
                objective_met=state.distance_to_target < 1e-6,
                efficiency_score=1.0 - min(1.0, state.distance_to_target),
                invariants_preserved=state.certificate.invariants_preserved,
                execution_time=state.certificate.execution_time,
                witness_count=len(self.computational_planner.witness_log),
                details={
                    "distance": state.distance_to_target,
                    "pi_star": state.certificate.pi_star,
                    "tau_star": state.certificate.tau_star,
                    "sigma_star": state.certificate.sigma_star
                }
            )
        elif isinstance(state, ExecutionState):
            # Operational certificate
            cert = self.operational_planner._generate_certificate(state)
            return UnifiedCertificate(
                goal_type="operational",
                objective_met=cert.sla_met,
                efficiency_score=cert.resource_efficiency,
                invariants_preserved=True,  # TODO: check operational invariants
                execution_time=cert.makespan,
                witness_count=len(self.operational_planner.witness_log),
                details={
                    "makespan": cert.makespan,
                    "success_rate": cert.success_rate,
                    "oom_count": cert.oom_count,
                    "total_demos": cert.total_demos
                }
            )
        else:
            raise ValueError(f"Unknown state type: {type(state)}")
    
    def search(self, initial_state: Union[PlannerState, ExecutionState], 
               beam_width: int = 5, tolerance: float = 0.01, max_steps: int = 10) -> Tuple[Any, UnifiedCertificate]:
        """Run unified beam search"""
        if isinstance(initial_state, PlannerState):
            # Computational search
            return self._search_computational(initial_state, beam_width, tolerance, max_steps)
        elif isinstance(initial_state, ExecutionState):
            # Operational search
            return self._search_operational(initial_state, beam_width, tolerance, max_steps)
        else:
            raise ValueError(f"Unknown state type: {type(state)}")
    
    def _search_computational(self, initial_state: PlannerState, beam_width: int, 
                             tolerance: float, max_steps: int) -> Tuple[PlannerState, UnifiedCertificate]:
        """Run computational search"""
        # Use CE1 planner shim greedy search
        result = self.computational_planner.greedy_search(
            initial_state.seedstream, 
            initial_state.seedstream,  # TODO: proper target
            max_iterations=max_steps,
            tolerance=tolerance
        )
        
        best_state = result.path[-1] if result.path else initial_state
        certificate = self.certificate(best_state)
        
        return best_state, certificate
    
    def _search_operational(self, initial_state: ExecutionState, beam_width: int,
                           tolerance: float, max_steps: int) -> Tuple[ExecutionState, UnifiedCertificate]:
        """Run operational search"""
        return self.operational_planner.search(initial_state, beam_width, tolerance, max_steps)
    
    def plan_execution_strategy(self, demos: List[Any], caps: Optional[Any] = None) -> Tuple[ExecutionState, UnifiedCertificate]:
        """
        Plan execution strategy for demo set
        
        This is the main entry point for operational planning
        """
        from .operational_planner import DemoProfile, ResourceCaps
        
        # Convert demos to DemoProfile objects
        demo_profiles = []
        for demo in demos:
            if hasattr(demo, 'name') and hasattr(demo, 'directory'):
                profile = DemoProfile(
                    name=demo.name,
                    directory=demo.directory,
                    size=getattr(demo, 'size', 1000),
                    lines=getattr(demo, 'lines', 50),
                    has_heavy_computation=getattr(demo, 'has_heavy_computation', False),
                    has_numpy=getattr(demo, 'has_numpy', False),
                    has_matplotlib=getattr(demo, 'has_matplotlib', False),
                    has_gpu=getattr(demo, 'has_gpu', False),
                    category=getattr(demo, 'category', 'standard'),
                    expected_duration=getattr(demo, 'expected_duration', 'medium')
                )
                demo_profiles.append(profile)
        
        # Create initial execution state
        initial_state = ExecutionState(
            queue=demo_profiles,
            strategy=ExecutionState.ExecutionStrategy.MODERATE_PARALLEL,
            max_workers=4,
            timeouts={"quantum": 60.0, "quick": 20.0, "comprehensive": 45.0, "standard": 30.0},
            caps=caps or ResourceCaps(cpu_cores=8, memory_gb=16.0)
        )
        
        # Run search
        best_state, cert = self.search(initial_state, beam_width=5, tolerance=0.01, max_steps=8)
        
        return best_state, cert

def demo_unified_planner():
    """Demonstrate unified planner"""
    print("=== Unified Planner Demonstration ===")
    
    # Test operational planning
    print("\n1. Operational Planning:")
    planner = UnifiedPlanner()
    
    # Create test demos
    from .operational_planner import DemoProfile, ResourceCaps, ExecutionState, ExecutionStrategy
    
    demos = [
        DemoProfile("quantum_lattice_demo.py", "demos", 10466, 286, True, True, False, False, "quantum", "medium"),
        DemoProfile("ce1_quick_demo.py", "seed_metric", 5829, 144, True, True, False, False, "quick", "fast"),
    ]
    
    initial_state = ExecutionState(
        queue=demos,
        strategy=ExecutionStrategy.MODERATE_PARALLEL,
        max_workers=4,
        timeouts={"quantum": 60.0, "quick": 20.0},
        caps=ResourceCaps(cpu_cores=8, memory_gb=16.0)
    )
    
    best_state, certificate = planner.search(initial_state, beam_width=3, tolerance=0.01, max_steps=5)
    
    print(f"  Best strategy: {best_state.strategy.value}")
    print(f"  Best workers: {best_state.max_workers}")
    print(f"  Certificate: {certificate.objective_met}")
    
    # Test computational planning
    print("\n2. Computational Planning:")
    import numpy as np
    
    # Create test seedstreams
    n = 8
    initial_seedstream = np.random.randn(n) + 1j * np.random.randn(n)
    initial_seedstream = initial_seedstream / np.linalg.norm(initial_seedstream)
    
    target_seedstream = initial_seedstream * np.exp(1j * 0.1)  # Small phase change
    
    # Create planner state
    initial_canonical = planner.computational_planner.metric.canonicalize(initial_seedstream)
    initial_distance, initial_cert = planner.computational_planner.metric_compare(initial_seedstream, target_seedstream)
    
    initial_comp_state = PlannerState(
        seedstream=initial_seedstream,
        canonical_form=initial_canonical,
        distance_to_target=initial_distance,
        certificate=initial_cert,
        generation=0
    )
    
    best_comp_state, comp_certificate = planner.search(initial_comp_state, beam_width=3, tolerance=0.01, max_steps=5)
    
    print(f"  Best distance: {best_comp_state.distance_to_target:.6f}")
    print(f"  Certificate: {comp_certificate.objective_met}")
    
    print("\n✓ Unified planner demonstration completed!")
    print("The planner provides single interface for both computational and operational planning.")

if __name__ == "__main__":
    demo_unified_planner()
