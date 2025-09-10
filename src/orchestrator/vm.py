"""
Main VM interface for Python orchestrator
"""

import numpy as np
from typing import List, Optional, Tuple, Any
from .kernels import KernelRegistry
from .mirrors import MirrorRegistry
# Removed unified_planner import - now using Rust core directly

# Import the Rust core (will be available after building)
try:
    import redoxa_core
    CoreVM = redoxa_core.VM
except ImportError:
    # Fallback for development
    class CoreVM:
        def __init__(self, db_path: Optional[str] = None):
            self.db_path = db_path or "vm.db"
            self.data = {}
        
        def put(self, data: bytes) -> str:
            import hashlib
            cid = hashlib.sha256(data).hexdigest()
            self.data[cid] = data
            return cid
        
        def view(self, cid: str, data_type: str) -> bytes:
            return self.data.get(cid, b"")
        
        def apply(self, step: str, inputs: List[str], boundary: Optional[str]) -> List[str]:
            # Simple fallback implementation
            outputs = []
            for input_cid in inputs:
                data = self.view(input_cid, "raw")
                if step == "mirror.bitcast64":
                    # Simple bitcast
                    output_data = data
                elif step == "kernel.hilbert_lift":
                    # Simple complex lift
                    output_data = data + b"\x00" * len(data)  # Double size for complex
                else:
                    output_data = data
                outputs.append(self.put(output_data))
            return outputs
        
        def score(self, before: List[str], after: List[str]) -> float:
            return 0.0
        
        def execute_plan(self, plan: List[Tuple], inputs: List[str]) -> List[str]:
            current = inputs
            for step, input_types, output_types, boundary in plan:
                current = self.apply(step, current, boundary)
            return current
        
        def tick(self, frontier: List[str], beam: int) -> List[str]:
            return frontier[:beam]
        
        def select_best(self, frontier: List[str]) -> str:
            return frontier[0] if frontier else ""


class VM:
    """Main VM interface combining Rust core with Python orchestration"""
    
    def __init__(self, db_path: Optional[str] = None):
        self.core = CoreVM(db_path)
        self.kernels = KernelRegistry()
        self.mirrors = MirrorRegistry()
        # Using Rust core directly - no unified_planner needed
    
    def put(self, data: bytes) -> str:
        """Store bytes and return CID"""
        return self.core.put(data)
    
    def put_array(self, array: np.ndarray) -> str:
        """Store numpy array and return CID"""
        return self.put(array.tobytes())
    
    def view(self, cid: str, dtype: str = "raw") -> bytes:
        """View data by CID with type casting"""
        return self.core.view(cid, dtype)
    
    def view_array(self, cid: str, dtype: np.dtype, shape: Tuple[int, ...]) -> np.ndarray:
        """View data as numpy array"""
        data = self.view(cid, "raw")
        return np.frombuffer(data, dtype=dtype).reshape(shape)
    
    def apply(self, step: str, inputs: List[str], boundary: Optional[str] = None) -> List[str]:
        """Apply a computational step"""
        return self.core.apply(step, inputs, boundary)
    
    def score(self, before: List[str], after: List[str]) -> float:
        """Score the difference between states using MDL"""
        return self.core.score(before, after)
    
    def execute_plan(self, plan: List[Tuple], inputs: List[str]) -> List[str]:
        """Execute a plan and return frontier"""
        return self.core.execute_plan(plan, inputs)
    
    def tick(self, frontier: List[str], beam: int = 6) -> List[str]:
        """Tick the frontier with beam search"""
        return self.core.tick(frontier, beam)
    
    def select_best(self, frontier: List[str]) -> str:
        """Select best result from frontier"""
        return self.core.select_best(frontier)
    
    def create_plan(self, goal: str) -> List[Tuple]:
        """Create a plan to reach a goal"""
        if goal == "audio_to_complex":
            return [
                ("mirror.bitcast64", ["audio:u64"], ["audio:f64"], None),
                ("kernel.hilbert_lift", ["audio:f64"], ["audio:c64"], "causal"),
            ]
        elif goal == "quantize_complex":
            return [
                ("kernel.mantissa_quant", ["audio:c64"], ["audio:c64"], None),
            ]
        else:
            return [("mirror.bitcast64", ["input:raw"], ["output:raw"], None)]
    
    def run_experiment(self, inputs: List[str], goal: str, iterations: int = 8) -> str:
        """Run a complete experiment"""
        plan = self.create_plan(goal)
        frontier = self.execute_plan(plan, inputs)
        
        for _ in range(iterations):
            frontier = self.tick(frontier, beam=6)
        
        return self.select_best(frontier)
    
    def plan_execution_strategy(self, demos: List[Any], goal_type: str = "operational") -> Tuple[Any, Any]:
        """
        Plan execution strategy using unified planner
        
        Args:
            demos: List of demo objects to plan for
            goal_type: "operational" for execution planning, "computational" for state planning
            
        Returns:
            Tuple of (best_state, certificate)
        """
        goal = UnifiedGoal(
            goal_type=goal_type,
            description="Execute demos efficiently",
            constraints={"max_workers": 8, "timeout": 300},
            weights={"makespan": 1.0, "cpu_usage": 0.3, "memory_peak": 0.2}
        )
        
        self.unified_planner.init(goal)
        
        if goal_type == "operational":
            return self.unified_planner.plan_execution_strategy(demos)
        else:
            # For computational planning, would need to create appropriate initial state
            raise NotImplementedError("Computational planning via VM not yet implemented")
    
    def execute_with_planning(self, demos: List[Any]) -> List[Any]:
        """
        Execute demos using intelligent execution planning
        
        This is the main entry point that replaces the simple demo runner
        """
        # Plan execution strategy
        best_state, certificate = self.plan_execution_strategy(demos, "operational")
        
        print(f"Planned execution strategy: {best_state.strategy.value}")
        print(f"Workers: {best_state.max_workers}")
        print(f"Certificate: SLA met = {certificate.objective_met}")
        
        # Execute according to plan
        witnesses = self.unified_planner.operational_planner.execute(best_state)
        
        return witnesses
