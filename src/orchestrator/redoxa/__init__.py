"""
Redoxa: Three-ring VM for reversible computation

Ring 0: Rust core (memory, planning, WASM hosting)
Ring 1: WASM kernels (sandboxed computation)  
Ring 2: Python orchestrator (control plane, gene authoring)
"""

from .vm import VM
from .kernels import KernelRegistry
from .mirrors import MirrorRegistry

__version__ = "0.1.0"
__all__ = ["VM", "KernelRegistry", "MirrorRegistry"]
