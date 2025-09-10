"""
Kernel registry for managing WASM computational kernels
"""

from typing import Dict, List, Optional, Any
import json

class KernelRegistry:
    """Registry for WASM kernels with metadata"""
    
    def __init__(self):
        self.kernels: Dict[str, Dict[str, Any]] = {
            "mantissa_quant": {
                "name": "mantissa_quant",
                "description": "Quantize mantissa bits of complex numbers",
                "input_types": ["c64"],
                "output_types": ["c64"],
                "boundary_policy": None,
                "wasm_path": "wasm/mantissa_quant.wasm",
            },
            "hilbert_lift": {
                "name": "hilbert_lift", 
                "description": "Lift real numbers to complex domain",
                "input_types": ["f64"],
                "output_types": ["c64"],
                "boundary_policy": "causal",
                "wasm_path": "wasm/hilbert_lift.wasm",
            },
            "stft": {
                "name": "stft",
                "description": "Short-time Fourier transform",
                "input_types": ["f64"],
                "output_types": ["c64"],
                "boundary_policy": "windowed",
                "wasm_path": "wasm/stft.wasm",
            },
            "istft": {
                "name": "istft", 
                "description": "Inverse short-time Fourier transform",
                "input_types": ["c64"],
                "output_types": ["f64"],
                "boundary_policy": "windowed",
                "wasm_path": "wasm/istft.wasm",
            },
            "optical_flow_tiny": {
                "name": "optical_flow_tiny",
                "description": "Tiny optical flow computation",
                "input_types": ["f64", "f64"],
                "output_types": ["f64"],
                "boundary_policy": "spatial",
                "wasm_path": "wasm/optical_flow_tiny.wasm",
            },
            "textbert_tiny": {
                "name": "textbert_tiny",
                "description": "Tiny text embedding model",
                "input_types": ["u8"],
                "output_types": ["f64"],
                "boundary_policy": "tokenized",
                "wasm_path": "wasm/textbert_tiny.wasm",
            },
        }
    
    def get_kernel(self, name: str) -> Optional[Dict[str, Any]]:
        """Get kernel metadata by name"""
        return self.kernels.get(name)
    
    def list_kernels(self) -> List[str]:
        """List all available kernels"""
        return list(self.kernels.keys())
    
    def get_kernels_by_type(self, input_type: str) -> List[str]:
        """Get kernels that accept a specific input type"""
        return [
            name for name, metadata in self.kernels.items()
            if input_type in metadata["input_types"]
        ]
    
    def validate_boundary(self, kernel_name: str, boundary: Optional[str]) -> bool:
        """Validate boundary policy for a kernel"""
        kernel = self.get_kernel(kernel_name)
        if not kernel:
            return False
        
        expected_policy = kernel["boundary_policy"]
        if expected_policy is None:
            return boundary is None
        return boundary == expected_policy
    
    def get_kernel_signature(self, name: str) -> str:
        """Get kernel signature for WASM interface"""
        kernel = self.get_kernel(name)
        if not kernel:
            return ""
        
        inputs = " -> ".join(kernel["input_types"])
        outputs = " -> ".join(kernel["output_types"])
        return f"{inputs} -> {outputs}"
    
    def to_json(self) -> str:
        """Export kernel registry as JSON"""
        return json.dumps(self.kernels, indent=2)
