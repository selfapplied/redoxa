"""
Mirror registry for reversible transformations
"""

from typing import Dict, List, Optional, Any
import struct

class MirrorRegistry:
    """Registry for reversible mirror transformations"""
    
    def __init__(self):
        self.mirrors: Dict[str, Dict[str, Any]] = {
            "bitcast64": {
                "name": "bitcast64",
                "description": "Reversible bitcast between u64 and f64",
                "input_types": ["u64", "f64"],
                "output_types": ["f64", "u64"],
                "reversible": True,
                "cost": 0.0,  # Mirrors are free
            },
            "bitcast32": {
                "name": "bitcast32", 
                "description": "Reversible bitcast between u32 and f32",
                "input_types": ["u32", "f32"],
                "output_types": ["f32", "u32"],
                "reversible": True,
                "cost": 0.0,
            },
            "endian_swap": {
                "name": "endian_swap",
                "description": "Swap endianness of multi-byte values",
                "input_types": ["u16", "u32", "u64"],
                "output_types": ["u16", "u32", "u64"],
                "reversible": True,
                "cost": 0.0,
            },
            "sign_flip": {
                "name": "sign_flip",
                "description": "Flip sign bit of floating point numbers",
                "input_types": ["f32", "f64"],
                "output_types": ["f32", "f64"],
                "reversible": True,
                "cost": 0.0,
            },
        }
    
    def get_mirror(self, name: str) -> Optional[Dict[str, Any]]:
        """Get mirror metadata by name"""
        return self.mirrors.get(name)
    
    def list_mirrors(self) -> List[str]:
        """List all available mirrors"""
        return list(self.mirrors.keys())
    
    def is_reversible(self, name: str) -> bool:
        """Check if a mirror is reversible"""
        mirror = self.get_mirror(name)
        return mirror["reversible"] if mirror else False
    
    def get_cost(self, name: str) -> float:
        """Get the cost of a mirror operation"""
        mirror = self.get_mirror(name)
        return mirror["cost"] if mirror else 1.0
    
    def apply_mirror(self, name: str, data: bytes, direction: str = "forward") -> bytes:
        """Apply a mirror transformation"""
        if name == "bitcast64":
            return self._bitcast64(data, direction)
        elif name == "bitcast32":
            return self._bitcast32(data, direction)
        elif name == "endian_swap":
            return self._endian_swap(data)
        elif name == "sign_flip":
            return self._sign_flip(data)
        else:
            raise ValueError(f"Unknown mirror: {name}")
    
    def _bitcast64(self, data: bytes, direction: str) -> bytes:
        """Bitcast between u64 and f64"""
        if len(data) % 8 != 0:
            raise ValueError("Data must be 8-byte aligned for bitcast64")
        
        result = bytearray()
        for i in range(0, len(data), 8):
            chunk = data[i:i+8]
            if direction == "forward":
                # u64 -> f64
                u64_val = struct.unpack('<Q', chunk)[0]
                f64_val = struct.unpack('<d', struct.pack('<Q', u64_val))[0]
                result.extend(struct.pack('<d', f64_val))
            else:
                # f64 -> u64
                f64_val = struct.unpack('<d', chunk)[0]
                u64_val = struct.unpack('<Q', struct.pack('<d', f64_val))[0]
                result.extend(struct.pack('<Q', u64_val))
        
        return bytes(result)
    
    def _bitcast32(self, data: bytes, direction: str) -> bytes:
        """Bitcast between u32 and f32"""
        if len(data) % 4 != 0:
            raise ValueError("Data must be 4-byte aligned for bitcast32")
        
        result = bytearray()
        for i in range(0, len(data), 4):
            chunk = data[i:i+4]
            if direction == "forward":
                # u32 -> f32
                u32_val = struct.unpack('<I', chunk)[0]
                f32_val = struct.unpack('<f', struct.pack('<I', u32_val))[0]
                result.extend(struct.pack('<f', f32_val))
            else:
                # f32 -> u32
                f32_val = struct.unpack('<f', chunk)[0]
                u32_val = struct.unpack('<I', struct.pack('<f', f32_val))[0]
                result.extend(struct.pack('<I', u32_val))
        
        return bytes(result)
    
    def _endian_swap(self, data: bytes) -> bytes:
        """Swap endianness"""
        if len(data) % 2 != 0:
            raise ValueError("Data must be 2-byte aligned for endian swap")
        
        result = bytearray()
        for i in range(0, len(data), 2):
            chunk = data[i:i+2]
            result.extend(reversed(chunk))
        
        return bytes(result)
    
    def _sign_flip(self, data: bytes) -> bytes:
        """Flip sign bit of floating point numbers"""
        if len(data) % 4 != 0:
            raise ValueError("Data must be 4-byte aligned for sign flip")
        
        result = bytearray()
        for i in range(0, len(data), 4):
            chunk = data[i:i+4]
            # Flip the sign bit (MSB)
            chunk_array = bytearray(chunk)
            chunk_array[3] ^= 0x80  # Flip sign bit for f32
            result.extend(chunk_array)
        
        return bytes(result)
