"""
Python bindings for redoxa_core using cffi
"""

from cffi import FFI
import os

# Load the compiled library
lib_path = os.path.join(os.path.dirname(__file__), 'target', 'debug', 'libredoxa_core.dylib')
if not os.path.exists(lib_path):
    lib_path = os.path.join(os.path.dirname(__file__), 'target', 'release', 'libredoxa_core.dylib')

ffi = FFI()

# Define the C interface
ffi.cdef("""
    typedef struct VM VM;
    
    VM* vm_new(const char* db_path);
    void vm_free(VM* vm);
    
    char* vm_put(VM* vm, const char* data, size_t len);
    char* vm_view(VM* vm, const char* cid, const char* dtype);
    
    char* vm_apply(VM* vm, const char* step, const char* inputs, const char* boundary);
    double vm_score(VM* vm, const char* before, const char* after);
    
    char* vm_execute_plan(VM* vm, const char* plan_json, const char* inputs_json);
    char* vm_tick(VM* vm, const char* frontier_json, int beam);
    char* vm_select_best(VM* vm, const char* frontier_json);
""")

# Load the library
lib = ffi.dlopen(lib_path)

class VM:
    """Python wrapper for the Rust VM"""
    
    def __init__(self, db_path=None):
        if db_path is None:
            db_path = "vm.db"
        
        db_path_bytes = db_path.encode('utf-8')
        self.vm_ptr = lib.vm_new(db_path_bytes)
        if self.vm_ptr == ffi.NULL:
            raise RuntimeError("Failed to create VM")
    
    def __del__(self):
        if hasattr(self, 'vm_ptr') and self.vm_ptr != ffi.NULL:
            lib.vm_free(self.vm_ptr)
    
    def put(self, data: bytes) -> str:
        """Store bytes and return CID"""
        cid_ptr = lib.vm_put(self.vm_ptr, data, len(data))
        if cid_ptr == ffi.NULL:
            raise RuntimeError("Failed to store data")
        
        cid = ffi.string(cid_ptr).decode('utf-8')
        lib.free(cid_ptr)  # Free the C string
        return cid
    
    def view(self, cid: str, dtype: str = "raw") -> bytes:
        """View data by CID"""
        cid_bytes = cid.encode('utf-8')
        dtype_bytes = dtype.encode('utf-8')
        
        data_ptr = lib.vm_view(self.vm_ptr, cid_bytes, dtype_bytes)
        if data_ptr == ffi.NULL:
            return b""
        
        data = ffi.string(data_ptr)
        lib.free(data_ptr)  # Free the C string
        return data
    
    def apply(self, step: str, inputs: list, boundary: str = None) -> list:
        """Apply a computational step"""
        # For now, return a simple implementation
        outputs = []
        for input_cid in inputs:
            data = self.view(input_cid, "raw")
            if step == "mirror.bitcast64":
                output_data = data
            elif step == "kernel.hilbert_lift":
                # Simple complex lift - double the data size
                output_data = data + b"\x00" * len(data)
            elif step == "kernel.mantissa_quant":
                output_data = data
            else:
                output_data = data
            outputs.append(self.put(output_data))
        return outputs
    
    def score(self, before: list, after: list) -> float:
        """Score the difference between states"""
        return 0.0
    
    def execute_plan(self, plan: list, inputs: list) -> list:
        """Execute a plan and return frontier"""
        current = inputs
        for step, input_types, output_types, boundary in plan:
            current = self.apply(step, current, boundary)
        return current
    
    def tick(self, frontier: list, beam: int = 6) -> list:
        """Tick the frontier with beam search"""
        return frontier[:beam]
    
    def select_best(self, frontier: list) -> str:
        """Select best result from frontier"""
        return frontier[0] if frontier else ""
