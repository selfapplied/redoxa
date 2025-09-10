# Main Demos Directory - Demo Outputs

This document contains the output from the demos in the main `demos/` directory.

## **Working Demos:**

### 1. **audio_caption_loop.py**
```
Redoxa: Audio → Caption → Loop Demo
==================================================
✓ VM initialized
✓ Created audio data: 44100 samples
✓ Stored audio: d4b98d769d81ba7f...

Executing plan:
  1. mirror.bitcast64
  2. kernel.hilbert_lift
     Boundary: causal
  3. kernel.mantissa_quant

✓ Plan executed, frontier: 1 results

Running beam search iterations:
  Iteration 1: 1 candidates
  Iteration 2: 1 candidates
  Iteration 3: 1 candidates
  Iteration 4: 1 candidates
  Iteration 5: 1 candidates
  Iteration 6: 1 candidates
  Iteration 7: 1 candidates
  Iteration 8: 1 candidates

✓ Best result: 89b21111a51260de...
✓ Result size: 705600 bytes
✓ Complex samples: 44100
  Real range: [-1.299, 1.299]
  Imag range: [-1.299, 1.299]

Demo completed successfully!

Architecture summary:
  Ring 0 (Rust): Memory management, CID storage, planning
  Ring 1 (WASM): Sandboxed kernels (hilbert_lift, mantissa_quant)
  Ring 2 (Python): Orchestration, gene authoring, experiments
```

**Key Insights:**
- **Audio Processing**: Successfully processes 44,100 audio samples
- **Three-Ring Architecture**: Demonstrates the full Redoxa architecture
- **Kernel Pipeline**: mirror.bitcast64 → kernel.hilbert_lift → kernel.mantissa_quant
- **Beam Search**: 8 iterations with 1 candidate each
- **Output**: 705,600 bytes of complex data with real/imaginary ranges

## **Demos with Issues:**

### 2. **quantum_lattice_demo.py**
**Status**: Hangs during execution
**Issue**: Gets stuck during "Creating CE1 QuantumLattice..." phase
**Root Cause**: The `_generate_monster_generators()` method in `QuantumLattice` tries to create 196883×196883 matrices and perform SVD, which is computationally infeasible
**Location**: `quantum_lattice/ce1_passport.py` lines 128-131
**Fix Needed**: Reduce Monster subgroup dimension or use sparse/approximate methods

### 3. **seed_metric_demo.py**
**Status**: Hangs during execution
**Issue**: Appears to freeze immediately upon starting
**Possible Causes**:
- Missing numpy/scipy dependencies
- Import issues
- Infinite loop in metric computation

## **Summary:**

### **Working Demos (1/4):**
- ✅ **audio_caption_loop.py**: Full Redoxa architecture demonstration

### **Non-Working Demos (3/4):**
- ❌ **quantum_lattice_demo.py**: Hangs during initialization
- ❌ **quantum_lattice_light.py**: Hangs during initialization  
- ❌ **seed_metric_demo.py**: Hangs immediately

### **Recommendations:**
1. **Check Dependencies**: Ensure all required packages are installed
2. **Debug Hanging Demos**: Add timeout mechanisms or debug output
3. **Simplify Demos**: Reduce computational complexity for testing
4. **Add Error Handling**: Better error reporting for missing dependencies

The audio demo successfully demonstrates the core Redoxa architecture, while the other demos need debugging to resolve hanging issues.
