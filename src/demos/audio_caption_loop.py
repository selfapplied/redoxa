#!/usr/bin/env python3
"""
Audio → Caption → Loop Demo

Demonstrates the three-ring architecture:
- Ring 0: Rust core handles memory and planning
- Ring 1: WASM kernels perform computation
- Ring 2: Python orchestrates the experiment
"""

import numpy as np
import sys
import os

# Add the orchestrator to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'orchestrator'))

from redoxa import VM
from redoxa.paths import get_db_path

def create_audio_data(sample_rate: int = 44100, duration: float = 1.0) -> np.ndarray:
    """Create synthetic audio data for testing"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    # Create a simple sine wave with some harmonics
    audio = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t)
    return audio.astype(np.float64)

def main():
    print("Redoxa: Audio → Caption → Loop Demo")
    print("=" * 50)
    
    # Initialize VM
    vm = VM(db_path=get_db_path("audio_caption.db"))
    print("✓ VM initialized")
    
    # Create synthetic audio data
    audio_data = create_audio_data()
    print(f"✓ Created audio data: {len(audio_data)} samples")
    
    # Store audio in VM
    audio_cid = vm.put_array(audio_data)
    print(f"✓ Stored audio: {audio_cid[:16]}...")
    
    # Define the computation plan
    plan = [
        ("mirror.bitcast64", ["audio:u64"], ["audio:f64"], None),
        ("kernel.hilbert_lift", ["audio:f64"], ["audio:c64"], "causal"),
        ("kernel.mantissa_quant", ["audio:c64"], ["audio:c64"], None),
    ]
    
    print("\nExecuting plan:")
    for i, (step, inputs, outputs, boundary) in enumerate(plan):
        print(f"  {i+1}. {step}")
        if boundary:
            print(f"     Boundary: {boundary}")
    
    # Execute the plan
    frontier = vm.execute_plan(plan, [audio_cid])
    print(f"\n✓ Plan executed, frontier: {len(frontier)} results")
    
    # Run beam search iterations
    print("\nRunning beam search iterations:")
    for iteration in range(8):
        frontier = vm.tick(frontier, beam=6)
        print(f"  Iteration {iteration + 1}: {len(frontier)} candidates")
    
    # Select best result
    best_cid = vm.select_best(frontier)
    print(f"\n✓ Best result: {best_cid[:16]}...")
    
    # Retrieve and analyze result
    result_data = vm.view(best_cid, "raw")
    print(f"✓ Result size: {len(result_data)} bytes")
    
    # Show some statistics
    if len(result_data) >= 16:
        # Try to interpret as complex numbers
        complex_data = np.frombuffer(result_data, dtype=np.complex128)
        if len(complex_data) > 0:
            print(f"✓ Complex samples: {len(complex_data)}")
            print(f"  Real range: [{complex_data.real.min():.3f}, {complex_data.real.max():.3f}]")
            print(f"  Imag range: [{complex_data.imag.min():.3f}, {complex_data.imag.max():.3f}]")
    
    print("\nDemo completed successfully!")
    print("\nArchitecture summary:")
    print("  Ring 0 (Rust): Memory management, CID storage, planning")
    print("  Ring 1 (WASM): Sandboxed kernels (hilbert_lift, mantissa_quant)")
    print("  Ring 2 (Python): Orchestration, gene authoring, experiments")

if __name__ == "__main__":
    main()
