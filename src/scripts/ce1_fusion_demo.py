#!/usr/bin/env python3
"""
CE1 Seed Fusion Demonstration

Shows the complete oracle-planner fusion in action:
- T (measure): Shadow ledger → Kravchuk → Mellin → Mirror → Posterior β_t
- S (act): Policy π* samples action a_t from energy-minimizing simplex  
- Φ (re-seed): Prior drifts via gyroglide on S³, preserving audit trail

This demonstrates the living lattice organism that actively steers builds.
"""

import os
import sys
import time
import json

# Add orchestrator to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'orchestrator'))

try:
    from jit import CE1Seed
    CE1_AVAILABLE = True
except ImportError:
    CE1_AVAILABLE = False
    print("Error: CE1 seed not available")
    sys.exit(1)

def demonstrate_ce1_fusion():
    """Demonstrate the complete CE1 seed fusion"""
    
    print("🌱 CE1 Seed Fusion Demonstration")
    print("=" * 50)
    
    # Initialize CE1 seed
    seed = CE1Seed()
    
    # Show initial state
    print("\n📊 Initial Lattice State:")
    initial_status = seed.status()
    print(f"Prior: θ={initial_status['prior']['theta']:.3f}, φ={initial_status['prior']['phi']:.3f}, ψ={initial_status['prior']['psi']:.3f}")
    print(f"Energy: {initial_status['prior']['energy']:.3f}, Confidence: {initial_status['prior']['confidence']:.3f}")
    
    # Test scripts to demonstrate oracle learning
    test_scripts = [
        "demos/audio_caption_loop.py",
        "seed_metric/ce1_quick_demo.py", 
        "quantum_lattice/chess_gray8.py",
        "witness/witness_system.py"
    ]
    
    print(f"\n🔄 Running {len(test_scripts)} test cycles...")
    
    for i, script in enumerate(test_scripts, 1):
        print(f"\n--- Cycle {i}: {script} ---")
        
        # Get oracle hint
        hint_result = seed.hint(script)
        print(f"🔮 Oracle: confidence={hint_result['confidence']:.3f}, energy={hint_result['energy']:.3f}")
        
        # Get planner action
        plan_result = seed.plan(script)
        action = plan_result['action']
        print(f"🎯 Planner: {action['type']} - {action['rationale']}")
        
        # Execute complete loop
        loop_result = seed.loop(script)
        new_prior = loop_result['tick']['reseed']['new_prior']
        
        print(f"🌱 Prior evolved: θ={new_prior['theta']:.3f}, φ={new_prior['phi']:.3f}")
        print(f"   Energy: {new_prior['energy']:.3f}, Confidence: {new_prior['confidence']:.3f}")
        
        # Show posterior distribution
        posterior = loop_result['tick']['measure']['posterior']
        print(f"📈 Posterior β_t: {[f'{p:.3f}' for p in posterior]}")
        
        time.sleep(0.5)  # Brief pause for demonstration
    
    # Show final state
    print(f"\n📊 Final Lattice State:")
    final_status = seed.status()
    print(f"Prior: θ={final_status['prior']['theta']:.3f}, φ={final_status['prior']['phi']:.3f}, ψ={final_status['prior']['psi']:.3f}")
    print(f"Energy: {final_status['prior']['energy']:.3f}, Confidence: {final_status['prior']['confidence']:.3f}")
    
    # Show evolution
    energy_change = final_status['prior']['energy'] - initial_status['prior']['energy']
    confidence_change = final_status['prior']['confidence'] - initial_status['prior']['confidence']
    
    print(f"\n🌊 Evolution Summary:")
    print(f"Energy change: {energy_change:+.3f}")
    print(f"Confidence change: {confidence_change:+.3f}")
    print(f"Audit trail length: {final_status['audit_trail_length']}")
    
    # Show the three-tick cycle in action
    print(f"\n🔄 Three-Tick Cycle Demonstration:")
    print("T (measure): Shadow ledger → Kravchuk → Mellin → Mirror → Posterior β_t")
    print("S (act): Policy π* samples action a_t from energy-minimizing simplex")
    print("Φ (re-seed): Prior drifts via gyroglide on S³, preserving audit trail")
    
    print(f"\n✅ CE1 Seed Fusion Complete!")
    print("The oracle and planner are now unified in a single lattice organism.")
    print("This system actively steers future builds with predictive intelligence.")

if __name__ == "__main__":
    demonstrate_ce1_fusion()
