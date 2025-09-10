#!/usr/bin/env python3
"""
Test the connected build trainer and shadow ledger learning system
"""

import sys
sys.path.append('orchestrator')
from redoxa.shadow_ledger import ShadowLedger
from build_trainer import BuildTrainer

def test_learning_system():
    """Test the learning system with different scenarios"""
    print("🧪 Testing Build Trainer + Shadow Ledger Learning System")
    print("=" * 60)
    
    # Initialize trainer with shadow ledger
    trainer = BuildTrainer()
    
    if not trainer.shadow_ledger:
        print("❌ Shadow ledger not available")
        return
    
    print("✓ Shadow ledger connected")
    
    # Test 1: Build with warnings (should go to penumbra)
    print("\n--- Test 1: Build with warnings ---")
    cert1 = trainer.run_build(["--no-default-features", "--features", "standalone"])
    
    # Test 2: Try a different configuration
    print("\n--- Test 2: Different configuration ---")
    cert2 = trainer.run_build(["--no-default-features", "--features", "standalone", "--release"])
    
    # Generate learning report
    print("\n--- Learning Report ---")
    trainer._generate_learning_report()
    
    # Show timeline
    timeline = trainer.shadow_ledger.get_timeline()
    print(f"\n📊 Timeline Summary:")
    for entry in timeline:
        print(f"   {entry['realm']} {entry['script']} - {entry['status']} (energy: {entry['energy']:.3f})")
    
    # Show what we learned
    print(f"\n🎓 What We Learned:")
    print(f"   - Total builds: {len(timeline)}")
    print(f"   - All builds had warnings (penumbra)")
    print(f"   - Energy signatures: {[e['energy'] for e in timeline]}")
    print(f"   - Build patterns: {[e['script'].split('_')[1] for e in timeline]}")
    
    # Test shadow unfolding
    print(f"\n🔮 Testing Shadow Unfolding:")
    for entry in timeline:
        if entry['realm'] == '🌓':  # Penumbra
            shadow_hash = entry['script']
            unfolded = trainer.shadow_ledger.unfold_shadow(shadow_hash)
            if unfolded:
                print(f"   {entry['realm']} {shadow_hash}: {unfolded}")

if __name__ == "__main__":
    test_learning_system()
