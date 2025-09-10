"""
Network Lattice Demo: Networking as Distributed Memory Operations

This demo shows how networking fits into the CE1 lattice view:
- A packet is a probe
- Latency and jitter are disturbances you measure  
- Protocols are planners that map observations to actions
- The shadow ledger of past flows gives priors for scheduling

Networking is not fundamentally about wires - it's about creating a 
time-mirrored filesystem where "remote" just means "slower to load."
"""

import asyncio
import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'orchestrator'))

from redoxa.unified_probe_system import UnifiedProbeSystem, ProbeType, ProbeRealm

class NetworkLatticeDemo:
    """Demo showing networking as lattice operations"""
    
    def __init__(self):
        self.system = UnifiedProbeSystem()
        self.lattice_observations = []
    
    async def demonstrate_packet_as_probe(self):
        """Show how a packet is just a probe in the lattice"""
        print("=== Packet as Probe ===")
        print("A packet is a probe that measures the 'distance' between memories")
        print()
        
        # Multiple probes to same endpoint to show lattice behavior
        urls = [
            "https://httpbin.org/json",
            "https://httpbin.org/uuid", 
            "https://httpbin.org/headers"
        ]
        
        probes = []
        for url in urls:
            print(f"Probing {url}...")
            probe = await self.system.probe_network(url)
            probes.append(probe)
            
            # Show probe as lattice measurement
            print(f"  Probe: {probe.probe_type.value} {probe.realm.value}")
            print(f"  Latency: {probe.latency_ms:.1f}ms (temporal distance)")
            print(f"  Packet size: {probe.packet_size} bytes (memory load)")
            print(f"  Success: {probe.realm == ProbeRealm.ILLUMINATED}")
            print()
        
        return probes
    
    async def demonstrate_latency_as_disturbance(self):
        """Show how latency and jitter are disturbances in the lattice"""
        print("=== Latency as Disturbance ===")
        print("Latency and jitter are disturbances you measure in the lattice")
        print()
        
        # Multiple probes to same endpoint to measure jitter
        url = "https://httpbin.org/delay/1"  # 1 second delay
        print(f"Measuring disturbances for {url}:")
        
        latencies = []
        for i in range(5):
            probe = await self.system.probe_network(url)
            latencies.append(probe.latency_ms)
            
            print(f"  Probe {i+1}: {probe.latency_ms:.1f}ms")
            
            # Show jitter calculation
            if len(latencies) > 1:
                mean_latency = sum(latencies) / len(latencies)
                variance = sum((lat - mean_latency) ** 2 for lat in latencies) / len(latencies)
                jitter = variance ** 0.5
                print(f"    Mean: {mean_latency:.1f}ms, Jitter: {jitter:.1f}ms")
            print()
        
        return latencies
    
    async def demonstrate_protocol_as_planner(self):
        """Show how protocols are planners mapping observations to actions"""
        print("=== Protocol as Planner ===")
        print("Protocols are planners that map observations (delays, errors) to actions (retries, reroutes)")
        print()
        
        # Test with a flaky endpoint to show retry planning
        flaky_url = "https://httpbin.org/status/500"  # Will fail
        
        print(f"Testing protocol planner with flaky endpoint: {flaky_url}")
        
        # First probe - will fail
        probe1 = await self.system.probe_network(flaky_url)
        print(f"  First probe: {probe1.realm.value} (retries: {probe1.retry_count})")
        
        # Second probe - planner should adapt
        probe2 = await self.system.probe_network(flaky_url)
        print(f"  Second probe: {probe2.realm.value} (retries: {probe2.retry_count})")
        
        # Show how planner learns
        strategy = self.system.protocol_planner.get_retry_strategy(flaky_url)
        print(f"  Planner strategy: max_retries={strategy.max_retries}, backoff={strategy.backoff_ms}")
        print()
        
        return [probe1, probe2]
    
    async def demonstrate_shadow_ledger_priors(self):
        """Show how shadow ledger provides priors for scheduling"""
        print("=== Shadow Ledger Priors ===")
        print("The shadow ledger of past flows gives priors for how to schedule the next transmission")
        print()
        
        # Build up history with different endpoints
        endpoints = [
            ("https://httpbin.org/json", "reliable"),
            ("https://httpbin.org/uuid", "reliable"), 
            ("https://httpbin.org/status/500", "flaky"),
            ("https://httpbin.org/delay/2", "slow")
        ]
        
        print("Building shadow ledger history:")
        for url, description in endpoints:
            probe = await self.system.probe_network(url)
            print(f"  {description}: {url} -> {probe.realm.value} ({probe.latency_ms:.1f}ms)")
        
        print()
        print("Shadow ledger priors:")
        
        # Show how priors affect planning
        for url, description in endpoints:
            strategy = self.system.protocol_planner.get_retry_strategy(url)
            observations = self.system.protocol_planner.observations.get(url, [])
            success_rate = len([p for p in observations if p.realm == ProbeRealm.ILLUMINATED]) / max(len(observations), 1)
            
            print(f"  {description}:")
            print(f"    Success rate: {success_rate:.1%}")
            print(f"    Strategy: {strategy.max_retries} retries, {strategy.backoff_ms}ms backoff")
            print()
    
    def demonstrate_temporal_mirroring(self):
        """Show the time-mirrored filesystem illusion"""
        print("=== Temporal Mirroring ===")
        print("Creating time-mirrored filesystem where 'remote' just means 'slower to load'")
        print()
        
        # Show unified metrics across all probe types
        metrics = self.system.get_unified_metrics()
        
        print("Unified I/O metrics:")
        print(f"  Total operations: {metrics['total_probes']}")
        print(f"  Success rate: {metrics['success_rate']:.1%}")
        print(f"  Network operations: {metrics['network_probes']}")
        print(f"  Local operations: {metrics['local_probes']}")
        print(f"  Average duration: {metrics['avg_duration_ms']:.1f}ms")
        print()
        
        # Show probe distribution
        print("Operation distribution:")
        for op_type, ratio in metrics['probe_distribution'].items():
            print(f"  {op_type}: {ratio:.1%}")
        print()
        
        # Show temporal characteristics
        network_probes = [p for p in self.system.probes if p.probe_type == ProbeType.NETWORK_HTTP]
        if network_probes:
            latencies = [p.latency_ms for p in network_probes if p.latency_ms]
            if latencies:
                print("Temporal characteristics:")
                print(f"  Min latency: {min(latencies):.1f}ms")
                print(f"  Max latency: {max(latencies):.1f}ms")
                print(f"  Avg latency: {sum(latencies)/len(latencies):.1f}ms")
                print()
    
    async def run_demo(self):
        """Run the complete network lattice demo"""
        print("Network Lattice Demo: Networking as Distributed Memory Operations")
        print("=" * 70)
        print()
        
        # 1. Packet as probe
        await self.demonstrate_packet_as_probe()
        
        # 2. Latency as disturbance  
        await self.demonstrate_latency_as_disturbance()
        
        # 3. Protocol as planner
        await self.demonstrate_protocol_as_planner()
        
        # 4. Shadow ledger priors
        await self.demonstrate_shadow_ledger_priors()
        
        # 5. Temporal mirroring
        self.demonstrate_temporal_mirroring()
        
        print("Key Insight:")
        print("Networking is not fundamentally about wires.")
        print("It's about creating a time-mirrored filesystem where")
        print("'remote' just means 'slower to load.'")
        print()
        print("The rest is mathematical smoke and mirrors:")
        print("- Compression tricks")
        print("- Checksum tricks") 
        print("- Ordering tricks")
        print()
        print("That keep the illusion coherent.")

async def main():
    """Main demo function"""
    demo = NetworkLatticeDemo()
    await demo.run_demo()

if __name__ == "__main__":
    asyncio.run(main())
