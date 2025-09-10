"""
CE1 Network Integration: Folding Network Packets into the Shadow Ledger

This integrates the unified probe system with the existing CE1 shadow ledger,
making local and remote probes indistinguishable in the lattice view.

Key insight: Networking is just I/O at planetary scale - loading and computing
extended through spacetime. The shadow ledger becomes a unified memory of all
probe operations, whether they're local execution or network packets.
"""

import json
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

# Import existing shadow ledger
from .simple_shadow_ledger import SimpleShadowLedger, RunRecord, Realm
from .unified_probe_system import UnifiedProbeSystem, NetworkProbe, LocalProbe, ProbeType, ProbeRealm

class CE1NetworkIntegration:
    """
    Integrates network probes into the CE1 shadow ledger system.
    
    Makes local and remote probes indistinguishable by treating both as
    the same fundamental I/O operation with different temporal characteristics.
    """
    
    def __init__(self, vm=None):
        self.shadow_ledger = SimpleShadowLedger(vm)
        self.probe_system = UnifiedProbeSystem(vm)
        self.integrated_records: List[Union[RunRecord, NetworkProbe, LocalProbe]] = []
    
    def _convert_probe_to_run_record(self, probe: Union[NetworkProbe, LocalProbe]) -> RunRecord:
        """Convert a probe to a RunRecord for shadow ledger compatibility"""
        
        # Map probe realm to shadow ledger realm
        if probe.realm == ProbeRealm.ILLUMINATED:
            shadow_realm = Realm.ILLUMINATED
        elif probe.realm == ProbeRealm.PENUMBRA:
            shadow_realm = Realm.PENUMBRA
        else:
            shadow_realm = Realm.UMBRA
        
        # Create script name based on probe type
        if isinstance(probe, NetworkProbe):
            script_name = f"network:{probe.url}"
            output = f"HTTP {probe.method} {probe.url} -> {probe.status_code} ({probe.latency_ms:.1f}ms)"
        else:
            script_name = f"local:{probe.script_name}"
            output = probe.output
        
        # Create resource metrics
        resource_metrics = {
            'duration_ms': probe.duration_ms,
            'probe_type': probe.probe_type.value,
            'success_rate': 1.0 if probe.realm == ProbeRealm.ILLUMINATED else 0.0
        }
        
        # Add network-specific metrics
        if isinstance(probe, NetworkProbe):
            resource_metrics.update({
                'latency_ms': probe.latency_ms,
                'jitter_ms': probe.jitter_ms,
                'packet_size': probe.packet_size,
                'retry_count': probe.retry_count,
                'protocol': probe.protocol
            })
        else:
            resource_metrics.update({
                'cpu_percent': probe.cpu_percent,
                'memory_mb': probe.memory_mb,
                'exit_code': probe.exit_code
            })
        
        return RunRecord(
            script_name=script_name,
            timestamp=probe.timestamp,
            realm=shadow_realm,
            exit_code=0 if probe.realm == ProbeRealm.ILLUMINATED else 1,
            output=output,
            resource_metrics=resource_metrics,
            cid=probe.cid
        )
    
    def add_network_probe(self, probe: NetworkProbe):
        """Add network probe to integrated shadow ledger"""
        # Convert to RunRecord for shadow ledger compatibility
        run_record = self._convert_probe_to_run_record(probe)
        
        # Add to shadow ledger
        self.shadow_ledger.add_run(
            run_record.script_name,
            run_record.exit_code,
            run_record.output,
            run_record.resource_metrics
        )
        
        # Store original probe for detailed analysis
        self.integrated_records.append(probe)
    
    def add_local_probe(self, probe: LocalProbe):
        """Add local probe to integrated shadow ledger"""
        # Convert to RunRecord for shadow ledger compatibility
        run_record = self._convert_probe_to_run_record(probe)
        
        # Add to shadow ledger
        self.shadow_ledger.add_run(
            run_record.script_name,
            run_record.exit_code,
            run_record.output,
            run_record.resource_metrics
        )
        
        # Store original probe for detailed analysis
        self.integrated_records.append(probe)
    
    def get_unified_metrics(self) -> Dict[str, Any]:
        """Get unified metrics across all probe types and shadow ledger records"""
        
        # Get shadow ledger metrics
        shadow_metrics = self.shadow_ledger.get_compression_metrics()
        
        # Get probe system metrics
        probe_metrics = self.probe_system.get_unified_metrics()
        
        # Analyze integrated records
        network_probes = [r for r in self.integrated_records if isinstance(r, NetworkProbe)]
        local_probes = [r for r in self.integrated_records if isinstance(r, LocalProbe)]
        
        # Compute unified statistics
        total_operations = len(self.integrated_records)
        successful_operations = len([r for r in self.integrated_records if r.realm == ProbeRealm.ILLUMINATED])
        
        # Network-specific metrics
        network_latencies = [p.latency_ms for p in network_probes if p.latency_ms]
        network_success_rate = len([p for p in network_probes if p.realm == ProbeRealm.ILLUMINATED]) / max(len(network_probes), 1)
        
        # Local-specific metrics
        local_success_rate = len([p for p in local_probes if p.realm == ProbeRealm.ILLUMINATED]) / max(len(local_probes), 1)
        
        return {
            'total_operations': total_operations,
            'success_rate': successful_operations / max(total_operations, 1),
            'network_operations': len(network_probes),
            'local_operations': len(local_probes),
            'network_success_rate': network_success_rate,
            'local_success_rate': local_success_rate,
            'avg_network_latency': sum(network_latencies) / max(len(network_latencies), 1) if network_latencies else 0,
            'min_network_latency': min(network_latencies) if network_latencies else 0,
            'max_network_latency': max(network_latencies) if network_latencies else 0,
            'shadow_ledger_compression': shadow_metrics.get('compression_ratio', 0),
            'shadow_ledger_records': shadow_metrics.get('total_records', 0)
        }
    
    def generate_unified_report(self) -> str:
        """Generate unified report showing both shadow ledger and probe system data"""
        
        # Get shadow ledger report
        shadow_report = self.shadow_ledger.generate_report()
        
        # Get unified metrics
        metrics = self.get_unified_metrics()
        
        # Create unified report
        lines = []
        lines.append("# CE1 Unified Shadow Ledger Report")
        lines.append("")
        lines.append("## Networking as I/O at Planetary Scale")
        lines.append("")
        lines.append("This report shows how network packets and local execution")
        lines.append("are unified as the same fundamental I/O operation.")
        lines.append("")
        
        # Unified metrics table
        lines.append("### Unified Metrics")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Total Operations | {metrics['total_operations']} |")
        lines.append(f"| Overall Success Rate | {metrics['success_rate']:.1%} |")
        lines.append(f"| Network Operations | {metrics['network_operations']} |")
        lines.append(f"| Local Operations | {metrics['local_operations']} |")
        lines.append(f"| Network Success Rate | {metrics['network_success_rate']:.1%} |")
        lines.append(f"| Local Success Rate | {metrics['local_success_rate']:.1%} |")
        lines.append(f"| Avg Network Latency | {metrics['avg_network_latency']:.1f}ms |")
        lines.append(f"| Network Latency Range | {metrics['min_network_latency']:.1f}ms - {metrics['max_network_latency']:.1f}ms |")
        lines.append(f"| Shadow Ledger Compression | {metrics['shadow_ledger_compression']:.1%} |")
        lines.append("")
        
        # Key insight
        lines.append("### Key Insight")
        lines.append("")
        lines.append("Networking is not fundamentally about wires. It's about creating")
        lines.append("a time-mirrored filesystem where 'remote' just means 'slower to load.'")
        lines.append("")
        lines.append("Both network packets and local execution are:")
        lines.append("- **Loading**: Byte-arrays loaded into memory")
        lines.append("- **Computing**: Protocol logic, retry strategies, error handling")
        lines.append("- **Probing**: Measuring temporal and spatial distances")
        lines.append("")
        
        # Add shadow ledger report
        lines.append("## Shadow Ledger Details")
        lines.append("")
        lines.append(shadow_report)
        
        return "\n".join(lines)
    
    def demonstrate_unified_view(self):
        """Demonstrate the unified view of local and remote operations"""
        print("=== CE1 Unified Shadow Ledger ===")
        print("Local and remote probes become indistinguishable")
        print()
        
        # Show all integrated records
        print("Integrated Operations:")
        for record in self.integrated_records:
            if isinstance(record, NetworkProbe):
                print(f"  🌐 {record.url} -> {record.realm.value} ({record.latency_ms:.1f}ms)")
            else:
                print(f"  💻 {record.script_name} -> {record.realm.value} ({record.duration_ms:.1f}ms)")
        
        print()
        
        # Show unified metrics
        metrics = self.get_unified_metrics()
        print("Unified Metrics:")
        print(f"  Total operations: {metrics['total_operations']}")
        print(f"  Success rate: {metrics['success_rate']:.1%}")
        print(f"  Network/Local: {metrics['network_operations']}/{metrics['local_operations']}")
        print(f"  Avg network latency: {metrics['avg_network_latency']:.1f}ms")
        print()
        
        print("Key insight: Both are just loading and computing -")
        print("the 'remote' just means 'slower to load'!")

# Demo function
async def demo_ce1_integration():
    """Demonstrate CE1 network integration"""
    print("CE1 Network Integration Demo")
    print("=" * 40)
    print()
    
    integration = CE1NetworkIntegration()
    
    # Add some network probes
    print("Adding network probes...")
    network_probe1 = await integration.probe_system.probe_network("https://httpbin.org/json")
    integration.add_network_probe(network_probe1)
    
    network_probe2 = await integration.probe_system.probe_network("https://httpbin.org/uuid")
    integration.add_network_probe(network_probe2)
    
    # Add some local probes
    print("Adding local probes...")
    local_probe1 = integration.probe_system.probe_local("test_script", "echo 'Hello from local memory'")
    integration.add_local_probe(local_probe1)
    
    # Demonstrate unified view
    integration.demonstrate_unified_view()
    
    # Generate unified report
    print("\nGenerating unified report...")
    report = integration.generate_unified_report()
    print(report)

if __name__ == "__main__":
    import asyncio
    asyncio.run(demo_ce1_integration())
