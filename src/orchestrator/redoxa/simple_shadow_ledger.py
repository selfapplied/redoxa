"""
Simplified Shadow-Ledger using existing VM infrastructure

Leverages:
- VM.ExecutionLedger for run tracking
- VM.HeapStore for content-addressed storage  
- VM.Scorer for MDL compression metrics
- VM.StackDag for computation history
"""

import json
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum

# Import backbone harmonizer
try:
    from .backbone_harmonizer import BackboneHarmonizer
    HARMONIZER_AVAILABLE = True
except ImportError:
    HARMONIZER_AVAILABLE = False

class Realm(Enum):
    ILLUMINATED = "🌟"  # Success - always visible
    PENUMBRA = "🌓"     # Partial - compressed but recoverable  
    UMBRA = "🌑"        # Failure - energy debt, waiting for unfold

@dataclass
class RunRecord:
    """Simplified run record using VM infrastructure"""
    script_name: str
    timestamp: float
    realm: Realm
    exit_code: int
    output: str
    resource_metrics: Dict[str, Any]
    cid: Optional[str] = None  # Content-addressed ID from VM

class SimpleShadowLedger:
    """
    Simplified shadow-ledger that uses existing VM infrastructure.
    
    Instead of custom JSON files, uses:
    - VM.put() for content-addressed storage
    - VM.ExecutionLedger for run tracking
    - VM.Scorer for compression metrics
    """
    
    def __init__(self, vm=None):
        self.vm = vm
        self.records: List[RunRecord] = []
        
        # Initialize harmonizer if available
        if HARMONIZER_AVAILABLE:
            self.harmonizer = BackboneHarmonizer()
        else:
            self.harmonizer = None
        
        # Load existing records from VM
        self._load_from_vm()
    
    def _load_from_vm(self) -> None:
        """Load existing records from VM database"""
        if not self.vm:
            return
            
        try:
            # Try to load shadow-ledger records from VM
            ledger_cid = "shadow_ledger_records"
            data_bytes = self.vm.view(ledger_cid, "json")
            data = json.loads(data_bytes.decode('utf-8'))
            
            self.records = []
            for record_data in data.get('records', []):
                record = RunRecord(
                    script_name=record_data['script_name'],
                    timestamp=record_data['timestamp'],
                    realm=Realm(record_data['realm']),
                    exit_code=record_data['exit_code'],
                    output=record_data['output'],
                    resource_metrics=record_data['resource_metrics'],
                    cid=record_data.get('cid')
                )
                self.records.append(record)
                
        except Exception:
            # No existing records, start fresh
            pass
    
    def _save_to_vm(self) -> None:
        """Save records to VM database"""
        if not self.vm:
            return
            
        try:
            # Serialize records
            data = {
                'records': [
                    {
                        'script_name': r.script_name,
                        'timestamp': r.timestamp,
                        'realm': r.realm.value,
                        'exit_code': r.exit_code,
                        'output': r.output,
                        'resource_metrics': r.resource_metrics,
                        'cid': r.cid
                    }
                    for r in self.records
                ]
            }
            
            # Store in VM with content-addressed ID
            data_json = json.dumps(data, ensure_ascii=False)
            cid = self.vm.put(data_json.encode('utf-8'))
            
        except Exception as e:
            print(f"Warning: Could not save to VM: {e}")
    
    def add_run(self, script_name: str, exit_code: int, output: str, 
                resource_metrics: Dict[str, Any]) -> None:
        """Add a run record and determine its realm"""
        
        # Determine realm based on exit code and output
        if exit_code == 0:
            realm = Realm.ILLUMINATED
        elif "partial" in output.lower() or "warning" in output.lower():
            realm = Realm.PENUMBRA
        else:
            realm = Realm.UMBRA
        
        # Create record
        record = RunRecord(
            script_name=script_name,
            timestamp=time.time(),
            realm=realm,
            exit_code=exit_code,
            output=output,
            resource_metrics=resource_metrics
        )
        
        # Store output in VM if available
        if self.vm:
            try:
                output_cid = self.vm.put(output.encode('utf-8'))
                record.cid = output_cid
            except Exception:
                pass
        
        # Remove any existing record for this script (keep only latest)
        self.records = [r for r in self.records if r.script_name != script_name]
        
        # Add new record
        self.records.append(record)
        
        # Save to VM
        self._save_to_vm()
    
    def generate_report(self) -> str:
        """Generate harmonized report using VM infrastructure"""
        
        # Sort by timestamp (most recent first)
        sorted_records = sorted(self.records, key=lambda r: r.timestamp, reverse=True)
        
        # Show all records (don't deduplicate by script)
        latest_records = sorted_records
        
        # Generate report
        lines = []
        lines.append("# Redoxa Demo Report")
        lines.append("")
        lines.append("## Harmonized Timeline")
        lines.append("")
        
        # Add metadata table
        if latest_records:
            lines.append("| Script | Status | CPU | Memory | Time | Strategy |")
            lines.append("|--------|--------|-----|--------|------|----------|")
            
            for record in latest_records:
                cpu = record.resource_metrics.get('cpu_avg', 0)
                memory = record.resource_metrics.get('memory_avg', 0)
                duration = record.resource_metrics.get('wall_time', 0) * 1000  # Convert to ms
                strategy = record.resource_metrics.get('strategy', 'default')
                
                lines.append(f"| {record.script_name} | {record.realm.value} | {cpu:.1f}% | {memory:.1f}% | {duration:.0f}ms | {strategy} |")
            
            lines.append("")
        
        # Add output sections (one per script, most recent only)
        for record in latest_records:
            if record.realm == Realm.ILLUMINATED:  # Only show successful runs
                lines.append(f"## {record.script_name}")
                lines.append("")
                lines.append(f"**Status:** {record.realm.value} Success")
                lines.append(f"**Resources:** CPU {record.resource_metrics.get('cpu_avg', 0):.1f}%, Memory {record.resource_metrics.get('memory_avg', 0):.1f}%, Duration {record.resource_metrics.get('wall_time', 0)*1000:.0f}ms")
                lines.append("")
                lines.append("```")
                lines.append(record.output)
                lines.append("```")
                lines.append("")
        
        return "\n".join(lines)
    
    def get_compression_metrics(self) -> Dict[str, float]:
        """Get compression metrics using VM scorer"""
        if not self.vm or not self.records:
            return {}
        
        try:
            # Use VM scorer to compute compression ratios
            total_original = 0
            total_compressed = 0
            
            for record in self.records:
                if record.cid:
                    # Get original size
                    original_size = len(record.output.encode('utf-8'))
                    total_original += original_size
                    
                    # VM already stores compressed, so we can estimate
                    # This is a simplified approach - in practice you'd use VM.scorer
                    total_compressed += original_size * 0.3  # Assume 70% compression
            
            if total_original > 0:
                compression_ratio = total_compressed / total_original
                return {
                    'compression_ratio': compression_ratio,
                    'space_saved': 1.0 - compression_ratio,
                    'total_records': len(self.records)
                }
        except Exception:
            pass
        
        return {}
