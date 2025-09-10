"""
Unified Probe System: Local and Remote I/O as the Same Operation

Networking is just I/O at planetary scale - loading and computing extended through spacetime.
This system treats local execution and network packets as the same fundamental probe operation,
maintaining the illusion of shared memory through temporal mirroring and protocol planning.
"""

import json
import time
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import asyncio
import aiohttp
from urllib.parse import urlparse

class ProbeType(Enum):
    """Types of probes - all fundamentally I/O operations"""
    LOCAL_EXEC = "💻"      # Local script execution
    NETWORK_HTTP = "🌐"    # HTTP request/response
    NETWORK_TCP = "📡"     # Raw TCP packet
    FILE_IO = "📁"         # File read/write
    MEMORY_LOAD = "🧠"     # Memory allocation/access

class ProbeRealm(Enum):
    """Probe outcome realms - same as shadow ledger"""
    ILLUMINATED = "🌟"     # Success - data loaded/computed
    PENUMBRA = "🌓"        # Partial - some data, retries needed
    UMBRA = "🌑"           # Failure - energy debt, waiting for unfold

@dataclass
class ProbeMetadata:
    """Metadata that applies to all probe types"""
    probe_id: str
    timestamp: float
    probe_type: ProbeType
    realm: ProbeRealm
    duration_ms: float
    resource_metrics: Dict[str, Any] = field(default_factory=dict)
    content_hash: Optional[str] = None
    cid: Optional[str] = None  # Content-addressed ID

@dataclass 
class NetworkProbe(ProbeMetadata):
    """Network-specific probe data"""
    url: Optional[str] = None
    method: Optional[str] = None
    status_code: Optional[int] = None
    latency_ms: Optional[float] = None
    jitter_ms: Optional[float] = None
    packet_size: Optional[int] = None
    retry_count: int = 0
    protocol: Optional[str] = None  # HTTP/1.1, TCP, etc.

@dataclass
class LocalProbe(ProbeMetadata):
    """Local execution probe data"""
    script_name: str = ""
    exit_code: int = 0
    output: str = ""
    cpu_percent: float = 0.0
    memory_mb: float = 0.0

class UnifiedProbeSystem:
    """
    Unified system treating local and remote operations as the same I/O choreography.
    
    Key insight: A packet is just a byte-array that one machine loads into memory
    after another machine stored it. TCP/IP is glorified file I/O with weird buffering.
    """
    
    def __init__(self, vm=None):
        self.vm = vm
        self.probes: List[Union[NetworkProbe, LocalProbe]] = []
        self.protocol_planner = ProtocolPlanner(self)
        self.temporal_mirror = TemporalMirror(self)
        
    def _generate_probe_id(self, probe_type: ProbeType, target: str) -> str:
        """Generate content-addressed probe ID"""
        content = f"{probe_type.value}:{target}:{time.time()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _compute_content_hash(self, data: Union[str, bytes]) -> str:
        """Compute hash of probe content"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        return hashlib.sha256(data).hexdigest()
    
    async def probe_network(self, url: str, method: str = "GET", 
                          timeout: float = 5.0) -> NetworkProbe:
        """
        Network probe - just I/O with temporal constraints.
        
        The 'remote' just means 'slower to load' - same fundamental operation
        as local file I/O, just with different timing characteristics.
        """
        probe_id = self._generate_probe_id(ProbeType.NETWORK_HTTP, url)
        start_time = time.time()
        
        try:
            # Use protocol planner to determine retry strategy
            retry_strategy = self.protocol_planner.get_retry_strategy(url)
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                for attempt in range(retry_strategy.max_retries + 1):
                    try:
                        async with session.request(method, url) as response:
                            end_time = time.time()
                            duration_ms = (end_time - start_time) * 1000
                            
                            # Load response content (the "file" from remote memory)
                            content = await response.text()
                            content_hash = self._compute_content_hash(content)
                            
                            # Determine realm based on response
                            if response.status == 200:
                                realm = ProbeRealm.ILLUMINATED
                            elif 200 <= response.status < 400:
                                realm = ProbeRealm.PENUMBRA
                            else:
                                realm = ProbeRealm.UMBRA
                            
                            # Create network probe
                            probe = NetworkProbe(
                                probe_id=probe_id,
                                timestamp=start_time,
                                probe_type=ProbeType.NETWORK_HTTP,
                                realm=realm,
                                duration_ms=duration_ms,
                                url=url,
                                method=method,
                                status_code=response.status,
                                latency_ms=duration_ms,
                                jitter_ms=self._compute_jitter(url, duration_ms),
                                packet_size=len(content.encode('utf-8')),
                                retry_count=attempt,
                                protocol="HTTP/1.1",
                                content_hash=content_hash,
                                resource_metrics={
                                    'bandwidth_bps': len(content.encode('utf-8')) / (duration_ms / 1000),
                                    'success_rate': 1.0 if realm == ProbeRealm.ILLUMINATED else 0.0
                                }
                            )
                            
                            # Store in temporal mirror
                            await self.temporal_mirror.store_probe(probe, content)
                            
                            # Update protocol planner with observation
                            self.protocol_planner.record_observation(url, probe)
                            
                            return probe
                            
                    except asyncio.TimeoutError:
                        if attempt < retry_strategy.max_retries:
                            await asyncio.sleep(retry_strategy.backoff_ms[attempt] / 1000)
                            continue
                        else:
                            # Final failure
                            end_time = time.time()
                            duration_ms = (end_time - start_time) * 1000
                            
                            probe = NetworkProbe(
                                probe_id=probe_id,
                                timestamp=start_time,
                                probe_type=ProbeType.NETWORK_HTTP,
                                realm=ProbeRealm.UMBRA,
                                duration_ms=duration_ms,
                                url=url,
                                method=method,
                                retry_count=attempt,
                                protocol="HTTP/1.1",
                                resource_metrics={'success_rate': 0.0}
                            )
                            
                            self.protocol_planner.record_observation(url, probe)
                            return probe
                            
        except Exception as e:
            # Network error - still a valid probe, just failed
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            
            probe = NetworkProbe(
                probe_id=probe_id,
                timestamp=start_time,
                probe_type=ProbeType.NETWORK_HTTP,
                realm=ProbeRealm.UMBRA,
                duration_ms=duration_ms,
                url=url,
                method=method,
                retry_count=0,
                protocol="HTTP/1.1",
                resource_metrics={'error': str(e), 'success_rate': 0.0}
            )
            
            return probe
    
    def probe_local(self, script_name: str, command: str) -> LocalProbe:
        """
        Local execution probe - same I/O operation, just faster.
        
        Local execution is just loading and computing without the temporal
        constraints of network latency.
        """
        probe_id = self._generate_probe_id(ProbeType.LOCAL_EXEC, script_name)
        start_time = time.time()
        
        try:
            # Execute command (load and compute)
            import subprocess
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            
            # Determine realm
            if result.returncode == 0:
                realm = ProbeRealm.ILLUMINATED
            elif result.returncode < 0:
                realm = ProbeRealm.PENUMBRA
            else:
                realm = ProbeRealm.UMBRA
            
            # Create local probe
            probe = LocalProbe(
                probe_id=probe_id,
                timestamp=start_time,
                probe_type=ProbeType.LOCAL_EXEC,
                realm=realm,
                duration_ms=duration_ms,
                script_name=script_name,
                exit_code=result.returncode,
                output=result.stdout,
                content_hash=self._compute_content_hash(result.stdout),
                resource_metrics={
                    'cpu_percent': 0.0,  # Would need psutil for real metrics
                    'memory_mb': 0.0,
                    'success_rate': 1.0 if realm == ProbeRealm.ILLUMINATED else 0.0
                }
            )
            
            # Store in temporal mirror
            self.temporal_mirror.store_probe_sync(probe, result.stdout)
            
            return probe
            
        except Exception as e:
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            
            probe = LocalProbe(
                probe_id=probe_id,
                timestamp=start_time,
                probe_type=ProbeType.LOCAL_EXEC,
                realm=ProbeRealm.UMBRA,
                duration_ms=duration_ms,
                script_name=script_name,
                exit_code=-1,
                output=str(e),
                resource_metrics={'error': str(e), 'success_rate': 0.0}
            )
            
            return probe
    
    def _compute_jitter(self, url: str, current_latency: float) -> float:
        """Compute jitter based on historical latency measurements"""
        # Get historical latencies for this URL
        historical = [p.latency_ms for p in self.probes 
                     if isinstance(p, NetworkProbe) and p.url == url and p.latency_ms]
        
        if len(historical) < 2:
            return 0.0
        
        # Simple jitter calculation (variance from mean)
        mean_latency = sum(historical) / len(historical)
        variance = sum((lat - mean_latency) ** 2 for lat in historical) / len(historical)
        return variance ** 0.5
    
    def get_unified_metrics(self) -> Dict[str, Any]:
        """Get unified metrics across all probe types"""
        if not self.probes:
            return {}
        
        total_probes = len(self.probes)
        successful_probes = len([p for p in self.probes if p.realm == ProbeRealm.ILLUMINATED])
        network_probes = len([p for p in self.probes if isinstance(p, NetworkProbe)])
        local_probes = len([p for p in self.probes if isinstance(p, LocalProbe)])
        
        avg_duration = sum(p.duration_ms for p in self.probes) / total_probes
        
        return {
            'total_probes': total_probes,
            'success_rate': successful_probes / total_probes,
            'network_probes': network_probes,
            'local_probes': local_probes,
            'avg_duration_ms': avg_duration,
            'probe_distribution': {
                'network': network_probes / total_probes,
                'local': local_probes / total_probes
            }
        }

class ProtocolPlanner:
    """
    Protocol planner that maps network observations to actions.
    
    Uses shadow ledger of past flows to provide priors for scheduling
    the next transmission - the "mathematical smoke and mirrors" that
    keep the networking illusion coherent.
    """
    
    def __init__(self, probe_system):
        self.probe_system = probe_system
        self.observations: Dict[str, List[NetworkProbe]] = {}
    
    def get_retry_strategy(self, url: str) -> 'RetryStrategy':
        """Get retry strategy based on historical observations"""
        if url not in self.observations:
            return RetryStrategy(max_retries=3, backoff_ms=[100, 500, 1000])
        
        recent_probes = self.observations[url][-10:]  # Last 10 probes
        success_rate = len([p for p in recent_probes if p.realm == ProbeRealm.ILLUMINATED]) / len(recent_probes)
        
        if success_rate > 0.8:
            return RetryStrategy(max_retries=1, backoff_ms=[200])
        elif success_rate > 0.5:
            return RetryStrategy(max_retries=2, backoff_ms=[300, 800])
        else:
            return RetryStrategy(max_retries=5, backoff_ms=[100, 300, 600, 1200, 2400])
    
    def record_observation(self, url: str, probe: NetworkProbe):
        """Record network observation for future planning"""
        if url not in self.observations:
            self.observations[url] = []
        
        self.observations[url].append(probe)
        
        # Keep only recent observations (last 100)
        if len(self.observations[url]) > 100:
            self.observations[url] = self.observations[url][-100:]

@dataclass
class RetryStrategy:
    """Retry strategy for network probes"""
    max_retries: int
    backoff_ms: List[float]

class TemporalMirror:
    """
    Temporal mirror for creating the time-mirrored filesystem illusion.
    
    Maintains the illusion that "remote" just means "slower to load"
    by storing probe results with temporal metadata.
    """
    
    def __init__(self, probe_system):
        self.probe_system = probe_system
    
    async def store_probe(self, probe: Union[NetworkProbe, LocalProbe], content: str):
        """Store probe result in temporal mirror"""
        if self.probe_system.vm:
            try:
                # Store content with temporal metadata
                temporal_data = {
                    'probe_id': probe.probe_id,
                    'timestamp': probe.timestamp,
                    'probe_type': probe.probe_type.value,
                    'realm': probe.realm.value,
                    'duration_ms': probe.duration_ms,
                    'content': content,
                    'content_hash': probe.content_hash
                }
                
                data_json = json.dumps(temporal_data, ensure_ascii=False)
                cid = self.probe_system.vm.put(data_json.encode('utf-8'))
                probe.cid = cid
                
            except Exception as e:
                print(f"Warning: Could not store in temporal mirror: {e}")
        
        # Add to probe system
        self.probe_system.probes.append(probe)
    
    def store_probe_sync(self, probe: Union[NetworkProbe, LocalProbe], content: str):
        """Synchronous version for local probes"""
        if self.probe_system.vm:
            try:
                temporal_data = {
                    'probe_id': probe.probe_id,
                    'timestamp': probe.timestamp,
                    'probe_type': probe.probe_type.value,
                    'realm': probe.realm.value,
                    'duration_ms': probe.duration_ms,
                    'content': content,
                    'content_hash': probe.content_hash
                }
                
                data_json = json.dumps(temporal_data, ensure_ascii=False)
                cid = self.probe_system.vm.put(data_json.encode('utf-8'))
                probe.cid = cid
                
            except Exception as e:
                print(f"Warning: Could not store in temporal mirror: {e}")
        
        self.probe_system.probes.append(probe)

# Demo function
async def demo_unified_probes():
    """Demonstrate unified probe system"""
    print("=== Unified Probe System Demo ===")
    print("Networking as I/O at planetary scale\n")
    
    system = UnifiedProbeSystem()
    
    # Local probe (fast I/O)
    print("1. Local probe (fast I/O):")
    local_probe = system.probe_local("test_script", "echo 'Hello from local memory'")
    print(f"   {local_probe.probe_type.value} {local_probe.realm.value} {local_probe.duration_ms:.1f}ms")
    print(f"   Output: {local_probe.output.strip()}")
    print()
    
    # Network probe (slow I/O)
    print("2. Network probe (slow I/O):")
    network_probe = await system.probe_network("https://httpbin.org/json")
    print(f"   {network_probe.probe_type.value} {network_probe.realm.value} {network_probe.duration_ms:.1f}ms")
    print(f"   Status: {network_probe.status_code}, Latency: {network_probe.latency_ms:.1f}ms")
    print()
    
    # Unified metrics
    metrics = system.get_unified_metrics()
    print("3. Unified metrics:")
    print(f"   Total probes: {metrics['total_probes']}")
    print(f"   Success rate: {metrics['success_rate']:.1%}")
    print(f"   Network/Local: {metrics['network_probes']}/{metrics['local_probes']}")
    print(f"   Avg duration: {metrics['avg_duration_ms']:.1f}ms")
    print()
    
    print("Key insight: Both are just loading and computing -")
    print("the 'remote' just means 'slower to load'!")

if __name__ == "__main__":
    asyncio.run(demo_unified_probes())
