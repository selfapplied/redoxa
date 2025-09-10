#!/usr/bin/env python3
"""
Build Trainer - Converts every compile into training data

This script runs cargo builds and converts the JSON output into certificates
that the ArtifactModelManager can learn from.
"""

import json
import subprocess
import sys
import time
import hashlib
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

# Import shadow ledger for learning
try:
    sys.path.append('orchestrator')
    from redoxa.shadow_ledger import ShadowLedger
    SHADOW_LEDGER_AVAILABLE = True
except ImportError:
    SHADOW_LEDGER_AVAILABLE = False

@dataclass
class BuildCertificate:
    """Certificate for a build operation"""
    kind: str = "compile.certificate"
    crate: str = ""
    target: str = ""
    features: List[str] = None
    duration_ms: int = 0
    success: bool = False
    errors: List[Dict[str, str]] = None
    warnings: List[Dict[str, Any]] = None
    argv_cid: str = ""
    witness: Dict[str, str] = None
    
    def __post_init__(self):
        if self.features is None:
            self.features = []
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.witness is None:
            self.witness = {}

class BuildTrainer:
    """Collects build training data and generates certificates"""
    
    def __init__(self, output_dir: str = ".out/reports/builds"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize shadow ledger for learning
        if SHADOW_LEDGER_AVAILABLE:
            ledger_path = self.output_dir / "shadow_ledger.json"
            self.shadow_ledger = ShadowLedger(ledger_path=str(ledger_path))
            print(f"✓ Shadow ledger connected for learning: {ledger_path}")
        else:
            self.shadow_ledger = None
            print("⚠️  Shadow ledger not available - learning disabled")
        
    def run_build(self, args: List[str]) -> BuildCertificate:
        """Run a cargo build and collect training data"""
        start_time = time.time()
        
        # Add JSON output format
        cmd = ["cargo", "build"] + args + ["--message-format=json"]
        
        print(f"Running: {' '.join(cmd)}")
        
        # Generate CID for this build command
        argv_cid = self._generate_cid(cmd)
        
        # Tap into build artifacts before they get cleaned
        self._tap_build_artifacts(argv_cid)
        
        # Run the build
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd="core"
            )
            
            duration_ms = int((time.time() - start_time) * 1000)
            success = result.returncode == 0
            
            # Parse JSON output
            artifacts = []
            warnings = []
            errors = []
            
            for line in result.stdout.split('\n'):
                if line.strip():
                    try:
                        data = json.loads(line)
                        if data.get("reason") == "compiler-artifact":
                            artifacts.append(data)
                        elif data.get("reason") == "compiler-message":
                            if data.get("message", {}).get("level") == "warning":
                                warnings.append({
                                    "code": data.get("message", {}).get("code", {}).get("code", "unknown"),
                                    "message": data.get("message", {}).get("message", ""),
                                    "count": 1
                                })
                            elif data.get("message", {}).get("level") == "error":
                                errors.append({
                                    "code": data.get("message", {}).get("code", {}).get("code", "unknown"),
                                    "message": data.get("message", {}).get("message", ""),
                                    "symbol": self._extract_symbol(data.get("message", {}).get("message", ""))
                                })
                    except json.JSONDecodeError:
                        continue
            
            # Find our target crate
            target_crate = "redoxa-core"
            target_triple = "aarch64-apple-darwin"  # Default for macOS ARM
            
            for artifact in artifacts:
                if artifact.get("package_id", "").startswith("redoxa-core"):
                    target_crate = artifact.get("package_id", "").split("@")[0].split("+")[-1]
                    break
            
            # Create certificate
            cert = BuildCertificate(
                crate=target_crate,
                target=target_triple,
                features=self._extract_features(args),
                duration_ms=duration_ms,
                success=success,
                errors=errors,
                warnings=warnings,
                argv_cid=argv_cid,
                witness={
                    "rustc_version": self._get_rustc_version(),
                    "cargo_version": self._get_cargo_version(),
                    "build_args": " ".join(args)
                }
            )
            
            # Save certificate
            self._save_certificate(cert)
            
            # Capture compilation signal (source → binary)
            self._capture_compilation_signal(argv_cid, result)
            
            # Feed into shadow ledger for learning
            self._learn_from_certificate(cert)
            
            return cert
            
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            cert = BuildCertificate(
                crate="redoxa-core",
                target="aarch64-apple-darwin",
                features=self._extract_features(args),
                duration_ms=duration_ms,
                success=False,
                errors=[{"code": "BUILD_ERROR", "message": str(e)}],
                argv_cid=argv_cid,
                witness={"error": "build_exception"}
            )
            self._save_certificate(cert)
            
            # Feed into shadow ledger for learning
            self._learn_from_certificate(cert)
            
            return cert
    
    def _generate_cid(self, cmd: List[str]) -> str:
        """Generate a CID for the build command"""
        cmd_str = " ".join(cmd)
        return hashlib.sha256(cmd_str.encode()).hexdigest()[:16]
    
    def _extract_features(self, args: List[str]) -> List[str]:
        """Extract features from build arguments"""
        features = []
        for i, arg in enumerate(args):
            if arg == "--features" and i + 1 < len(args):
                features = args[i + 1].split(",")
                break
        return features
    
    def _extract_symbol(self, message: str) -> str:
        """Extract symbol name from linker error message"""
        if "_Py" in message:
            # Extract Python symbol
            import re
            match = re.search(r'_Py[A-Za-z_]+', message)
            if match:
                return match.group(0)
        return ""
    
    def _get_rustc_version(self) -> str:
        """Get rustc version"""
        try:
            result = subprocess.run(["rustc", "--version"], capture_output=True, text=True)
            return result.stdout.strip()
        except:
            return "unknown"
    
    def _get_cargo_version(self) -> str:
        """Get cargo version"""
        try:
            result = subprocess.run(["cargo", "--version"], capture_output=True, text=True)
            return result.stdout.strip()
        except:
            return "unknown"
    
    def _save_certificate(self, cert: BuildCertificate):
        """Save certificate to file"""
        timestamp = int(time.time())
        filename = f"{timestamp}_{cert.crate}_{cert.argv_cid}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(asdict(cert), f, indent=2)
        
        print(f"Saved certificate: {filepath}")
    
    def _tap_build_artifacts(self, argv_cid: str):
        """Tap into build artifacts before they get cleaned - capture training data"""
        if not self.shadow_ledger:
            return
        
        try:
            # Scan target directory for build artifacts
            target_dir = Path("core/target")
            if not target_dir.exists():
                return
            
            # Collect artifact metadata
            artifacts = []
            total_size = 0
            
            for artifact_path in target_dir.rglob("*"):
                if artifact_path.is_file():
                    size = artifact_path.stat().st_size
                    total_size += size
                    
                    artifacts.append({
                        "path": str(artifact_path.relative_to(target_dir)),
                        "size": size,
                        "modified": artifact_path.stat().st_mtime
                    })
            
            # Create artifact summary
            artifact_summary = f"Build Artifacts: {len(artifacts)} files, {total_size / (1024*1024):.1f}MB\n"
            artifact_summary += f"Largest artifacts:\n"
            
            # Sort by size and show top 10
            sorted_artifacts = sorted(artifacts, key=lambda x: x['size'], reverse=True)
            for artifact in sorted_artifacts[:10]:
                size_mb = artifact['size'] / (1024*1024)
                if size_mb > 0.1:  # Only show files > 100KB
                    artifact_summary += f"  - {artifact['path']}: {size_mb:.1f}MB\n"
            
            # Create resource metrics for artifacts
            resource_metrics = {
                "artifact_count": len(artifacts),
                "total_size_mb": total_size / (1024*1024),
                "largest_artifact_mb": sorted_artifacts[0]['size'] / (1024*1024) if sorted_artifacts else 0,
                "cpu_avg": 0.0,  # Artifact collection doesn't use CPU
                "memory_avg": 0.0,
                "wall_time": 0.0
            }
            
            # Feed artifact data into shadow ledger as penumbra (compressed but recoverable)
            script_name = f"artifacts_{argv_cid}.json"
            self.shadow_ledger.fold_into_penumbra(script_name, artifact_summary, resource_metrics)
            print(f"🔍 Tapped {len(artifacts)} artifacts ({total_size / (1024*1024):.1f}MB) into penumbra")
            
        except Exception as e:
            print(f"⚠️  Failed to tap build artifacts: {e}")
    
    def _capture_compilation_signal(self, argv_cid: str, result: subprocess.CompletedProcess):
        """Capture the source → binary compilation signal"""
        if not self.shadow_ledger:
            return
        
        try:
            # Parse the compilation signal from JSON output
            compilation_signals = []
            source_files = []
            binary_files = []
            
            for line in result.stdout.split('\n'):
                if line.strip():
                    try:
                        data = json.loads(line)
                        if data.get("reason") == "compiler-artifact":
                            # This is a source → binary transformation
                            package_id = data.get("package_id", "")
                            target = data.get("target", {})
                            filenames = data.get("filenames", [])
                            
                            # Extract source info
                            source_info = {
                                "package": package_id.split("@")[0] if "@" in package_id else package_id,
                                "target_name": target.get("name", ""),
                                "target_kind": target.get("kind", [""])[0] if target.get("kind") else "",
                                "crate_types": target.get("crate_types", [])
                            }
                            
                            # Extract binary info - find actual .rlib files
                            package_name = package_id.split("@")[0] if "@" in package_id else package_id
                            target_dir = Path("core/target")
                            actual_artifacts = self._find_actual_artifacts(package_name, target_dir)
                            
                            binary_info = {
                                "filenames": filenames,
                                "actual_artifacts": actual_artifacts,
                                "total_size": sum(os.path.getsize(f) for f in actual_artifacts if os.path.exists(f)),
                                "artifact_count": len(actual_artifacts)
                            }
                            
                            # Create compilation signal - focus on the actual transformation
                            signal = {
                                "source": source_info,
                                "binary": binary_info,
                                "transformation": {
                                    "binary_size": binary_info["total_size"],
                                    "artifact_count": binary_info["artifact_count"],
                                    "crate_types": source_info["crate_types"],
                                    "target_kind": source_info["target_kind"]
                                }
                            }
                            
                            compilation_signals.append(signal)
                            
                    except json.JSONDecodeError:
                        continue
            
            # Create compilation signal summary
            signal_summary = f"Compilation Signal: {len(compilation_signals)} transformations\n"
            signal_summary += f"Source → Binary mappings:\n"
            
            for signal in compilation_signals:
                source = signal["source"]
                binary = signal["binary"]
                transform = signal["transformation"]
                
                signal_summary += f"  {source['package']} ({source['target_kind']}) → {binary['artifact_count']} files ({binary['total_size']/1024:.1f}KB)\n"
                signal_summary += f"    Types: {', '.join(transform['crate_types'])}\n"
            
            # Create resource metrics for compilation signal
            resource_metrics = {
                "transformation_count": len(compilation_signals),
                "total_binary_size": sum(s["binary"]["total_size"] for s in compilation_signals),
                "avg_binary_size": sum(s["binary"]["total_size"] for s in compilation_signals) / max(1, len(compilation_signals)),
                "cpu_avg": 100.0,  # Compilation is CPU intensive
                "memory_avg": 80.0,  # Compilation uses memory
                "wall_time": 0.0
            }
            
            # Feed compilation signal into shadow ledger as illuminated (this is the core learning)
            script_name = f"compilation_signal_{argv_cid}.json"
            self.shadow_ledger.illuminate(script_name, signal_summary, resource_metrics)
            print(f"📡 Captured {len(compilation_signals)} compilation signals into illuminated realm")
            
        except Exception as e:
            print(f"⚠️  Failed to capture compilation signal: {e}")
    
    def _find_actual_artifacts(self, package_name: str, target_dir: Path) -> List[str]:
        """Find actual .rlib files for a package"""
        artifacts = []
        
        # Clean package name
        clean_name = package_name.replace("registry+https://github.com/rust-lang/crates.io-index#", "")
        
        # Look for .rlib files that match this package
        for artifact_path in target_dir.rglob("*.rlib"):
            if clean_name in artifact_path.name:
                artifacts.append(str(artifact_path))
        
        # Also look for .dylib files
        for artifact_path in target_dir.rglob("*.dylib"):
            if clean_name in artifact_path.name:
                artifacts.append(str(artifact_path))
        
        # Also look for .a files (static libs)
        for artifact_path in target_dir.rglob("*.a"):
            if clean_name in artifact_path.name:
                artifacts.append(str(artifact_path))
        
        return artifacts
    
    def _estimate_source_lines(self, package_name: str) -> int:
        """Count actual source lines for a package"""
        try:
            # Extract package name from package_id
            if "@" in package_name:
                pkg_name = package_name.split("@")[0]
            else:
                pkg_name = package_name
            
            # For local packages, count our actual source
            if "redoxa-core" in pkg_name:
                source_dir = Path("core/src")
                if not source_dir.exists():
                    return 1000
                
                total_lines = 0
                for source_file in source_dir.rglob("*.rs"):
                    try:
                        with open(source_file, 'r') as f:
                            total_lines += len(f.readlines())
                    except:
                        continue
                
                return total_lines
            
            # For external packages, try to find them in Cargo cache
            cargo_home = Path.home() / ".cargo" / "registry" / "src"
            if cargo_home.exists():
                # Look for the package in the registry
                for registry_dir in cargo_home.iterdir():
                    if registry_dir.is_dir():
                        for pkg_dir in registry_dir.rglob(f"*{pkg_name}*"):
                            if pkg_dir.is_dir():
                                # Count actual source lines
                                total_lines = 0
                                for source_file in pkg_dir.rglob("*.rs"):
                                    try:
                                        with open(source_file, 'r') as f:
                                            total_lines += len(f.readlines())
                                    except:
                                        continue
                                
                                if total_lines > 0:
                                    return total_lines
            
            # Fallback: try to find in target directory
            target_dir = Path("core/target")
            if target_dir.exists():
                for pkg_dir in target_dir.rglob(f"*{pkg_name}*"):
                    if pkg_dir.is_dir():
                        total_lines = 0
                        for source_file in pkg_dir.rglob("*.rs"):
                            try:
                                with open(source_file, 'r') as f:
                                    total_lines += len(f.readlines())
                            except:
                                continue
                        
                        if total_lines > 0:
                            return total_lines
            
            return 1000  # Default if we can't find the source
        except:
            return 1000  # Default estimate
    
    def _learn_from_certificate(self, cert: BuildCertificate):
        """Feed certificate into shadow ledger for learning"""
        if not self.shadow_ledger:
            return
        
        # Create script name from build args
        script_name = f"build_{cert.argv_cid}.json"
        
        # Create output representation
        output = self._format_certificate_output(cert)
        
        # Create resource metrics from build data
        resource_metrics = {
            "duration_ms": cert.duration_ms,
            "cpu_avg": 100.0 if cert.success else 0.0,  # Simplified
            "memory_avg": 50.0,  # Simplified
            "wall_time": cert.duration_ms / 1000.0,
            "warnings_count": len(cert.warnings),
            "errors_count": len(cert.errors)
        }
        
        # Feed into shadow ledger based on success/failure
        if cert.success:
            if len(cert.warnings) == 0:
                # Perfect build - illuminate
                self.shadow_ledger.illuminate(script_name, output, resource_metrics)
                print(f"🌟 Illuminated build: {script_name}")
            else:
                # Build with warnings - penumbra
                self.shadow_ledger.fold_into_penumbra(script_name, output, resource_metrics)
                print(f"🌓 Folded into penumbra: {script_name} ({len(cert.warnings)} warnings)")
        else:
            # Failed build - umbra
            self.shadow_ledger.fold_into_umbra(script_name, output, resource_metrics)
            print(f"🌑 Folded into umbra: {script_name} ({len(cert.errors)} errors)")
    
    def _format_certificate_output(self, cert: BuildCertificate) -> str:
        """Format certificate as readable output"""
        output = f"Build Certificate: {cert.crate}\n"
        output += f"Target: {cert.target}\n"
        output += f"Features: {', '.join(cert.features)}\n"
        output += f"Duration: {cert.duration_ms}ms\n"
        output += f"Success: {cert.success}\n"
        
        if cert.warnings:
            output += f"\nWarnings ({len(cert.warnings)}):\n"
            for warning in cert.warnings:
                output += f"  - {warning['code']}: {warning['message']}\n"
        
        if cert.errors:
            output += f"\nErrors ({len(cert.errors)}):\n"
            for error in cert.errors:
                output += f"  - {error['code']}: {error['message']}\n"
        
        output += f"\nWitness: {cert.witness}\n"
        return output
    
    def run_training_matrix(self):
        """Run a matrix of builds for training"""
        print("Running build training matrix...")
        
        # Matrix of build configurations
        configs = [
            ["--no-default-features", "--features", "standalone"],
            ["--no-default-features", "--features", "standalone", "--release"],
            ["--no-default-features", "--features", "standalone", "--profile", "dev"],
        ]
        
        results = []
        for config in configs:
            print(f"\n--- Testing config: {' '.join(config)} ---")
            cert = self.run_build(config)
            results.append(cert)
            
            # Brief pause between builds
            time.sleep(1)
        
        # Generate summary
        self._generate_summary(results)
        
        return results
    
    def _generate_summary(self, results: List[BuildCertificate]):
        """Generate a summary of build results"""
        summary = {
            "timestamp": int(time.time()),
            "total_builds": len(results),
            "successful_builds": sum(1 for r in results if r.success),
            "failed_builds": sum(1 for r in results if not r.success),
            "avg_duration_ms": sum(r.duration_ms for r in results) / len(results) if results else 0,
            "builds": [asdict(r) for r in results]
        }
        
        summary_file = self.output_dir / f"summary_{int(time.time())}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nGenerated summary: {summary_file}")
        print(f"Successful builds: {summary['successful_builds']}/{summary['total_builds']}")
        print(f"Average duration: {summary['avg_duration_ms']:.1f}ms")
        
        # Generate learning report from shadow ledger
        if self.shadow_ledger:
            self._generate_learning_report()
    
    def _generate_learning_report(self):
        """Generate learning report from shadow ledger"""
        try:
            report = self.shadow_ledger.generate_report()
            report_file = self.output_dir / f"learning_report_{int(time.time())}.md"
            
            with open(report_file, 'w') as f:
                f.write(report)
            
            print(f"\n📚 Generated learning report: {report_file}")
            
            # Print summary
            timeline = self.shadow_ledger.get_timeline()
            illuminated = len([t for t in timeline if t['realm'] == '🌟'])
            penumbra = len([t for t in timeline if t['realm'] == '🌓'])
            umbra = len([t for t in timeline if t['realm'] == '🌑'])
            
            print(f"📊 Learning Summary:")
            print(f"   🌟 Illuminated: {illuminated} (perfect builds)")
            print(f"   🌓 Penumbra: {penumbra} (builds with warnings)")
            print(f"   🌑 Umbra: {umbra} (failed builds)")
            
        except Exception as e:
            print(f"⚠️  Failed to generate learning report: {e}")

def main():
    """Main entry point"""
    trainer = BuildTrainer()
    
    if len(sys.argv) > 1:
        # Run specific build command
        args = sys.argv[1:]
        cert = trainer.run_build(args)
        print(f"Build {'succeeded' if cert.success else 'failed'}")
        print(f"Duration: {cert.duration_ms}ms")
        print(f"Warnings: {len(cert.warnings)}")
        print(f"Errors: {len(cert.errors)}")
    else:
        # Run training matrix
        trainer.run_training_matrix()

if __name__ == "__main__":
    main()
