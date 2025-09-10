#!/usr/bin/env python3
"""
Unified Demo Runner with CE1 Seed Fusion

Runs demos with intelligent execution planning using the CE1 seed fusion (living lattice organism).
The CE1 seed fusion unifies oracle and planner into a self-contained, reversible system that learns and evolves.

AUTODETECTION: Automatically detects Python files with main functions or __name__ == '__main__' guards.
Only executable scripts are run - library files are automatically skipped.

Default behavior: Run all executable demos with CE1 seed fusion (oracle-planner unity)
Override with specific demos: python src/scripts/run.py src/demos/audio_caption_loop.py

Usage:
    python src/scripts/run.py                       # Run all executable demos with CE1 (default)
    python src/scripts/run.py <script1> <script2>   # Run specific executable demos with CE1
    python src/scripts/run.py <directory>           # Run all executable demos in directory with CE1
    python src/scripts/run.py -l                    # List available demos (shows executable vs library)
    python src/scripts/run.py -f                    # Use fixed execution (no intelligence)
    
Examples:
    python src/scripts/run.py                       # Run all executable scripts with living lattice organism
    python src/scripts/run.py src/demos/audio_caption_loop.py  # Run specific executable demo with CE1
    python src/scripts/run.py src/demos/ce1_*.py    # Run all executable CE1 demos with CE1
    python src/scripts/run.py src/demos             # Run all executable demos in src/demos/ with CE1
    python src/scripts/run.py -l                    # See what's available (executable vs library files)
    python src/scripts/run.py -n                    # Dry run (show what would be done)
    python src/scripts/run.py -j 8 -t 120           # 8 parallel workers, 2min timeout
    python src/scripts/run.py -f                    # Use fixed execution (no intelligence)
"""

import os
import sys
import subprocess
import argparse
import re
import time
import psutil
import threading
import ast
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue

# Add orchestrator to path for VM access
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'orchestrator'))

# Import path utilities
from redoxa.paths import get_db_path

try:
    import redoxa_core
    VM_AVAILABLE = True
except ImportError:
    VM_AVAILABLE = False
    print("Warning: Rust VM not available, falling back to standard execution")

# Import shadow-ledger
try:
    from redoxa.simple_shadow_ledger import SimpleShadowLedger
    SHADOW_LEDGER_AVAILABLE = True
    # Create a global ledger instance to persist records across runs
    _global_ledger = None
except ImportError:
    SHADOW_LEDGER_AVAILABLE = False
    _global_ledger = None
    print("Warning: Shadow-ledger not available, falling back to standard reports")

# Import CE1 seed fusion
try:
    from jit import CE1Seed
    CE1_SEED_AVAILABLE = True
except ImportError:
    CE1_SEED_AVAILABLE = False
    print("Warning: CE1 seed fusion not available, falling back to standard planning")

class ResourceMonitor:
    """Monitor system resources during script execution"""
    
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.monitoring = False
        self.metrics = []
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start resource monitoring in background thread"""
        self.monitoring = True
        self.metrics = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self) -> Dict:
        """Stop monitoring and return summary metrics"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
            
        if not self.metrics:
            return {}
            
        # Calculate summary statistics
        cpu_values = [m['cpu'] for m in self.metrics]
        memory_values = [m['memory'] for m in self.metrics]
        
        return {
            'duration': len(self.metrics) * self.interval,
            'cpu_avg': sum(cpu_values) / len(cpu_values),
            'cpu_max': max(cpu_values),
            'memory_avg': sum(memory_values) / len(memory_values),
            'memory_max': max(memory_values),
            'samples': len(self.metrics)
        }
        
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                
                self.metrics.append({
                    'timestamp': time.time(),
                    'cpu': cpu_percent,
                    'memory': memory.percent,
                    'memory_available': memory.available / (1024**3)  # GB
                })
                
                time.sleep(self.interval)
            except Exception:
                break

class PythonFileAnalyzer:
    """Analyzes Python files to detect main functions and executable scripts"""
    
    def __init__(self):
        self.main_files = set()
        self.executable_files = set()
    
    def has_main_function(self, file_path: str) -> bool:
        """Check if a Python file has a main function"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the AST
            tree = ast.parse(content)
            
            # Look for main function definition
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == 'main':
                    return True
            
            return False
        except Exception:
            return False
    
    def has_main_guard(self, file_path: str) -> bool:
        """Check if a Python file has if __name__ == '__main__' guard"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the AST
            tree = ast.parse(content)
            
            # Look for if __name__ == '__main__' guard
            for node in ast.walk(tree):
                if isinstance(node, ast.If):
                    if (isinstance(node.test, ast.Compare) and
                        isinstance(node.test.left, ast.Name) and
                        node.test.left.id == '__name__' and
                        len(node.test.comparators) == 1 and
                        isinstance(node.test.comparators[0], ast.Constant) and
                        node.test.comparators[0].value == '__main__'):
                        return True
            
            return False
        except Exception:
            return False
    
    def is_executable_script(self, file_path: str) -> bool:
        """Check if a Python file is executable (has main function or main guard)"""
        return self.has_main_function(file_path) or self.has_main_guard(file_path)
    
    def analyze_directory(self, directory: str) -> Tuple[List[str], List[str]]:
        """Analyze directory for executable Python scripts"""
        executable_files = []
        non_executable_files = []
        
        if not os.path.exists(directory):
            return executable_files, non_executable_files
        
        for filename in os.listdir(directory):
            if filename.endswith('.py') and not filename.startswith('__'):
                file_path = os.path.join(directory, filename)
                if self.is_executable_script(file_path):
                    executable_files.append(filename)
                else:
                    non_executable_files.append(filename)
        
        return sorted(executable_files), sorted(non_executable_files)

class DemoProfiler:
    """Profiles demo characteristics to inform execution planning"""
    
    def __init__(self):
        self.demo_profiles = {}
        self.analyzer = PythonFileAnalyzer()
        
    def analyze_script(self, script_path: str) -> dict:
        """Analyze a script to determine its characteristics"""
        try:
            with open(script_path, 'r') as f:
                content = f.read()
                
            # Analyze script characteristics
            profile = {
                'size': len(content),
                'lines': content.count('\n'),
                'imports': len(re.findall(r'^import\s+|^from\s+', content, re.MULTILINE)),
                'has_numpy': 'numpy' in content or 'np.' in content,
                'has_matplotlib': 'matplotlib' in content or 'plt.' in content,
                'has_rust_vm': 'redoxa' in content or 'VM(' in content,
                'has_ce1': 'ce1_' in content or 'CE1' in content,
                'has_async': 'async' in content or 'await' in content,
                'has_threading': 'threading' in content or 'Thread' in content,
                'estimated_complexity': self._estimate_complexity(content)
            }
            
            return profile
            
        except Exception as e:
            return {'error': str(e)}
    
    def _estimate_complexity(self, content: str) -> str:
        """Estimate script complexity based on content analysis"""
        lines = content.count('\n')
        imports = len(re.findall(r'^import\s+|^from\s+', content, re.MULTILINE))
        
        if lines < 50 and imports < 5:
            return 'simple'
        elif lines < 200 and imports < 15:
            return 'medium'
        else:
            return 'complex'

class OperationalPlanner:
    """Plans execution strategy using Rust VM's operational planning"""
    
    def __init__(self):
        self.vm = None
        if VM_AVAILABLE:
            try:
                self.vm = redoxa_core.VM(get_db_path("redoxa_operational.db"))
                print("✓ Rust VM operational planner available")
            except Exception as e:
                print(f"Warning: Could not initialize VM: {e}")
                self.vm = None
    
    def plan_execution(self, demo_paths: List[str], max_workers: int = 4, 
                      timeout_seconds: int = 60) -> Tuple[str, float]:
        """Plan execution strategy using CE1 search"""
        if not self.vm:
            # Fallback to simple strategy
            return f"workers={max_workers}, timeout={timeout_seconds}s, retry=exponential", 0.0
        
        try:
            # Use Rust VM's operational planning
            strategy, cost = self.vm.plan_operational_execution(
                demo_paths, max_workers, timeout_seconds
            )
            return strategy, cost
        except Exception as e:
            print(f"Warning: Operational planning failed: {e}")
            return f"workers={max_workers}, timeout={timeout_seconds}s, retry=exponential", 0.0

def run_script(directory: str, script_name: str, timeout: int = 30, 
               monitor_resources: bool = False, use_vm: bool = False) -> Tuple[str, int, Dict]:
    """
    Run a script with optional resource monitoring and VM integration
    
    Returns:
        Tuple of (output, exit_code, resource_metrics)
    """
    script_path = os.path.join(directory, script_name)
    
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found: {script_path}")
    
    print(f"Running {script_name} in {directory} (timeout: {timeout}s)...")
    
    # Start resource monitoring if requested
    monitor = None
    if monitor_resources:
        monitor = ResourceMonitor()
        monitor.start_monitoring()
    
    start_time = time.time()
    
    try:
        # Run the script
        # Get absolute path to .out/__pycache__ from project root
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        cache_dir = os.path.join(project_root, '.out', '__pycache__')
        result = subprocess.run(
            [sys.executable, "-X", f"pycache_prefix={cache_dir}", script_name],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        # Combine stdout and stderr
        output = result.stdout
        if result.stderr:
            output += "\n" + "="*50 + " STDERR " + "="*50 + "\n"
            output += result.stderr
        
        exit_code = result.returncode
        
    except subprocess.TimeoutExpired:
        output = f"ERROR: Script timed out after {timeout} seconds"
        exit_code = 1
    except Exception as e:
        output = f"ERROR: Failed to run script: {e}"
        exit_code = 1
    
    # Stop monitoring and get metrics
    resource_metrics = {}
    if monitor:
        resource_metrics = monitor.stop_monitoring()
        resource_metrics['wall_time'] = time.time() - start_time
    
    return output, exit_code, resource_metrics

def find_or_create_section(content: str, section_name: str) -> Tuple[str, int, int]:
    """Find a section in markdown content or create it if it doesn't exist"""
    # Look for existing section
    section_pattern = rf"^## \*\*{re.escape(section_name)}\*\*.*$"
    match = re.search(section_pattern, content, re.MULTILINE)
    
    if match:
        # Find the end of this section (next ## or end of file)
        start_pos = match.start()
        next_section = re.search(r"^## ", content[match.end():], re.MULTILINE)
        
        if next_section:
            end_pos = match.end() + next_section.start()
        else:
            end_pos = len(content)
            
        return content, start_pos, end_pos
    else:
        # Create new section at the end
        new_section = f"\n## **{section_name}**\n\n"
        content += new_section
        start_pos = len(content) - len(new_section)
        end_pos = len(content)
        
        return content, start_pos, end_pos

def update_report(directory: str, script_name: str, output: str, exit_code: int, 
                 resource_metrics: Dict, strategy: str = None, replace: bool = False, 
                 section_name: str = None, timeout: int = 30, vm=None) -> None:
    """Update the report using shadow-ledger cosmology"""
    report_path = os.path.join(directory, "report.md")
    
    if SHADOW_LEDGER_AVAILABLE:
        # Use global ledger instance to persist records across runs
        global _global_ledger
        if _global_ledger is None:
            _global_ledger = SimpleShadowLedger(vm=vm)
        
        # Add strategy to resource metrics
        if strategy:
            resource_metrics['strategy'] = strategy
        
        # Add run to ledger (determines realm automatically)
        _global_ledger.add_run(script_name, exit_code, output, resource_metrics)
        
        # Generate report
        new_content = _global_ledger.generate_report()
        
        # Write updated content
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"Updated {report_path} using shadow-ledger cosmology")
    else:
        # Fallback to standard report system
        print("Warning: Using fallback report system")
        _update_report_fallback(directory, script_name, output, exit_code, resource_metrics, strategy, replace, section_name, timeout)

def _update_report_fallback(directory: str, script_name: str, output: str, exit_code: int, 
                           resource_metrics: Dict, strategy: str = None, replace: bool = False, 
                           section_name: str = None, timeout: int = 30) -> None:
    """Fallback report system when shadow-ledger is not available"""
    report_path = os.path.join(directory, "report.md")
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Format output
    status = "✅ SUCCESS" if exit_code == 0 else "❌ FAILED"
    timeout_info = f" (timeout: {timeout}s)" if exit_code != 0 and "timed out" in output else ""
    
    # Create run record for metadata table
    run_record = {
        'script': script_name,
        'status': status,
        'timestamp': timestamp,
        'exit_code': exit_code,
        'wall_time': resource_metrics.get('wall_time', 0) if resource_metrics else 0,
        'cpu_avg': resource_metrics.get('cpu_avg', 0) if resource_metrics else 0,
        'memory_avg': resource_metrics.get('memory_avg', 0) if resource_metrics else 0,
        'output': output,
        'strategy': strategy
    }
    
    if not os.path.exists(report_path):
        print(f"Creating new report.md in {directory}")
        content = f"# {directory.title()} Report\n\n"
        runs_history = []
    else:
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        runs_history = _extract_runs_history(content)
    
    # Add new run to history
    runs_history.append(run_record)
    
    # Generate new report with table format
    new_content = _generate_table_report(directory, runs_history)
    
    # Write updated content
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"Updated {report_path}")

def _extract_runs_history(content: str) -> List[Dict]:
    """Extract run history from existing report"""
    import re
    
    runs = []
    
    # Extract runs from metadata table - improved regex
    table_matches = re.findall(r'\| `([^`]+)` \| ([^|]+) \| ([^|]+) \| ([^|]+) \| ([^|]+) \| ([^|]+) \|', content)
    
    # Debug: print what we found
    if table_matches:
        print(f"Found {len(table_matches)} table entries")
    else:
        print("No table entries found")
    
    for match in table_matches:
        script, status, timestamp, exit_code, wall_time, cpu_avg = match
        runs.append({
            'script': script.strip(),
            'status': status.strip(),
            'timestamp': timestamp.strip(),
            'exit_code': int(exit_code.strip()) if exit_code.strip().isdigit() else 0,
            'wall_time': float(wall_time.strip()) if wall_time.strip().replace('.', '').isdigit() else 0,
            'cpu_avg': float(cpu_avg.strip()) if cpu_avg.strip().replace('.', '').isdigit() else 0,
            'memory_avg': 0,  # Not in table format
            'output': '',  # Will be extracted from sections
            'strategy': ''
        })
    
    # Extract outputs from sections - improved regex to capture all outputs
    section_matches = re.findall(r'## \*\*([^*]+)\*\*\n\n(.*?)(?=## \*\*|\Z)', content, re.DOTALL)
    
    for section_name, section_content in section_matches:
        # Find all successful runs in this section
        success_matches = re.findall(r'\*\*Status\*\*: ✅ SUCCESS.*?\*\*Timestamp\*\*: ([^\n]+).*?```bash\n\$ python [^\n]+\n(.*?)\n```', section_content, re.DOTALL)
        
        for timestamp, output in success_matches:
            # Find the corresponding run record by script name and timestamp
            script_name = section_name.lower().replace(' ', '_').replace('_demo', '_demo.py')
            for run in runs:
                if run['script'] == script_name and run['timestamp'] == timestamp:
                    run['output'] = output.strip()
                    break
    
    return runs

def _generate_table_report(directory: str, runs_history: List[Dict]) -> str:
    """Generate report with metadata table and single output per program"""
    
    # Group runs by script
    script_groups = {}
    for run in runs_history:
        script = run['script']
        if script not in script_groups:
            script_groups[script] = []
        script_groups[script].append(run)
    
    # Create metadata table
    table_header = """# {directory} Report

## Run History

| Script | Status | Timestamp | Exit Code | Wall Time (s) | CPU Avg (%) |
|--------|--------|-----------|-----------|---------------|-------------|
""".format(directory=directory.title())
    
    table_rows = []
    # Show all runs, not just the latest per script
    all_runs = []
    for script, runs in script_groups.items():
        all_runs.extend(runs)
    
    # Sort all runs by timestamp (most recent first)
    all_runs.sort(key=lambda x: x['timestamp'], reverse=True)
    
    for run in all_runs:
        table_rows.append(f"| `{run['script']}` | {run['status']} | {run['timestamp']} | {run['exit_code']} | {run['wall_time']:.2f} | {run['cpu_avg']:.1f} |")
    
    table_content = table_header + '\n'.join(table_rows) + "\n\n"
    
    # Create output sections (one per program, most recent successful run)
    output_sections = ""
    for script, runs in script_groups.items():
        # Sort by timestamp (most recent first)
        runs.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Find most recent successful run
        successful_runs = [r for r in runs if r['status'] == '✅ SUCCESS']
        if successful_runs:
            latest_success = successful_runs[0]
            
            # Count failed runs for compression info
            failed_runs = [r for r in runs if r['status'] == '❌ FAILED']
            failed_count = len(failed_runs)
            
            section_name = script.replace('.py', '').replace('_', ' ').title()
            
            # Add compression info if there were failed runs
            compression_info = ""
            if failed_count > 0:
                first_fail = failed_runs[0]['timestamp']
                last_fail = failed_runs[-1]['timestamp']
                compression_info = f"""
**Compression**: {failed_count} failed runs compressed ({first_fail} → {last_fail})

"""
            
            # Create compact timestamp
            timestamp_parts = latest_success['timestamp'].split(' ')
            date_part = timestamp_parts[0].split('-')[1] + '/' + timestamp_parts[0].split('-')[2]  # MM/DD
            time_part = timestamp_parts[1][:5]  # HH:MM
            compact_timestamp = f"{date_part} {time_part}"
            
            # Create glyph-based status and metrics
            status_glyph = "✓" if latest_success['status'] == '✅ SUCCESS' else "✗"
            cpu_glyph = "🔥" if latest_success['cpu_avg'] > 50 else "⚡" if latest_success['cpu_avg'] > 10 else "💤"
            mem_glyph = "📈" if latest_success['memory_avg'] > 80 else "📊" if latest_success['memory_avg'] > 50 else "📉"
            
            # Only add output if we have it (don't erase existing outputs)
            output_content = latest_success['output'] if latest_success['output'] else ""
            
            output_sections += f"""## **{section_name}**

{compression_info}{status_glyph} `{script}` {compact_timestamp} | {latest_success['wall_time']:.1f}s {cpu_glyph}{latest_success['cpu_avg']:.0f}% {mem_glyph}{latest_success['memory_avg']:.0f}%

```bash
$ python {script}
{output_content}
```

"""
    
    return table_content + output_sections

def _compress_section_history(content: str, start_pos: int, end_pos: int, new_output: str, is_success: bool) -> str:
    """Compress historical runs using smart compression"""
    import re
    
    section_content = content[start_pos:end_pos]
    
    # Extract all runs from the section
    runs = re.findall(r'\*\*Status\*\*: (.*?)\n\*\*Timestamp\*\*: (.*?)\n.*?```bash\n(.*?)\n```', section_content, re.DOTALL)
    
    if not runs:
        # No previous runs found, just append
        return content[:end_pos] + new_output + content[end_pos:]
    
    # Separate successful and failed runs
    successful_runs = []
    failed_runs = []
    
    for status, timestamp, output in runs:
        if "✅ SUCCESS" in status:
            successful_runs.append((status, timestamp, output))
        else:
            failed_runs.append((status, timestamp, output))
    
    # If this is a success, compress all previous failed runs
    if is_success and failed_runs:
        # Create compressed summary of failed runs
        failed_count = len(failed_runs)
        first_fail = failed_runs[0][1]  # First timestamp
        last_fail = failed_runs[-1][1]  # Last timestamp
        
        # Extract common error patterns
        error_patterns = set()
        for _, _, output in failed_runs:
            if "ModuleNotFoundError" in output:
                error_patterns.add("ModuleNotFoundError")
            if "timeout" in output:
                error_patterns.add("timeout")
            if "IndexError" in output:
                error_patterns.add("IndexError")
        
        # Create compressed summary
        compressed_summary = f"""
**Status**: ❌ FAILED (compressed {failed_count} runs)
**Timestamp**: {first_fail} → {last_fail}
**Script**: `{runs[0][2].split('$ python ')[1].split()[0] if '$ python ' in runs[0][2] else 'unknown'}`
**Error Patterns**: {', '.join(error_patterns) if error_patterns else 'various'}
**Compression**: {failed_count} runs → 1 summary (compression ratio: {failed_count:.1f}x)

"""
        
        # Build new section content with compressed runs
        new_section_content = ""
        
        # Add compressed summary
        new_section_content += compressed_summary
        
        # Keep only the most recent successful run (if any)
        if successful_runs:
            latest_success = successful_runs[-1]
            latest_success_output = f"""
**Status**: {latest_success[0]}
**Timestamp**: {latest_success[1]}
**Script**: `{latest_success[2].split('$ python ')[1].split()[0] if '$ python ' in latest_success[2] else 'unknown'}`
```bash
{latest_success[2]}
```

"""
            new_section_content += latest_success_output
        
        # Add the new run
        new_section_content += new_output
        
        # Replace the entire section content
        new_content = content[:start_pos] + new_section_content + content[end_pos:]
    else:
        # Not a success, just append normally
        new_content = content[:end_pos] + new_output + content[end_pos:]
    
    return new_content

def list_available_demos() -> None:
    """List all available demo scripts with autodetection of executable files"""
    demo_dirs = ['src/demos']
    analyzer = PythonFileAnalyzer()
    
    print("Available Demo Scripts:")
    print("=" * 50)
    
    total_executable = 0
    total_files = 0
    
    for demo_dir in demo_dirs:
        if os.path.exists(demo_dir):
            print(f"\n{demo_dir}/")
            executable_files, non_executable_files = analyzer.analyze_directory(demo_dir)
            
            if executable_files:
                print("  🚀 Executable scripts (with main functions):")
                for script in executable_files:
                    print(f"    ✓ {script}")
                total_executable += len(executable_files)
            
            if non_executable_files:
                print("  📚 Library files (no main function):")
                for script in non_executable_files:
                    print(f"    📖 {script}")
                total_files += len(non_executable_files)
            
            total_files += len(executable_files)
    
    print(f"\n📊 Summary: {total_executable} executable scripts, {total_files - total_executable} library files")
    print(f"\nUsage: python src/scripts/run.py [demos...]")
    print(f"Example: python src/scripts/run.py src/demos/audio_caption_loop.py")
    print(f"Auto-run: python src/scripts/run.py  # Runs all executable scripts")

def run_all_demos(directory: str, parallel: int = 1, adaptive: bool = False) -> None:
    """Run all executable demo scripts in a directory (autodetects main functions)"""
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist")
        return
    
    analyzer = PythonFileAnalyzer()
    executable_scripts, non_executable_scripts = analyzer.analyze_directory(directory)
    
    if not executable_scripts:
        print(f"No executable Python scripts found in {directory}")
        if non_executable_scripts:
            print(f"Found {len(non_executable_scripts)} library files (no main functions): {', '.join(non_executable_scripts)}")
        return
    
    print(f"Found {len(executable_scripts)} executable scripts in {directory}")
    if non_executable_scripts:
        print(f"Skipping {len(non_executable_scripts)} library files: {', '.join(non_executable_scripts)}")
    
    if parallel > 1:
        # Run in parallel
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = []
            for script in executable_scripts:
                future = executor.submit(run_single_demo, directory, script, adaptive=adaptive)
                futures.append((script, future))
            
            for script, future in futures:
                try:
                    future.result()
                except Exception as e:
                    print(f"Error running {script}: {e}")
    else:
        # Run sequentially
        for script in executable_scripts:
            run_single_demo(directory, script, adaptive=adaptive)

def run_single_demo(directory: str, script: str, adaptive: bool = False, 
                   use_vm: bool = False, timeout: int = 30, use_ce1: bool = False) -> None:
    """Run a single demo script"""
    try:
        # Plan execution strategy
        strategy = None
        vm = None
        ce1_result = None
        
        if use_ce1 and CE1_SEED_AVAILABLE:
            # Use CE1 seed fusion for intelligent planning
            vm = redoxa_core.VM(get_db_path("ce1_seed.db")) if VM_AVAILABLE else None
            seed = CE1Seed(vm=vm)
            
            # Get oracle hint and planner action
            hint_result = seed.hint(script)
            plan_result = seed.plan(script)
            
            print(f"🔮 Oracle hint: confidence={hint_result['confidence']:.3f}, energy={hint_result['energy']:.3f}")
            print(f"🎯 Planner action: {plan_result['action']['type']} - {plan_result['action']['rationale']}")
            
            # Execute the complete loop
            ce1_result = seed.loop(script)
            strategy = f"CE1-{plan_result['action']['type']}"
            
        elif use_vm and VM_AVAILABLE:
            # Fallback to standard VM planning
            planner = OperationalPlanner()
            strategy, cost = planner.plan_execution([script], max_workers=1, timeout_seconds=timeout)
            print(f"Planned strategy: {strategy} (cost: {cost:.3f})")
            vm = planner.vm
        
        # Run the script
        output, exit_code, resource_metrics = run_script(
            directory, script, timeout, 
            monitor_resources=adaptive, use_vm=use_vm
        )
        
        # Update report
        update_report(directory, script, output, exit_code, resource_metrics, 
                     strategy, timeout=timeout, vm=vm)
        
        # Print summary with CE1 info
        status = "✅ SUCCESS" if exit_code == 0 else "❌ FAILED"
        if ce1_result:
            print(f"{status}: {script} completed with exit code {exit_code}")
            print(f"🌱 CE1 prior evolved: θ={ce1_result['tick']['reseed']['new_prior']['theta']:.2f}, φ={ce1_result['tick']['reseed']['new_prior']['phi']:.2f}")
        else:
            print(f"{status}: {script} completed with exit code {exit_code}")
        
    except Exception as e:
        print(f"Error running {script}: {e}")

def find_all_demos() -> List[Tuple[str, str]]:
    """Find all executable demo scripts in the project (autodetects main functions)"""
    demo_dirs = ['src/demos']
    all_demos = []
    analyzer = PythonFileAnalyzer()
    
    for demo_dir in demo_dirs:
        if os.path.exists(demo_dir):
            executable_scripts, _ = analyzer.analyze_directory(demo_dir)
            for script in executable_scripts:
                all_demos.append((demo_dir, script))
    
    return all_demos

def expand_demo_patterns(patterns: List[str]) -> List[Tuple[str, str]]:
    """Expand demo patterns and directories into actual executable demo paths (autodetects main functions)"""
    import glob
    
    expanded = []
    analyzer = PythonFileAnalyzer()
    
    for pattern in patterns:
        if os.path.isdir(pattern):
            # It's a directory, include only executable Python files
            executable_scripts, _ = analyzer.analyze_directory(pattern)
            for script in executable_scripts:
                expanded.append((pattern, script))
        elif '/' in pattern:
            # Pattern like "src/demos/ce1_*.py"
            matches = glob.glob(pattern)
            for match in matches:
                if os.path.exists(match) and analyzer.is_executable_script(match):
                    directory, script = os.path.split(match)
                    expanded.append((directory, script))
        else:
            # Just a script name, find it in all directories and check if executable
            demo_dirs = ['src/demos']
            for demo_dir in demo_dirs:
                script_path = os.path.join(demo_dir, pattern)
                if os.path.exists(script_path) and analyzer.is_executable_script(script_path):
                    expanded.append((demo_dir, pattern))
                    break
    
    return expanded

def main():
    parser = argparse.ArgumentParser(description="Unified demo runner with CE1 seed fusion (living lattice organism) - autodetects executable scripts")
    parser.add_argument("demos", nargs='*', help="Specific demos or directories to run (default: run all executable scripts)")
    parser.add_argument("-l", "--list", action="store_true", help="List available demo scripts (shows executable vs library files)")
    parser.add_argument("-j", "--jobs", type=int, help="Number of parallel workers (auto-detect if not specified)")
    parser.add_argument("-t", "--timeout", type=int, default=60, help="Timeout in seconds (default: 60)")
    parser.add_argument("-n", "--dry", action="store_true", help="Show what would be done without actually running")
    parser.add_argument("--no-vm", action="store_true", help="Disable VM operational planning")
    parser.add_argument("--no-monitor", action="store_true", help="Disable resource monitoring")
    parser.add_argument("-f", "--fixed", action="store_true", help="Use fixed execution (just run scripts, no intelligence)")
    parser.add_argument("--vm-only", action="store_true", help="Use Rust VM planning only (no CE1 learning system)")
    
    args = parser.parse_args()
    
    # Handle list command
    if args.list:
        list_available_demos()
        return
    
    # Determine which demos to run
    if args.demos:
        # Run specific demos or directories
        demos_to_run = expand_demo_patterns(args.demos)
        if not demos_to_run:
            print("Error: No matching demos found")
            sys.exit(1)
    else:
        # Default: run all demos
        demos_to_run = find_all_demos()
    
    # Auto-detect optimal parallelism if not specified
    if args.jobs is None:
        args.jobs = min(len(demos_to_run), os.cpu_count() or 4)
    
    # Determine execution settings
    use_vm = not args.no_vm and VM_AVAILABLE
    monitor_resources = not args.no_monitor
    
    # Execution mode logic
    if args.fixed:
        use_ce1 = False
        use_vm = False
        execution_mode = "fixed"
    elif args.vm_only:
        use_ce1 = False
        use_vm = True
        execution_mode = "vm-only"
    else:
        use_ce1 = CE1_SEED_AVAILABLE  # CE1 is default!
        execution_mode = "ce1"
    
    if args.dry:
        print(f"Would run {len(demos_to_run)} demos:")
        for directory, script in demos_to_run:
            print(f"  {directory}/{script}")
        print(f"Parallel workers: {args.jobs}")
        print(f"VM planning: {use_vm}")
        print(f"Execution mode: {execution_mode}")
        print(f"CE1 seed fusion: {use_ce1}")
        print(f"VM planning: {use_vm}")
        print(f"Resource monitoring: {monitor_resources}")
        print(f"Timeout: {args.timeout}s")
        return
    
    # Run the demos
    print(f"Running {len(demos_to_run)} demos with {args.jobs} parallel workers...")
    if use_ce1:
        print("🌱 Using CE1 seed fusion for intelligent planning (default)")
    elif use_vm:
        print("✓ Using Rust VM operational planning (CE1 disabled)")
    else:
        print("⚠️  Using basic execution (CE1 and VM disabled)")
    if monitor_resources:
        print("✓ Resource monitoring enabled")
    
    if args.jobs > 1:
        # Run in parallel
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = []
            for directory, script in demos_to_run:
                future = executor.submit(run_single_demo, directory, script, 
                                       monitor_resources, use_vm, args.timeout, use_ce1)
                futures.append((directory, script, future))
            
            for directory, script, future in futures:
                try:
                    future.result()
                except Exception as e:
                    print(f"Error running {directory}/{script}: {e}")
    else:
        # Run sequentially
        for directory, script in demos_to_run:
            run_single_demo(directory, script, monitor_resources, use_vm, args.timeout, use_ce1)

if __name__ == "__main__":
    main()
