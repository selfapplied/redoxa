#!/usr/bin/env python3
"""
RUSTC_WRAPPER for training data collection

This script can be used as a RUSTC_WRAPPER to collect detailed training data
from every rustc invocation during the build process.
"""

import sys
import subprocess
import time
import json
import os
from pathlib import Path
from typing import Dict, List, Any

class RustcTrainerWrapper:
    """Wrapper for rustc that collects training data"""
    
    def __init__(self, output_dir: str = ".out/reports/rustc"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def run_rustc(self, args: List[str]) -> int:
        """Run rustc with training data collection"""
        start_time = time.time()
        
        # Find the real rustc
        rustc_path = self._find_rustc()
        if not rustc_path:
            print("Error: Could not find rustc", file=sys.stderr)
            return 1
        
        # Run rustc - args already include 'rustc' as first element
        try:
            result = subprocess.run(
                [rustc_path] + args[1:],  # Skip 'rustc' from args
                capture_output=True,
                text=True
            )
            
            duration_ms = int((time.time() - start_time) * 1000)
            
            # Collect training data
            self._collect_training_data(args, result, duration_ms)
            
            # Forward stdout and stderr
            if result.stdout:
                print(result.stdout, end='')
            if result.stderr:
                print(result.stderr, end='', file=sys.stderr)
            
            return result.returncode
            
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            self._collect_training_data(args, None, duration_ms, str(e))
            print(f"Error running rustc: {e}", file=sys.stderr)
            return 1
    
    def _find_rustc(self) -> str:
        """Find the real rustc binary"""
        # Try to find rustc in PATH
        try:
            result = subprocess.run(["which", "rustc"], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        
        # Fallback to common locations
        common_paths = [
            "/usr/bin/rustc",
            "/usr/local/bin/rustc",
            "/opt/homebrew/bin/rustc",
            "/Users/honedbeat/.cargo/bin/rustc"
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def _collect_training_data(self, args: List[str], result: subprocess.CompletedProcess, duration_ms: int, error: str = None):
        """Collect training data from rustc invocation"""
        timestamp = int(time.time())
        
        # Extract key information from args
        input_file = None
        output_file = None
        crate_type = None
        opt_level = "0"
        
        for i, arg in enumerate(args):
            if arg.endswith('.rs'):
                input_file = arg
            elif arg == '-o' and i + 1 < len(args):
                output_file = args[i + 1]
            elif arg == '--crate-type' and i + 1 < len(args):
                crate_type = args[i + 1]
            elif arg == '-C' and i + 1 < len(args) and args[i + 1].startswith('opt-level='):
                opt_level = args[i + 1].split('=')[1]
        
        # Create training data
        training_data = {
            "timestamp": timestamp,
            "duration_ms": duration_ms,
            "success": result.returncode == 0 if result else False,
            "input_file": input_file,
            "output_file": output_file,
            "crate_type": crate_type,
            "opt_level": opt_level,
            "args": args,
            "stdout": result.stdout if result else "",
            "stderr": result.stderr if result else "",
            "error": error,
            "returncode": result.returncode if result else 1
        }
        
        # Save training data
        filename = f"rustc_{timestamp}_{duration_ms}ms.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        # Also append to a log file for easy analysis
        log_file = self.output_dir / "rustc_training.log"
        with open(log_file, 'a') as f:
            f.write(f"{timestamp},{duration_ms},{training_data['success']},{input_file},{crate_type},{opt_level}\n")

def main():
    """Main entry point for RUSTC_WRAPPER"""
    wrapper = RustcTrainerWrapper()
    
    # RUSTC_WRAPPER passes the script name as first arg, then rustc args
    # So we need to skip the first argument (script name)
    args = sys.argv[1:]
    
    # Run rustc with training data collection
    exit_code = wrapper.run_rustc(args)
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
