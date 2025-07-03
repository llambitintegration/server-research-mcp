#!/usr/bin/env python3
"""
CI-safe test runner that excludes external dependencies.
Run with: python tests/run_ci_tests.py
"""

import subprocess
import sys
import os

def main():
    """Run tests excluding external dependencies."""
    # Change to project root
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Test command that excludes external dependencies
    cmd = [
        "python", "-m", "pytest",
        "-m", "not real_servers and not mcp_live",
        "--tb=short",
        "-v",
        "tests/unit/",
        "tests/integration/",
        "--ignore=tests/integration/test_single_agent_mcp_live.py"  # Skip live MCP tests
    ]
    
    print("🧪 Running CI-safe tests (excluding external dependencies)...")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("✅ All CI-safe tests passed!")
    else:
        print("❌ Some tests failed")
        
    return result.returncode

if __name__ == "__main__":
    sys.exit(main()) 
