#!/usr/bin/env python3
"""Convenient test runner for different test scenarios."""

import subprocess
import sys
import argparse


def run_command(cmd):
    """Run a command and return its exit code."""
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print('='*60)
    return subprocess.run(cmd).returncode


def main():
    parser = argparse.ArgumentParser(description='Run server-research-mcp tests')
    parser.add_argument('--unit', action='store_true', help='Run unit tests only')
    parser.add_argument('--integration', action='store_true', help='Run integration tests only')
    parser.add_argument('--e2e', action='store_true', help='Run end-to-end tests only')
    parser.add_argument('--fast', action='store_true', help='Skip slow tests')
    parser.add_argument('--no-external', action='store_true', help='Skip tests requiring external services')
    parser.add_argument('--coverage', action='store_true', help='Run with coverage report')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Base pytest command
    cmd = ['pytest']
    
    # Add verbosity
    if args.verbose:
        cmd.append('-vv')
    
    # Add coverage
    if args.coverage:
        cmd.extend(['--cov=src', '--cov-report=term-missing', '--cov-report=html'])
    
    # Handle test selection
    if args.unit:
        cmd.append('tests/unit/')
        print("\n🧪 Running UNIT tests...")
    elif args.integration:
        cmd.append('tests/integration/')
        print("\n🔧 Running INTEGRATION tests...")
    elif args.e2e:
        cmd.append('tests/e2e/')
        print("\n🚀 Running END-TO-END tests...")
    else:
        print("\n📋 Running ALL tests...")
    
    # Handle markers
    markers = []
    if args.fast:
        markers.append('not slow')
    if args.no_external:
        markers.extend(['not requires_llm', 'not requires_mcp'])
    
    if markers:
        cmd.extend(['-m', ' and '.join(markers)])
    
    # Run the tests
    exit_code = run_command(cmd)
    
    # Print summary
    print(f"\n{'='*60}")
    if exit_code == 0:
        print("✅ Tests PASSED!")
    else:
        print("❌ Tests FAILED!")
    print('='*60)
    
    # Print coverage report location if generated
    if args.coverage and exit_code == 0:
        print("\n📊 Coverage report generated:")
        print("   - Terminal output above")
        print("   - HTML report: htmlcov/index.html")
    
    return exit_code


if __name__ == '__main__':
    sys.exit(main())