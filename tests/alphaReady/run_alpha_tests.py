#!/usr/bin/env python
"""
Alpha Release Test Runner
========================

Comprehensive test runner for validating alpha release readiness.
Runs all critical tests and generates a readiness report.

Usage:
    python tests/run_alpha_tests.py [--with-real-integration] [--performance] [--report-file PATH]
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path
from datetime import datetime
import argparse


class AlphaTestRunner:
    """Manages alpha readiness testing."""
    
    def __init__(self, with_real_integration=False, with_performance=False, report_file=None):
        self.with_real_integration = with_real_integration
        self.with_performance = with_performance
        self.report_file = report_file or "alpha_test_report.json"
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "test_suites": {},
            "summary": {},
            "readiness_status": "UNKNOWN"
        }
    
    def run_test_suite(self, name, test_pattern, extra_args=None):
        """Run a test suite and capture results."""
        print(f"\n{'=' * 60}")
        print(f"Running {name}")
        print(f"{'=' * 60}")
        
        cmd = ["python", "-m", "pytest", test_pattern, "-v", "--tb=short"]
        if extra_args:
            cmd.extend(extra_args)
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=os.path.dirname(os.path.dirname(__file__))  # Run from project root
            )
            
            duration = time.time() - start_time
            
            # Parse pytest output
            output_lines = result.stdout.split('\n')
            summary_line = None
            for line in reversed(output_lines):
                if 'passed' in line and ('failed' in line or 'error' in line or 'skipped' in line):
                    summary_line = line.strip()
                    break
                elif line.strip().endswith('passed'):
                    summary_line = line.strip()
                    break
            
            self.results["test_suites"][name] = {
                "status": "PASSED" if result.returncode == 0 else "FAILED",
                "duration": round(duration, 2),
                "return_code": result.returncode,
                "summary": summary_line or "No summary available",
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
            if result.returncode == 0:
                print(f"✅ {name}: PASSED ({duration:.2f}s)")
            else:
                print(f"❌ {name}: FAILED ({duration:.2f}s)")
                print(f"   Error: {result.stderr[:200]}...")
            
        except Exception as e:
            duration = time.time() - start_time
            self.results["test_suites"][name] = {
                "status": "ERROR",
                "duration": round(duration, 2),
                "error": str(e),
                "summary": f"Test execution error: {e}"
            }
            print(f"💥 {name}: ERROR - {e}")
    
    def run_critical_tests(self):
        """Run all critical alpha readiness tests."""
        print("🚀 Starting Alpha Readiness Validation")
        print(f"Timestamp: {self.results['timestamp']}")
        
        # Core functionality tests
        self.run_test_suite(
            "Basic Crew Tests",
            "tests/test_basic_crew.py"
        )
        
        self.run_test_suite(
            "User Input Validation",
            "tests/test_user_input_validation.py"
        )
        
        self.run_test_suite(
            "MCP Integration (Mocked)",
            "tests/test_mcp_integration.py"
        )
        
        self.run_test_suite(
            "Research Paper Parser",
            "tests/test_research_paper_parser.py"
        )
        
        self.run_test_suite(
            "LLM Connection",
            "tests/test_llm_connection.py"
        )
        
        # Critical alpha readiness tests
        self.run_test_suite(
            "Alpha Readiness Tests",
            "tests/test_alpha_readiness.py"
        )
        
        # Real integration tests (if requested)
        if self.with_real_integration:
            self.run_test_suite(
                "Real MCP Integration",
                "tests/test_real_integration.py",
                ["--real-integration"]
            )
        
        # Performance tests (if requested)
        if self.with_performance:
            self.run_test_suite(
                "Performance Tests",
                "tests/test_alpha_readiness.py::TestAlphaPerformance"
            )
    
    def generate_readiness_assessment(self):
        """Generate alpha readiness assessment."""
        total_suites = len(self.results["test_suites"])
        passed_suites = sum(1 for suite in self.results["test_suites"].values() if suite["status"] == "PASSED")
        failed_suites = sum(1 for suite in self.results["test_suites"].values() if suite["status"] == "FAILED")
        error_suites = sum(1 for suite in self.results["test_suites"].values() if suite["status"] == "ERROR")
        
        # Critical test requirements for alpha
        critical_tests = [
            "Basic Crew Tests",
            "User Input Validation", 
            "Alpha Readiness Tests"
        ]
        
        critical_passed = all(
            self.results["test_suites"].get(test, {}).get("status") == "PASSED"
            for test in critical_tests
        )
        
        # Calculate readiness percentage
        readiness_percentage = (passed_suites / total_suites) * 100 if total_suites > 0 else 0
        
        # Determine readiness status
        if critical_passed and readiness_percentage >= 90:
            readiness_status = "READY_FOR_ALPHA"
        elif critical_passed and readiness_percentage >= 75:
            readiness_status = "MOSTLY_READY"
        elif critical_passed:
            readiness_status = "CRITICAL_TESTS_PASS"
        else:
            readiness_status = "NOT_READY"
        
        self.results["summary"] = {
            "total_test_suites": total_suites,
            "passed_suites": passed_suites,
            "failed_suites": failed_suites,
            "error_suites": error_suites,
            "readiness_percentage": round(readiness_percentage, 1),
            "critical_tests_passed": critical_passed,
            "critical_tests": critical_tests
        }
        
        self.results["readiness_status"] = readiness_status
        
        return readiness_status
    
    def print_summary_report(self):
        """Print summary report to console."""
        print(f"\n{'=' * 80}")
        print("ALPHA READINESS SUMMARY REPORT")
        print(f"{'=' * 80}")
        
        summary = self.results["summary"]
        
        print(f"📊 Test Results Overview:")
        print(f"   Total Test Suites: {summary['total_test_suites']}")
        print(f"   ✅ Passed: {summary['passed_suites']}")
        print(f"   ❌ Failed: {summary['failed_suites']}")
        print(f"   💥 Errors: {summary['error_suites']}")
        print(f"   📈 Success Rate: {summary['readiness_percentage']}%")
        
        print(f"\n🎯 Critical Tests Status:")
        for test in summary["critical_tests"]:
            status = self.results["test_suites"].get(test, {}).get("status", "NOT_RUN")
            icon = "✅" if status == "PASSED" else "❌" if status == "FAILED" else "💥"
            print(f"   {icon} {test}: {status}")
        
        print(f"\n🚀 Alpha Readiness Status: {self.results['readiness_status']}")
        
        # Detailed recommendations
        status = self.results['readiness_status']
        if status == "READY_FOR_ALPHA":
            print("🎉 RECOMMENDATION: Application is ready for alpha release!")
            print("   • All critical tests are passing")
            print("   • High success rate across all test suites")
            print("   • Core functionality is validated")
        
        elif status == "MOSTLY_READY":
            print("⚠️  RECOMMENDATION: Application is mostly ready, minor issues to resolve")
            print("   • Critical tests are passing")
            print("   • Some non-critical tests need attention")
            print("   • Consider alpha release with known limitations")
        
        elif status == "CRITICAL_TESTS_PASS":
            print("🔄 RECOMMENDATION: Core functionality works, but needs improvement")
            print("   • Critical tests are passing")
            print("   • Several test suites need attention")
            print("   • Address failing tests before alpha release")
        
        else:
            print("🚫 RECOMMENDATION: NOT ready for alpha release")
            print("   • Critical tests are failing")
            print("   • Core functionality issues must be resolved")
            print("   • Focus on fixing basic functionality first")
        
        print(f"\n📄 Detailed report saved to: {self.report_file}")
    
    def save_detailed_report(self):
        """Save detailed JSON report."""
        with open(self.report_file, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def run_full_assessment(self):
        """Run complete alpha readiness assessment."""
        try:
            self.run_critical_tests()
            readiness_status = self.generate_readiness_assessment()
            self.print_summary_report()
            self.save_detailed_report()
            
            # Return exit code based on readiness
            if readiness_status in ["READY_FOR_ALPHA", "MOSTLY_READY"]:
                return 0
            else:
                return 1
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Test run interrupted by user")
            return 130
        
        except Exception as e:
            print(f"\n\n💥 Test runner error: {e}")
            return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Alpha Release Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/run_alpha_tests.py                    # Basic alpha tests
  python tests/run_alpha_tests.py --with-real       # Include real MCP integration
  python tests/run_alpha_tests.py --with-performance # Include performance tests
  python tests/run_alpha_tests.py --report-file custom_report.json
        """
    )
    
    parser.add_argument(
        "--with-real-integration",
        action="store_true",
        help="Include real MCP server integration tests (requires servers)"
    )
    
    parser.add_argument(
        "--with-performance",
        action="store_true", 
        help="Include performance tests"
    )
    
    parser.add_argument(
        "--report-file",
        type=str,
        default="alpha_test_report.json",
        help="Path for detailed JSON report (default: alpha_test_report.json)"
    )
    
    args = parser.parse_args()
    
    # Verify we're in the right directory
    if not os.path.exists("pyproject.toml"):
        print("❌ Error: Must run from project root directory")
        print("   Current directory:", os.getcwd())
        print("   Expected to find: pyproject.toml")
        return 1
    
    # Create test runner
    runner = AlphaTestRunner(
        with_real_integration=args.with_real_integration,
        with_performance=args.with_performance,
        report_file=args.report_file
    )
    
    # Run assessment
    return runner.run_full_assessment()


if __name__ == "__main__":
    sys.exit(main()) 