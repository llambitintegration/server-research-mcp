"""Test runner for comprehensive pytest suite."""

import pytest
import sys
import os
from datetime import datetime
import json


def run_test_suite(test_type=None, markers=None, verbose=True):
    """Run the comprehensive test suite with various options."""
    
    # Configure pytest arguments
    args = []
    
    # Add verbosity
    if verbose:
        args.append("-v")
    
    # Add coverage if available
    try:
        import pytest_cov
        args.extend(["--cov=server_research_mcp", "--cov-report=term-missing"])
    except ImportError:
        print("pytest-cov not installed, skipping coverage")
    
    # Add specific test type
    if test_type:
        test_mapping = {
            "core": "test_crew_core.py",
            "agents": "test_agents.py", 
            "mcp": "test_mcp_integration.py",
            "e2e": "test_end_to_end.py",
            "knowledge": "test_knowledge_management.py",
            "performance": "test_performance.py",
            "basic": "test_basic_crew.py",
            "historian": "test_historian_mcp.py"
        }
        
        if test_type in test_mapping:
            args.append(test_mapping[test_type])
        else:
            print(f"Unknown test type: {test_type}")
            return 1
    
    # Add markers
    if markers:
        for marker in markers:
            args.extend(["-m", marker])
    
    # Add output options
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSON report
    args.extend(["--json-report", f"--json-report-file=test_results_{timestamp}.json"])
    
    # HTML report if available
    try:
        import pytest_html
        args.extend(["--html", f"test_report_{timestamp}.html", "--self-contained-html"])
    except ImportError:
        print("pytest-html not installed, skipping HTML report")
    
    # Run tests
    print(f"Running tests with args: {' '.join(args)}")
    return pytest.main(args)


def run_quick_tests():
    """Run quick tests for rapid feedback."""
    print("\n=== Running Quick Tests ===")
    return run_test_suite(markers=["not slow", "not integration"])


def run_unit_tests():
    """Run only unit tests."""
    print("\n=== Running Unit Tests ===")
    return run_test_suite(markers=["not integration", "not performance"])


def run_integration_tests():
    """Run integration tests."""
    print("\n=== Running Integration Tests ===")
    return run_test_suite(markers=["integration"])


def run_performance_tests():
    """Run performance tests."""
    print("\n=== Running Performance Tests ===")
    return run_test_suite(markers=["performance"])


def run_mcp_tests():
    """Run MCP-specific tests."""
    print("\n=== Running MCP Tests ===")
    return run_test_suite(markers=["mcp"])


def run_async_tests():
    """Run async tests."""
    print("\n=== Running Async Tests ===")
    return run_test_suite(markers=["async"])


def generate_test_report():
    """Generate a comprehensive test report."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "test_categories": {
            "core": "Core crew functionality",
            "agents": "Agent-specific tests",
            "mcp": "MCP integration tests",
            "e2e": "End-to-end workflow tests",
            "knowledge": "Knowledge management tests",
            "performance": "Performance and scalability tests"
        },
        "markers": {
            "integration": "Tests requiring external services",
            "performance": "Performance benchmarking tests",
            "slow": "Long-running tests",
            "mcp": "MCP server tests",
            "async": "Asynchronous tests"
        }
    }
    
    # Count tests
    test_counts = {}
    for category in results["test_categories"]:
        try:
            # This would actually count tests in each file
            test_counts[category] = "Available"
        except:
            test_counts[category] = "Not found"
    
    results["test_counts"] = test_counts
    
    # Save report
    with open("test_suite_report.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nTest suite report generated: test_suite_report.json")


def main():
    """Main test runner entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run comprehensive test suite")
    parser.add_argument("--type", choices=["all", "quick", "unit", "integration", 
                                          "performance", "mcp", "async", "core",
                                          "agents", "e2e", "knowledge"],
                       default="quick", help="Type of tests to run")
    parser.add_argument("--markers", nargs="+", help="Pytest markers to include")
    parser.add_argument("--report", action="store_true", help="Generate test report")
    parser.add_argument("--verbose", action="store_true", default=True, 
                       help="Verbose output")
    
    args = parser.parse_args()
    
    if args.report:
        generate_test_report()
        return 0
    
    # Run appropriate test suite
    if args.type == "all":
        return run_test_suite(markers=args.markers, verbose=args.verbose)
    elif args.type == "quick":
        return run_quick_tests()
    elif args.type == "unit":
        return run_unit_tests()
    elif args.type == "integration":
        return run_integration_tests()
    elif args.type == "performance":
        return run_performance_tests()
    elif args.type == "mcp":
        return run_mcp_tests()
    elif args.type == "async":
        return run_async_tests()
    else:
        return run_test_suite(test_type=args.type, markers=args.markers, 
                            verbose=args.verbose)


if __name__ == "__main__":
    sys.exit(main())