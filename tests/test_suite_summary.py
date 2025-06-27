#!/usr/bin/env python3
"""Generate a summary of the test suite status."""

import os
import ast
import json
from datetime import datetime
from pathlib import Path


def count_test_functions(file_path):
    """Count test functions in a Python file."""
    try:
        with open(file_path, 'r') as f:
            tree = ast.parse(f.read())
        
        test_count = 0
        test_classes = 0
        test_functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name.startswith('Test'):
                test_classes += 1
            elif isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                test_count += 1
                test_functions.append(node.name)
        
        return test_count, test_classes, test_functions
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return 0, 0, []


def analyze_test_suite():
    """Analyze the entire test suite."""
    test_dir = Path(__file__).parent
    test_files = list(test_dir.glob("test_*.py"))
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "test_files": {},
        "totals": {
            "files": 0,
            "classes": 0,
            "tests": 0
        },
        "categories": {
            "core": [],
            "agents": [],
            "mcp": [],
            "end_to_end": [],
            "knowledge": [],
            "performance": [],
            "basic": [],
            "integration": []
        }
    }
    
    for test_file in test_files:
        if test_file.name == "test_run_all.py" or test_file.name == "test_suite_summary.py":
            continue
            
        test_count, class_count, functions = count_test_functions(test_file)
        
        file_info = {
            "path": str(test_file.relative_to(test_dir)),
            "classes": class_count,
            "tests": test_count,
            "functions": functions[:5]  # First 5 function names
        }
        
        summary["test_files"][test_file.stem] = file_info
        summary["totals"]["files"] += 1
        summary["totals"]["classes"] += class_count
        summary["totals"]["tests"] += test_count
        
        # Categorize
        if "core" in test_file.name:
            summary["categories"]["core"].append(test_file.stem)
        elif "agents" in test_file.name:
            summary["categories"]["agents"].append(test_file.stem)
        elif "mcp" in test_file.name:
            summary["categories"]["mcp"].append(test_file.stem)
        elif "end_to_end" in test_file.name:
            summary["categories"]["end_to_end"].append(test_file.stem)
        elif "knowledge" in test_file.name:
            summary["categories"]["knowledge"].append(test_file.stem)
        elif "performance" in test_file.name:
            summary["categories"]["performance"].append(test_file.stem)
        elif "basic" in test_file.name:
            summary["categories"]["basic"].append(test_file.stem)
        else:
            summary["categories"]["integration"].append(test_file.stem)
    
    return summary


def print_summary(summary):
    """Print a formatted summary."""
    print("\n" + "=" * 60)
    print("TEST SUITE SUMMARY")
    print("=" * 60)
    print(f"Generated: {summary['timestamp']}")
    print(f"\nTotal Files: {summary['totals']['files']}")
    print(f"Total Test Classes: {summary['totals']['classes']}")
    print(f"Total Test Functions: {summary['totals']['tests']}")
    
    print("\n" + "-" * 60)
    print("TEST FILES:")
    print("-" * 60)
    
    for file_name, info in summary["test_files"].items():
        print(f"\n{file_name}:")
        print(f"  Classes: {info['classes']}")
        print(f"  Tests: {info['tests']}")
        if info['functions']:
            print(f"  Sample functions:")
            for func in info['functions']:
                print(f"    - {func}")
    
    print("\n" + "-" * 60)
    print("CATEGORIES:")
    print("-" * 60)
    
    for category, files in summary["categories"].items():
        if files:
            print(f"\n{category.upper()}:")
            for file in files:
                print(f"  - {file}")
    
    print("\n" + "=" * 60)


def save_summary(summary):
    """Save summary to JSON file."""
    output_file = Path(__file__).parent / "test_suite_summary.json"
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {output_file}")


if __name__ == "__main__":
    print("Analyzing test suite...")
    summary = analyze_test_suite()
    print_summary(summary)
    save_summary(summary)
    
    # Quick status
    total_tests = summary["totals"]["tests"]
    if total_tests > 100:
        print("\n✅ Comprehensive test suite with excellent coverage!")
    elif total_tests > 50:
        print("\n✅ Good test coverage, room for expansion.")
    else:
        print("\n⚠️  Basic test coverage, consider adding more tests.")
    
    print(f"\n📊 Total test count: {total_tests}")
    print("🚀 Ready for production-grade testing!")