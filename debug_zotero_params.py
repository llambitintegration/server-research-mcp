#!/usr/bin/env python3
"""Debug Zotero parameter passing through MCPToolWrapper."""

import json
import os

def simulate_crewai_call():
    """Simulate how CrewAI calls the Zotero search tool."""
    
    # This is what CrewAI passes to the tool
    crewai_input = '{"query": "machine learning", "qmode": "everything", "limit": 5}'
    
    print("🧪 DEBUGGING ZOTERO PARAMETER PASSING")
    print("=" * 50)
    print(f"CrewAI input: {crewai_input}")
    
    # Simulate the MCPToolWrapper parameter processing
    args = (crewai_input,)
    kwargs = {}
    
    print(f"Initial - args: {args}, kwargs: {kwargs}")
    
    # Step 1: CrewAI sometimes wraps kwargs in a single "properties" dict
    if len(kwargs) == 1 and 'properties' in kwargs:
        kwargs = kwargs['properties']
        print(f"After properties unwrap - args: {args}, kwargs: {kwargs}")

    # Step 2: CrewAI often passes **all** parameters as a *single* JSON string
    if len(args) == 1 and isinstance(args[0], str):
        arg_str = args[0]
        print(f"Processing JSON string: {arg_str}")
        
        try:
            # Some CrewAI versions double-encode JSON
            for i in range(2):  # at most two decoding passes
                if isinstance(arg_str, str):
                    try:
                        loaded = json.loads(arg_str)
                        print(f"JSON decode pass {i+1}: {loaded}")
                    except json.JSONDecodeError:
                        loaded = arg_str
                        print(f"Not JSON at pass {i+1}: {loaded}")
                        break
                    arg_str = loaded
                else:
                    loaded = arg_str
                    print(f"Final loaded: {loaded}")
                    break

            # After potential double-decoding we have `loaded`
            if isinstance(loaded, dict):
                print(f"Loaded is dict: {loaded}")
                if 'query' in loaded and len(loaded) == 1:
                    kwargs.setdefault('query', loaded['query'])
                    args = tuple()
                    print(f"Single query conversion - args: {args}, kwargs: {kwargs}")
                else:
                    for k, v in loaded.items():
                        kwargs.setdefault(k, v)
                    args = tuple()
                    print(f"Multi-param conversion - args: {args}, kwargs: {kwargs}")
            else:
                args = (loaded,)
                print(f"Loaded not dict - args: {args}, kwargs: {kwargs}")
        except json.JSONDecodeError:
            print("JSON decode failed")
            pass

    print(f"After JSON processing - args: {args}, kwargs: {kwargs}")

    # CRITICAL FIX check
    tool_name = "zotero_search_items"
    if 'search' in tool_name.lower() and kwargs:
        print(f"Zotero search tool detected: {tool_name}, keeping kwargs format")
        # Don't convert to positional for search tools - they need kwargs
    elif not args and len(kwargs) == 1 and 'query' in kwargs:
        print(f"Converting single query kwarg to positional: {kwargs['query']}")
        args = (kwargs['query'],)
        kwargs = {}
        print(f"After conversion: args={args}, kwargs={kwargs}")

    print(f"Final parameters - args: {args}, kwargs: {kwargs}")
    
    # Validation check
    if 'search' in tool_name.lower() and not args and not kwargs:
        print("❌ VALIDATION FAILED: No parameters provided")
        return False
    else:
        print("✅ VALIDATION PASSED: Parameters available")
        return True

if __name__ == "__main__":
    success = simulate_crewai_call()
    if success:
        print("\n🎉 Parameter passing should work!")
    else:
        print("\n💥 Parameter passing will fail!") 