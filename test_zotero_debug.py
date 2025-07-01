#!/usr/bin/env python3
"""Test Zotero tools with debug logging enabled."""

import os
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

def test_direct_zotero_call():
    """Test direct Zotero tool call with debug logging."""
    print("🔍 TESTING DIRECT ZOTERO CALL WITH DEBUG LOGGING")
    print("=" * 55)
    
    try:
        from src.server_research_mcp.tools.mcp_tools import get_researcher_tools
        
        # Get tools
        tools = get_researcher_tools()
        search_tool = None
        
        for tool in tools:
            if 'search_items' in tool.name:
                search_tool = tool
                break
        
        if not search_tool:
            print("❌ Zotero search tool not found")
            return
        
        print(f"✅ Found search tool: {search_tool.name}")
        print(f"Tool type: {type(search_tool)}")
        
        # Test 1: Direct call with string (simulates CrewAI)
        print("\n🧪 Test 1: Direct call with JSON string")
        test_input = '{"query": "machine learning", "qmode": "everything", "limit": 3}'
        print(f"Input: {test_input}")
        
        try:
            result = search_tool._run(test_input)
            print(f"✅ Success! Result length: {len(str(result))}")
            print(f"Result preview: {str(result)[:200]}...")
        except Exception as e:
            print(f"❌ Failed: {e}")
        
        # Test 2: Direct call with kwargs
        print("\n🧪 Test 2: Direct call with kwargs")
        try:
            result = search_tool._run(query="artificial intelligence", limit=2)
            print(f"✅ Success! Result length: {len(str(result))}")
            print(f"Result preview: {str(result)[:200]}...")
        except Exception as e:
            print(f"❌ Failed: {e}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_direct_zotero_call() 