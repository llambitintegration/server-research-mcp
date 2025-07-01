#!/usr/bin/env python3
"""Test Zotero MCP integration comprehensively."""

import asyncio
import os
import sys
import logging
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

async def test_zotero_mcp_startup():
    """Test Zotero MCP server startup."""
    try:
        from src.server_research_mcp.utils.mcpadapt import MCPAdapt, CrewAIAdapter
        from mcp import StdioServerParameters
        
        print("🔧 TESTING ZOTERO MCP SERVER STARTUP")
        print("=" * 45)
        
        # Get credentials
        api_key = os.getenv("ZOTERO_API_KEY")
        library_id = os.getenv("ZOTERO_LIBRARY_ID")
        
        if not api_key or not library_id:
            print("❌ Missing Zotero credentials")
            return
        
        print(f"✅ Credentials found - API key: {len(api_key)} chars, Library ID: {library_id}")
        
        # Configure Zotero server
        zotero_config = StdioServerParameters(
            command="uvx",
            args=["zotero-mcp"],
            env={
                "ZOTERO_LOCAL": "false",
                "ZOTERO_API_KEY": api_key,
                "ZOTERO_LIBRARY_ID": library_id
            }
        )
        
        print("🚀 Starting Zotero MCP server...")
        
        # Test server startup
        try:
            with MCPAdapt([zotero_config], CrewAIAdapter()) as tools:
                print(f"✅ Server started successfully!")
                print(f"📊 Total tools loaded: {len(tools)}")
                
                if tools:
                    print("\n🔧 Available tools:")
                    for i, tool in enumerate(tools):
                        print(f"  {i+1}. {tool.name}")
                        if hasattr(tool, 'description'):
                            print(f"     Description: {tool.description}")
                    
                    # Test tool execution
                    print("\n⚙️ Testing tool execution...")
                    zotero_tools = [tool for tool in tools if 'zotero' in tool.name.lower()]
                    
                    if zotero_tools:
                        test_tool = zotero_tools[0]
                        print(f"🧪 Testing tool: {test_tool.name}")
                        
                        try:
                            # Try a simple search
                            result = test_tool._run("test search")
                            print(f"✅ Tool execution successful!")
                            print(f"📋 Result preview: {str(result)[:200]}...")
                        except Exception as e:
                            print(f"❌ Tool execution failed: {e}")
                            print(f"   Error type: {type(e)}")
                    else:
                        print("⚠️ No Zotero-specific tools found")
                else:
                    print("❌ No tools loaded from server")
                    
        except Exception as e:
            print(f"❌ Server startup failed: {e}")
            print(f"   Error type: {type(e)}")
            import traceback
            traceback.print_exc()
    
    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

def test_direct_researcher_tools():
    """Test researcher tools directly."""
    try:
        print("\n🔬 TESTING RESEARCHER TOOLS")
        print("=" * 35)
        
        from src.server_research_mcp.tools.mcp_tools import get_researcher_tools
        
        tools = get_researcher_tools()
        print(f"📊 Total researcher tools: {len(tools)}")
        
        if tools:
            print("\n🔧 Researcher tool details:")
            for i, tool in enumerate(tools):
                print(f"  {i+1}. {tool.name}")
                if hasattr(tool, 'description'):
                    print(f"     Description: {tool.description}")
                
                # Check if it's a Zotero tool
                if 'zotero' in tool.name.lower():
                    print("     🎯 This is a Zotero tool!")
        else:
            print("❌ No researcher tools found")
            
    except Exception as e:
        print(f"❌ Error testing researcher tools: {e}")
        import traceback
        traceback.print_exc()

def test_basic_mcp_tools():
    """Test basic MCP tools setup."""
    try:
        print("\n🏗️ TESTING BASIC MCP TOOLS SETUP")
        print("=" * 40)
        
        from src.server_research_mcp.tools.mcp_tools import get_all_mcp_tools
        
        all_tools = get_all_mcp_tools()
        print(f"📊 Tool categories: {list(all_tools.keys())}")
        
        for category, tools in all_tools.items():
            print(f"\n{category.upper()}: {len(tools)} tools")
            for tool in tools[:3]:  # Show first 3 tools
                print(f"  • {tool.name}")
        
    except Exception as e:
        print(f"❌ Error testing basic MCP tools: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Run all tests."""
    print("🧪 COMPREHENSIVE ZOTERO MCP DIAGNOSTIC")
    print("=" * 50)
    
    # Test 1: Basic MCP tools
    test_basic_mcp_tools()
    
    # Test 2: Researcher tools
    test_direct_researcher_tools()
    
    # Test 3: Zotero MCP server startup
    await test_zotero_mcp_startup()
    
    print("\n✅ Diagnostic complete!")

if __name__ == "__main__":
    asyncio.run(main()) 