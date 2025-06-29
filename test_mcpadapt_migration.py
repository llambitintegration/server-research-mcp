#!/usr/bin/env python3
"""
Quick test to validate MCPAdapt migration
"""

def test_basic_imports():
    """Test that all imports work correctly."""
    print("🧪 Testing basic imports...")
    
    try:
        from src.server_research_mcp.tools.mcp_tools import (
            get_mcp_server_configs, 
            get_historian_tools,
            get_researcher_tools,
            get_archivist_tools,
            get_publisher_tools
        )
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_server_configs():
    """Test server configuration loading."""
    print("\n🔧 Testing server configurations...")
    
    try:
        from src.server_research_mcp.tools.mcp_tools import get_mcp_server_configs
        configs = get_mcp_server_configs()
        print(f"✅ Loaded {len(configs)} server configurations")
        
        for i, config in enumerate(configs):
            print(f"  {i+1}. Command: {config.command}, Args: {config.args[:2]}...")
        
        return True
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False

def test_tool_loading():
    """Test tool loading (without actual MCP connection)."""
    print("\n🛠️ Testing tool loading functions...")
    
    tool_functions = [
        ('Historian', 'get_historian_tools'),
        ('Researcher', 'get_researcher_tools'), 
        ('Archivist', 'get_archivist_tools'),
        ('Publisher', 'get_publisher_tools')
    ]
    
    results = []
    
    for agent_name, func_name in tool_functions:
        try:
            from src.server_research_mcp.tools import mcp_tools
            func = getattr(mcp_tools, func_name)
            tools = func()
            print(f"✅ {agent_name} tools: {len(tools)} loaded")
            results.append(True)
        except Exception as e:
            print(f"❌ {agent_name} tools failed: {e}")
            results.append(False)
    
    return all(results)

def test_crew_integration():
    """Test crew integration with new system."""
    print("\n🚢 Testing crew integration...")
    
    try:
        from src.server_research_mcp.crew import get_crew_mcp_manager
        
        manager = get_crew_mcp_manager()
        print("✅ Crew MCP manager created")
        
        # Test tool access
        hist_tools = manager.get_historian_tools()
        print(f"✅ Historian tools via crew manager: {len(hist_tools)}")
        
        return True
    except Exception as e:
        print(f"❌ Crew integration failed: {e}")
        return False

def main():
    """Run all migration tests."""
    print("🚀 MCPAdapt Migration Validation")
    print("=" * 40)
    
    tests = [
        test_basic_imports,
        test_server_configs,
        test_tool_loading,
        test_crew_integration
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n📊 Results Summary:")
    print("=" * 40)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 All {total} tests PASSED! MCPAdapt migration successful.")
        return 0
    else:
        print(f"⚠️ {passed}/{total} tests passed. {total-passed} failures detected.")
        return 1

if __name__ == "__main__":
    exit(main())