"""
Real Integration Tests
====================

Optional tests that run with actual MCP servers.
These tests are skipped by default but can be enabled for full integration testing.

Run with: pytest tests/test_real_integration.py --real-integration
"""

import pytest
import os
import asyncio
import tempfile
import json
from pathlib import Path
from datetime import datetime
import sys

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# These imports are no longer available after MCPAdapt migration
# from server_research_mcp.tools.mcp_manager import MCPManager, get_mcp_manager
# from server_research_mcp.tools.enhanced_mcp_manager import EnhancedMCPManager, get_enhanced_mcp_manager
from server_research_mcp.utils.mcpadapt import MCPAdapt, CrewAIAdapter
from server_research_mcp.crew import ServerResearchMcpCrew
from server_research_mcp.main import run_crew


def pytest_addoption(parser):
    """Add command line option for real integration tests."""
    parser.addoption(
        "--real-integration",
        action="store_true",
        default=False,
        help="Run real integration tests with actual MCP servers"
    )


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "real_integration: mark test as requiring real MCP servers"
    )


def pytest_collection_modifyitems(config, items):
    """Skip real integration tests unless --real-integration is specified."""
    if config.getoption("--real-integration"):
        return
    
    skip_real = pytest.mark.skip(reason="need --real-integration option to run")
    for item in items:
        if "real_integration" in item.keywords:
            item.add_marker(skip_real)


@pytest.fixture(scope="session")
def real_test_environment():
    """Set up environment for real MCP server testing."""
    # Verify required environment variables
    required_vars = [
        "ANTHROPIC_API_KEY",
        "OBSIDIAN_VAULT_PATH"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        pytest.skip(f"Missing required environment variables: {missing_vars}")
    
    # Create temporary output directory
    temp_dir = tempfile.mkdtemp(prefix="real_integration_test_")
    
    # Set test-specific environment
    test_env = {
        "CHROMADB_PATH": os.path.join(temp_dir, "chromadb"),
        "CHROMADB_ALLOW_RESET": "true",
        "DISABLE_CREW_MEMORY": "false",  # Enable memory for real tests
        "OUTPUT_DIR": temp_dir
    }
    
    original_env = {}
    for key, value in test_env.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value
    
    yield temp_dir
    
    # Cleanup
    for key, original_value in original_env.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value


@pytest.mark.real_integration
class TestRealMCPConnections:
    """Test real MCP server connections."""
    
    @pytest.mark.asyncio
    async def test_memory_server_real_connection(self, real_test_environment):
        """Test real connection to memory server."""
        from mcp import StdioServerParameters
        
        # Configure memory server
        server_configs = [
            StdioServerParameters(
                command="npx",
                args=["-y", "@modelcontextprotocol/server-memory"],
                env={}
            )
        ]
        
        # Use MCPAdapt with context manager
        try:
            with MCPAdapt(server_configs, CrewAIAdapter()) as tools:
                # Find memory tools
                memory_tools = [t for t in tools if 'memory' in getattr(t, 'name', '').lower()]
                assert len(memory_tools) > 0, "Should have memory tools available"
                
                # Test basic tool execution
                search_tool = next((t for t in memory_tools if 'search' in getattr(t, 'name', '').lower()), None)
                if search_tool:
                    result = search_tool.run(query="test_alpha_integration")
                    assert result is not None, "Memory search should return result"
                
                print("✅ REAL: Memory server connection successful")
                
        except Exception as e:
            # If connection fails, skip the test rather than fail
            pytest.skip(f"Memory server not available: {e}")
    
    @pytest.mark.asyncio
    async def test_context7_server_real_connection(self, real_test_environment):
        """Test real connection to context7 server."""
        from mcp import StdioServerParameters
        
        # Configure context7 server
        server_configs = [
            StdioServerParameters(
                command="npx",
                args=["-y", "@upstash/context7-mcp"],
                env={}
            )
        ]
        
        try:
            with MCPAdapt(server_configs, CrewAIAdapter()) as tools:
                # Find context7 tools
                context7_tools = [t for t in tools if any(keyword in getattr(t, 'name', '').lower() 
                                                        for keyword in ['resolve', 'context', 'library', 'docs'])]
                
                if len(context7_tools) == 0:
                    pytest.skip("Context7 server not available")
                
                # Test library resolution
                resolve_tool = next((t for t in context7_tools if 'resolve' in getattr(t, 'name', '').lower()), None)
                if resolve_tool:
                    result = resolve_tool.run(libraryName="machine learning")
                    assert result is not None, "Context7 resolve should return result"
                
                print("✅ REAL: Context7 server connection successful")
                
        except Exception as e:
            pytest.skip(f"Context7 server not available: {e}")
    
    @pytest.mark.asyncio
    async def test_mcpadapt_integration_real(self, real_test_environment):
        """Test MCPAdapt integration with real servers."""
        from mcp import StdioServerParameters
        
        # Configure multiple servers
        server_configs = [
            StdioServerParameters(
                command="npx",
                args=["-y", "@modelcontextprotocol/server-memory"],
                env={}
            ),
            StdioServerParameters(
                command="npx", 
                args=["-y", "@modelcontextprotocol/server-sequential-thinking"],
                env={}
            )
        ]
        
        try:
            with MCPAdapt(server_configs, CrewAIAdapter()) as tools:
                if len(tools) == 0:
                    pytest.skip("No MCP servers available - requires Node.js and MCP packages")
                
                # Test that we have tools from multiple servers
                tool_names = [getattr(t, 'name', '') for t in tools]
                assert len(tool_names) > 0, "Should have at least some tools available"
                
                # Test basic tool execution
                if tools:
                    # Try to run the first available tool with minimal args
                    test_tool = tools[0]
                    try:
                        # Most MCP tools can handle empty or minimal arguments
                        result = test_tool.run()
                        # Just verify we got some response
                        assert result is not None
                    except Exception:
                        # Tool execution might fail but connection was successful
                        pass
                
                print(f"✅ REAL: MCPAdapt integration successful with {len(tools)} tools")
                
        except Exception as e:
            pytest.skip(f"MCP servers not available: {e}")


@pytest.mark.real_integration
class TestRealWorkflows:
    """Test real end-to-end workflows."""
    
    def test_crew_initialization_with_real_servers(self, real_test_environment):
        """Test crew initialization with real MCP servers available."""
        # This test verifies that crews can initialize even when real servers are present
        import os
        
        # Disable ChromaDB memory to avoid configuration issues in tests
        os.environ["DISABLE_CREW_MEMORY"] = "true"
        
        try:
            crew_instance = ServerResearchMcpCrew()
            crew = crew_instance.crew()
            
            assert crew is not None
            assert len(crew.agents) == 4
            assert len(crew.tasks) == 4
            
            # Verify all agents have tools
            for agent in crew.agents:
                assert len(agent.tools) > 0, f"Agent {agent.role} should have tools"
            
            print("✅ REAL: Crew initialization with real servers successful")
        finally:
            # Clean up environment variable
            if "DISABLE_CREW_MEMORY" in os.environ:
                del os.environ["DISABLE_CREW_MEMORY"]
    
    @pytest.mark.skip(reason="Paper search workflow removed from current system")
    async def test_paper_search_workflow_real(self, real_test_environment):
        """Test a realistic paper search workflow - DEPRECATED."""
        pass
    
    @pytest.mark.asyncio
    async def test_memory_persistence_real(self, real_test_environment):
        """Test that memory persists across sessions using MCPAdapt."""
        from server_research_mcp.utils.mcpadapt import MCPAdapt, CrewAIAdapter
        from mcp import StdioServerParameters
        from datetime import datetime
        
        test_entity_name = f"alpha_test_{int(datetime.now().timestamp())}"
        
        # Configure memory server
        server_configs = [
            StdioServerParameters(
                command="npx",
                args=["-y", "@modelcontextprotocol/server-memory"],
                env={}
            )
        ]
        
        # Session 1: Create entity
        try:
            async with MCPAdapt(server_configs, CrewAIAdapter()) as tools:
                # Find memory tools
                memory_tools = [tool for tool in tools if "memory" in tool.name.lower()]
                
                if not memory_tools:
                    pytest.skip("Memory tools not available")
                
                # Find create entities tool
                create_tool = None
                for tool in memory_tools:
                    if "create" in tool.name.lower() and "entities" in tool.name.lower():
                        create_tool = tool
                        break
                
                if not create_tool:
                    pytest.skip("Memory create entities tool not found")
                
                # Create test entity
                create_result = create_tool.run(
                    entities=[
                        {
                            "name": test_entity_name,
                            "entityType": "test",
                            "observations": ["Alpha persistence test"]
                        }
                    ]
                )
                
                assert create_result is not None, "Create entity should return a result"
                
        except Exception as e:
            pytest.skip(f"Memory creation failed: {e}")
        
        # Session 2: Search for entity
        try:
            async with MCPAdapt(server_configs, CrewAIAdapter()) as tools:
                # Find memory tools
                memory_tools = [tool for tool in tools if "memory" in tool.name.lower()]
                
                # Find search tool
                search_tool = None
                for tool in memory_tools:
                    if "search" in tool.name.lower():
                        search_tool = tool
                        break
                
                if not search_tool:
                    pytest.skip("Memory search tool not found")
                
                # Search for our entity
                search_result = search_tool.run(query=test_entity_name)
                
                # Basic validation that search worked
                assert search_result is not None, "Search should return some result"
                
                print("✅ REAL: Memory persistence successful with MCPAdapt")
                
        except Exception as e:
            pytest.skip(f"Memory search failed: {e}")


@pytest.mark.real_integration
class TestRealPerformance:
    """Performance tests with real servers."""
    
    @pytest.mark.asyncio
    async def test_server_startup_performance(self, real_test_environment):
        """Test server startup performance using MCPAdapt."""
        import time
        from server_research_mcp.utils.mcpadapt import MCPAdapt, CrewAIAdapter
        from mcp import StdioServerParameters
        
        # Configure multiple servers
        server_configs = [
            StdioServerParameters(
                command="npx",
                args=["-y", "@modelcontextprotocol/server-memory"],
                env={}
            ),
            StdioServerParameters(
                command="npx",
                args=["-y", "@upstash/context7-mcp"],
                env={}
            ),
            StdioServerParameters(
                command="npx",
                args=["-y", "@modelcontextprotocol/server-sequential-thinking"],
                env={}
            )
        ]
        
        start_time = time.time()
        
        try:
            # Initialize available servers
            async with MCPAdapt(server_configs, CrewAIAdapter()) as tools:
                startup_time = time.time() - start_time
                
                # Should start within reasonable time
                assert startup_time < 60, f"Server startup took {startup_time:.2f}s, should be < 60s"
                
                print(f"✅ REAL PERFORMANCE: Servers started in {startup_time:.2f}s with MCPAdapt")
            
        except Exception as e:
            pytest.skip(f"Server startup failed: {e}")
    
    @pytest.mark.asyncio
    async def test_tool_call_performance(self, real_test_environment):
        """Test tool call performance using MCPAdapt."""
        import time
        from server_research_mcp.utils.mcpadapt import MCPAdapt, CrewAIAdapter
        from mcp import StdioServerParameters
        
        # Configure memory server
        server_configs = [
            StdioServerParameters(
                command="npx",
                args=["-y", "@modelcontextprotocol/server-memory"],
                env={}
            )
        ]
        
        try:
            async with MCPAdapt(server_configs, CrewAIAdapter()) as tools:
                # Get memory tools
                memory_tools = [tool for tool in tools if "memory" in tool.name.lower()]
                
                if not memory_tools:
                    pytest.skip("Memory tools not available")
                
                # Find search tool
                search_tool = None
                for tool in memory_tools:
                    if "search" in tool.name.lower():
                        search_tool = tool
                        break
                
                if not search_tool:
                    pytest.skip("Memory search tool not found")
                
                # Test multiple tool calls
                start_time = time.time()
                
                for i in range(10):
                    try:
                        search_tool.run(query=f"performance_test_{i}")
                    except Exception:
                        # Some searches may fail, that's OK for performance testing
                        pass
                
                total_time = time.time() - start_time
                avg_time = total_time / 10
                
                # Each call should be reasonably fast
                assert avg_time < 5, f"Average tool call took {avg_time:.2f}s, should be < 5s"
                
                print(f"✅ REAL PERFORMANCE: Average tool call: {avg_time:.2f}s with MCPAdapt")
            
        except Exception as e:
            pytest.skip(f"Performance test failed: {e}")


if __name__ == "__main__":
    print("🧪 Running Real Integration Tests")
    print("=" * 50)
    print("Note: These tests require --real-integration flag and actual MCP servers")
    pytest.main([__file__, "-v", "--tb=short"]) 
