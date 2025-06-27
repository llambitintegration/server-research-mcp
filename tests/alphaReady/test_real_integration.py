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

from server_research_mcp.tools.mcp_manager import MCPManager, get_mcp_manager
from server_research_mcp.tools.enhanced_mcp_manager import EnhancedMCPManager, get_enhanced_mcp_manager
from server_research_mcp.crew import ServerResearchMcp
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
        manager = MCPManager()
        
        try:
            # Initialize memory server
            result = await manager.initialize(["memory"])
            assert "memory" in manager.clients, "Memory server should be connected"
            
            # Test basic memory operations
            search_result = await manager.call_tool(
                server="memory",
                tool="search_nodes",
                arguments={"query": "test_alpha_integration"}
            )
            
            assert "nodes" in search_result, "Memory search should return nodes"
            
            # Test entity creation
            create_result = await manager.call_tool(
                server="memory",
                tool="create_entities",
                arguments={
                    "entities": [
                        {
                            "name": "alpha_test_entity",
                            "entity_type": "test",
                            "observations": ["Alpha integration test entity"]
                        }
                    ]
                }
            )
            
            assert "status" in create_result, "Entity creation should return status"
            
            print("✅ REAL: Memory server connection successful")
            
        finally:
            await manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_context7_server_real_connection(self, real_test_environment):
        """Test real connection to context7 server."""
        manager = MCPManager()
        
        try:
            result = await manager.initialize(["context7"])
            if "context7" not in manager.clients:
                pytest.skip("Context7 server not available")
            
            # Test library resolution
            resolve_result = await manager.call_tool(
                server="context7",
                tool="resolve",
                arguments={"query": "machine learning"}
            )
            
            # Should get some kind of response
            assert resolve_result is not None
            
            print("✅ REAL: Context7 server connection successful")
            
        finally:
            await manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_enhanced_mcp_manager_real(self, real_test_environment):
        """Test enhanced MCP manager with real servers."""
        manager = EnhancedMCPManager(use_schema_fixing=True, enable_monitoring=True)
        
        try:
            # Initialize with available servers
            result = await manager.initialize(["memory"], timeout_per_server=30)
            
            # Check if memory server was initialized
            if not result.get("memory", False):
                pytest.skip("Memory server not available - requires Node.js and MCP packages")
            
            # Test health check
            health = await manager.health_check()
            assert "memory" in health, "Health check should include memory server"
            
            # Test performance metrics
            metrics = manager.get_performance_report()
            assert "total_servers" in metrics, "Performance report should include server count"
            
            print("✅ REAL: Enhanced MCP manager successful")
            
        finally:
            await manager.shutdown()


@pytest.mark.real_integration
class TestRealWorkflows:
    """Test real end-to-end workflows."""
    
    def test_crew_initialization_with_real_servers(self, real_test_environment):
        """Test crew initialization with real MCP servers available."""
        # This test verifies that crews can initialize even when real servers are present
        crew_instance = ServerResearchMcp()
        crew = crew_instance.crew()
        
        assert crew is not None
        assert len(crew.agents) == 4
        assert len(crew.tasks) == 4
        
        # Verify all agents have tools
        for agent in crew.agents:
            assert len(agent.tools) > 0, f"Agent {agent.role} should have tools"
        
        print("✅ REAL: Crew initialization with real servers successful")
    
    @pytest.mark.asyncio
    async def test_paper_search_workflow_real(self, real_test_environment):
        """Test a realistic paper search workflow."""
        # Only run if we have Zotero credentials
        if not (os.getenv("ZOTERO_API_KEY") and os.getenv("ZOTERO_LIBRARY_ID")):
            pytest.skip("Zotero credentials not available")
        
        manager = MCPManager()
        
        try:
            # Initialize Zotero server
            result = await manager.initialize(["zotero"])
            if "zotero" not in manager.clients:
                pytest.skip("Zotero server not available")
            
            # Test paper search
            search_result = await manager.call_tool(
                server="zotero",
                tool="search_items",
                arguments={
                    "query": "machine learning",
                    "limit": 5
                }
            )
            
            assert "items" in search_result, "Zotero search should return items"
            
            # If we found papers, test extraction
            if search_result["items"]:
                first_item = search_result["items"][0]
                
                # Test paper extraction
                extract_result = await manager.call_tool(
                    server="zotero",
                    tool="get_item_fulltext",
                    arguments={"item_key": first_item["key"]}
                )
                
                # Should get some content
                assert extract_result is not None
                
                print("✅ REAL: Paper search workflow successful")
            else:
                print("⚠️  REAL: No papers found, but search completed")
                
        finally:
            await manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_memory_persistence_real(self, real_test_environment):
        """Test that memory persists across sessions."""
        test_entity_name = f"alpha_test_{int(datetime.now().timestamp())}"
        
        # Session 1: Create entity
        manager1 = MCPManager()
        try:
            await manager1.initialize(["memory"])
            if "memory" not in manager1.clients:
                pytest.skip("Memory server not available")
            
            # Create test entity
            create_result = await manager1.call_tool(
                server="memory",
                tool="create_entities",
                arguments={
                    "entities": [
                        {
                            "name": test_entity_name,
                            "entity_type": "test",
                            "observations": ["Alpha persistence test"]
                        }
                    ]
                }
            )
            
            assert "status" in create_result
            
        finally:
            await manager1.shutdown()
        
        # Session 2: Search for entity
        manager2 = MCPManager()
        try:
            await manager2.initialize(["memory"])
            
            # Search for our entity
            search_result = await manager2.call_tool(
                server="memory",
                tool="search_nodes",
                arguments={"query": test_entity_name}
            )
            
            assert "nodes" in search_result
            
            # Check if our entity is found
            found_entity = False
            for node in search_result["nodes"]:
                if node.get("name") == test_entity_name:
                    found_entity = True
                    break
            
            assert found_entity, f"Entity {test_entity_name} should persist across sessions"
            
            print("✅ REAL: Memory persistence successful")
            
        finally:
            await manager2.shutdown()


@pytest.mark.real_integration
class TestRealPerformance:
    """Performance tests with real servers."""
    
    @pytest.mark.asyncio
    async def test_server_startup_performance(self, real_test_environment):
        """Test server startup performance."""
        import time
        
        manager = MCPManager()
        
        start_time = time.time()
        
        try:
            # Initialize available servers
            await manager.initialize(["memory", "context7", "sequential-thinking"])
            
            startup_time = time.time() - start_time
            
            # Should start within reasonable time
            assert startup_time < 60, f"Server startup took {startup_time:.2f}s, should be < 60s"
            
            print(f"✅ REAL PERFORMANCE: Servers started in {startup_time:.2f}s")
            
        finally:
            await manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_tool_call_performance(self, real_test_environment):
        """Test tool call performance."""
        import time
        
        manager = MCPManager()
        
        try:
            await manager.initialize(["memory"])
            if "memory" not in manager.clients:
                pytest.skip("Memory server not available")
            
            # Test multiple tool calls
            start_time = time.time()
            
            for i in range(10):
                await manager.call_tool(
                    server="memory",
                    tool="search_nodes",
                    arguments={"query": f"performance_test_{i}"}
                )
            
            total_time = time.time() - start_time
            avg_time = total_time / 10
            
            # Each call should be reasonably fast
            assert avg_time < 5, f"Average tool call took {avg_time:.2f}s, should be < 5s"
            
            print(f"✅ REAL PERFORMANCE: Average tool call: {avg_time:.2f}s")
            
        finally:
            await manager.shutdown()


if __name__ == "__main__":
    print("🧪 Running Real Integration Tests")
    print("=" * 50)
    print("Note: These tests require --real-integration flag and actual MCP servers")
    pytest.main([__file__, "-v", "--tb=short"]) 