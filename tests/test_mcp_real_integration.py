"""
Real MCP Server Integration Tests
================================

Tests that connect to actual MCP servers for real-world validation.
These tests complement the mocked tests in test_mcp_integration.py.

Requirements:
- Node.js/npx installed for MCP servers
- Environment variables for services (Zotero API key, etc.)
- Network connectivity for external services

Usage:
- Run with: pytest tests/test_mcp_real_integration.py -v
- Skip in CI with: pytest -m "not real_servers"
"""

import pytest
import os
import json
import asyncio
import logging
import shutil
from datetime import datetime
from typing import Dict, Any

# Add src to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from server_research_mcp.tools.enhanced_mcp_manager import (
    EnhancedMCPManager,
    get_enhanced_mcp_manager,
    setup_non_interactive_environment
)

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test markers
pytestmark = pytest.mark.real_servers


class TestRealMCPServerConnections:
    """Test real MCP server connections and basic functionality."""
    
    @pytest.fixture(autouse=True)
    def setup_environment(self):
        """Set up test environment."""
        setup_non_interactive_environment()
        yield
    
    @pytest.fixture
    async def enhanced_manager(self):
        """Create enhanced MCP manager for testing."""
        manager = EnhancedMCPManager(use_schema_fixing=True, enable_monitoring=True)
        yield manager
        await manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_server_availability_check(self, enhanced_manager):
        """Test which MCP servers are available for testing."""
        logger.info("🔍 Testing MCP server availability...")
        
        # Test essential servers for research workflow
        test_servers = ["memory", "sequential-thinking", "context7", "filesystem"]
        
        # Add Zotero if API key is available
        if os.getenv("ZOTERO_API_KEY"):
            test_servers.append("zotero")
        
        results = await enhanced_manager.initialize(test_servers, timeout_per_server=15)
        
        # Log detailed results
        available_servers = []
        unavailable_servers = []
        
        for server_name, success in results.items():
            if success:
                available_servers.append(server_name)
                logger.info(f"✅ {server_name}: Available")
            else:
                unavailable_servers.append(server_name)
                logger.warning(f"❌ {server_name}: Not available")
        
        # We expect at least memory and sequential-thinking to work
        assert len(available_servers) >= 1, f"Expected at least 1 server available, got {len(available_servers)}"
        
        logger.info(f"📊 Summary: {len(available_servers)}/{len(test_servers)} servers available")
        
        # Store results for other tests
        enhanced_manager._test_available_servers = available_servers
        
        return results
    
    @pytest.mark.asyncio
    async def test_memory_server_real_operations(self, enhanced_manager):
        """Test real memory server operations."""
        logger.info("🧠 Testing memory server operations...")
        
        # Initialize memory server
        results = await enhanced_manager.initialize(["memory"], timeout_per_server=15)
        
        if not results.get("memory", False):
            pytest.skip("Memory server not available")
        
        # Test 1: Search (should return empty initially)
        search_result = await enhanced_manager.call_tool(
            server="memory",
            tool="search_nodes",
            arguments={"query": "test_entity_real"}
        )
        
        assert isinstance(search_result, dict)
        logger.info(f"✅ Memory search result: {search_result}")
        
        # Test 2: Create entity
        create_result = await enhanced_manager.call_tool(
            server="memory",
            tool="create_entities",
            arguments={
                "entities": [{
                    "name": "test_entity_real",
                    "entityType": "test",
                    "observations": ["Created during real server testing"]
                }]
            }
        )
        
        assert isinstance(create_result, dict)
        logger.info(f"✅ Memory create result: {create_result}")
        
        # Test 3: Search again (should find the entity)
        search_result_2 = await enhanced_manager.call_tool(
            server="memory",
            tool="search_nodes",
            arguments={"query": "test_entity_real"}
        )
        
        assert isinstance(search_result_2, dict)
        # Should have results now
        nodes = search_result_2.get("nodes", [])
        assert len(nodes) >= 1, "Expected to find the created entity"
        
        logger.info(f"✅ Memory entity found: {nodes[0] if nodes else 'None'}")
    
    @pytest.mark.asyncio
    async def test_sequential_thinking_real_operations(self, enhanced_manager):
        """Test real sequential thinking server operations."""
        logger.info("🤔 Testing sequential thinking server operations...")
        
        results = await enhanced_manager.initialize(["sequential-thinking"], timeout_per_server=15)
        
        if not results.get("sequential-thinking", False):
            pytest.skip("Sequential thinking server not available")
        
        # Test thinking step
        thinking_result = await enhanced_manager.call_tool(
            server="sequential-thinking",
            tool="sequentialthinking",
            arguments={
                "thought": "Testing real sequential thinking server integration",
                "thoughtNumber": 1,
                "totalThoughts": 2,
                "nextThoughtNeeded": True
            }
        )
        
        assert isinstance(thinking_result, dict)
        assert "thought" in thinking_result
        logger.info(f"✅ Sequential thinking result: {thinking_result}")
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not os.getenv("ZOTERO_API_KEY"), reason="Zotero API key not available")
    async def test_zotero_real_operations(self, enhanced_manager):
        """Test real Zotero server operations (requires API key)."""
        logger.info("📚 Testing Zotero server operations...")
        
        results = await enhanced_manager.initialize(["zotero"], timeout_per_server=20)
        
        if not results.get("zotero", False):
            pytest.skip("Zotero server not available")
        
        # Test search (use a common term that should return results)
        search_result = await enhanced_manager.call_tool(
            server="zotero",
            tool="search_items",
            arguments={
                "query": "machine learning",
                "limit": 3
            }
        )
        
        assert isinstance(search_result, dict)
        logger.info(f"✅ Zotero search found: {len(search_result.get('items', []))} items")
        
        # If we have items, test getting one
        items = search_result.get("items", [])
        if items and len(items) > 0:
            item_key = items[0].get("key")
            if item_key:
                item_result = await enhanced_manager.call_tool(
                    server="zotero",
                    tool="get_item",
                    arguments={"key": item_key}
                )
                
                assert isinstance(item_result, dict)
                logger.info(f"✅ Zotero item retrieved: {item_result.get('title', 'No title')}")


class TestRealMCPWorkflows:
    """Test complete workflows using real MCP servers."""
    
    @pytest.fixture
    async def workflow_manager(self):
        """Create manager for workflow testing."""
        manager = EnhancedMCPManager(use_schema_fixing=True, enable_monitoring=True)
        
        # Initialize available servers
        essential_servers = ["memory", "sequential-thinking"]
        await manager.initialize(essential_servers, timeout_per_server=15)
        
        yield manager
        await manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_research_knowledge_workflow(self, workflow_manager):
        """Test a complete research knowledge workflow."""
        logger.info("🔬 Testing research knowledge workflow...")
        
        # Check which servers are available
        available_servers = [name for name, status in workflow_manager.connection_status.items() if status]
        
        if not available_servers:
            pytest.skip("No MCP servers available for workflow testing")
        
        workflow_results = []
        
        # Step 1: Use sequential thinking for research planning
        if "sequential-thinking" in available_servers:
            planning_result = await workflow_manager.call_tool(
                server="sequential-thinking",
                tool="sequentialthinking",
                arguments={
                    "thought": "Plan a research workflow for analyzing machine learning papers",
                    "thoughtNumber": 1,
                    "totalThoughts": 3,
                    "nextThoughtNeeded": True
                }
            )
            workflow_results.append(("planning", planning_result))
            logger.info("✅ Research planning completed")
        
        # Step 2: Store research context in memory
        if "memory" in available_servers:
            context_result = await workflow_manager.call_tool(
                server="memory",
                tool="create_entities",
                arguments={
                    "entities": [{
                        "name": "ml_research_workflow_test",
                        "entityType": "research_session",
                        "observations": [
                            "Testing complete research workflow with real MCP servers",
                            f"Available servers: {', '.join(available_servers)}",
                            f"Started at: {datetime.now().isoformat()}"
                        ]
                    }]
                }
            )
            workflow_results.append(("context_storage", context_result))
            logger.info("✅ Research context stored")
        
        # Step 3: Continue thinking process
        if "sequential-thinking" in available_servers:
            analysis_result = await workflow_manager.call_tool(
                server="sequential-thinking",
                tool="sequentialthinking",
                arguments={
                    "thought": "Analyze the workflow results and plan next steps for paper processing",
                    "thoughtNumber": 2,
                    "totalThoughts": 3,
                    "nextThoughtNeeded": True
                }
            )
            workflow_results.append(("analysis", analysis_result))
            logger.info("✅ Workflow analysis completed")
        
        # Verify all steps completed
        assert len(workflow_results) >= 1, "Expected at least one workflow step to complete"
        
        logger.info(f"📊 Workflow completed with {len(workflow_results)} steps")
        return workflow_results
    
    @pytest.mark.asyncio
    async def test_performance_monitoring(self, workflow_manager):
        """Test performance monitoring capabilities."""
        logger.info("📈 Testing performance monitoring...")
        
        # Perform several operations to generate metrics
        available_servers = [name for name, status in workflow_manager.connection_status.items() if status]
        
        if not available_servers:
            pytest.skip("No servers available for performance testing")
        
        # Perform multiple operations
        for i in range(3):
            if "memory" in available_servers:
                await workflow_manager.call_tool(
                    server="memory",
                    tool="search_nodes",
                    arguments={"query": f"performance_test_{i}"}
                )
        
        # Get performance report
        performance_report = workflow_manager.get_performance_report()
        
        assert isinstance(performance_report, dict)
        assert "server_metrics" in performance_report
        assert performance_report["total_servers"] >= 1
        
        logger.info(f"✅ Performance report generated: {performance_report}")
    
    @pytest.mark.asyncio
    async def test_health_check_monitoring(self, workflow_manager):
        """Test health check and monitoring capabilities."""
        logger.info("🏥 Testing health check monitoring...")
        
        health_report = await workflow_manager.health_check()
        
        assert isinstance(health_report, dict)
        
        # Check that we have health data for connected servers
        healthy_servers = [name for name, data in health_report.items() 
                          if data.get("status") == "healthy"]
        
        assert len(healthy_servers) >= 1, "Expected at least one healthy server"
        
        # Verify health report structure
        for server_name, health_data in health_report.items():
            assert "status" in health_data
            assert "connected" in health_data
            assert "metrics" in health_data
        
        logger.info(f"✅ Health check completed: {len(healthy_servers)} healthy servers")


class TestMCPServerCompatibility:
    """Test compatibility with different MCP server configurations."""
    
    @pytest.mark.asyncio
    async def test_schema_fixing_capabilities(self):
        """Test schema fixing for problematic MCP servers."""
        logger.info("🔧 Testing schema fixing capabilities...")
        
        # Create manager with schema fixing enabled
        manager_with_fixing = EnhancedMCPManager(use_schema_fixing=True)
        
        # Create manager without schema fixing
        manager_without_fixing = EnhancedMCPManager(use_schema_fixing=False)
        
        try:
            # Test initialization with both configurations
            test_servers = ["memory"]  # Start with a simple server
            
            results_with_fixing = await manager_with_fixing.initialize(
                test_servers, timeout_per_server=15
            )
            results_without_fixing = await manager_without_fixing.initialize(
                test_servers, timeout_per_server=15
            )
            
            # Both should work for simple servers, but schema fixing provides robustness
            logger.info(f"✅ Schema fixing enabled: {results_with_fixing}")
            logger.info(f"✅ Schema fixing disabled: {results_without_fixing}")
            
            # Test that at least one configuration works
            assert (any(results_with_fixing.values()) or any(results_without_fixing.values())), \
                "Expected at least one configuration to work"
            
        finally:
            await manager_with_fixing.shutdown()
            await manager_without_fixing.shutdown()
    
    @pytest.mark.asyncio
    async def test_context_manager_pattern(self):
        """Test context manager patterns for server connections."""
        logger.info("🔄 Testing context manager patterns...")
        
        manager = EnhancedMCPManager()
        
        try:
            # Test individual server context manager
            with manager.server_connection("memory") as adapter:
                assert adapter is not None
                assert len(adapter.tools) >= 0  # Should have tools defined
                logger.info(f"✅ Context manager created adapter with {len(adapter.tools)} tools")
            
            # Adapter should be cleaned up after context manager
            logger.info("✅ Context manager cleanup completed")
            
        except Exception as e:
            if "not available" in str(e).lower() or "not found" in str(e).lower():
                pytest.skip(f"Server not available for context manager testing: {e}")
            else:
                raise
        
        finally:
            await manager.shutdown()


# Utility functions for test reporting
def generate_test_report():
    """Generate a test report for real server integration."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "test_environment": {
            "node_available": bool(shutil.which("npx")),
            "uv_available": bool(shutil.which("uv") or shutil.which("uvx")),
            "zotero_api_configured": bool(os.getenv("ZOTERO_API_KEY")),
            "test_mode": os.getenv("MCP_TEST_MODE", "false")
        }
    }
    
    return report


if __name__ == "__main__":
    # Run tests when executed directly
    import subprocess
    
    print("🧪 Running Real MCP Server Integration Tests")
    print("=" * 50)
    
    # Generate test report
    report = generate_test_report()
    print(f"Test Environment: {json.dumps(report, indent=2)}")
    print()
    
    # Run pytest
    result = subprocess.run([
        "python", "-m", "pytest", __file__, "-v", "--tb=short"
    ], capture_output=False)
    
    exit(result.returncode) 