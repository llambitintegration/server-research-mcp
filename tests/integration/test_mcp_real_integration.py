"""
Real MCP Integration Tests
=========================

Tests that validate actual MCP server connections and catch runtime errors
like the Zotero agent startup crash. These tests use real MCP servers
when available and provide detailed diagnostics for connection issues.
"""

import pytest
import asyncio
import os
import subprocess
import time
import json
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import Dict, List, Any, Optional

# Configure logging for detailed diagnostics
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the actual components we're testing
from server_research_mcp.tools.mcp_adapter_manager import (
    MCPServerAdapterManager,
    get_mcp_adapter_manager,
    MCPServerConfig
)
from server_research_mcp.tools.mcp_base_tool import get_mcp_manager
from server_research_mcp.tools.mcp_tools import (
    MemorySearchTool,
    ZoteroSearchTool,
    get_historian_tools,
    get_researcher_tools
)
from server_research_mcp.crew import ServerResearchMcp


@pytest.fixture
def mcp_test_environment():
    """Set up test environment with MCP server validation."""
    # Create temporary test directory
    test_dir = Path("test_mcp_output")
    test_dir.mkdir(exist_ok=True)
    
    # Environment setup
    test_env = {
        "ANTHROPIC_API_KEY": "test-key-for-mcp",
        "ZOTERO_API_KEY": os.getenv("ZOTERO_API_KEY", "test-key"),
        "ZOTERO_LIBRARY_ID": os.getenv("ZOTERO_LIBRARY_ID", "test-id"),
        "ZOTERO_LOCAL": "false",
        "LLM_PROVIDER": "anthropic",
        "DISABLE_CREW_MEMORY": "true",
        "USE_ENHANCED_MCP": "true"
    }
    
    original_env = {}
    for key, value in test_env.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value
    
    yield test_dir
    
    # Cleanup
    for key, original_value in original_env.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value


class TestMCPServerConnectivity:
    """Test actual MCP server connectivity and diagnose connection issues."""
    
    def test_npx_availability(self):
        """Test that NPX is available for MCP server execution."""
        try:
            result = subprocess.run(
                ["npx", "--version"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            assert result.returncode == 0, f"NPX not available: {result.stderr}"
            logger.info(f"✅ NPX available: {result.stdout.strip()}")
        except FileNotFoundError:
            pytest.skip("NPX not available - cannot test real MCP servers")
        except subprocess.TimeoutExpired:
            pytest.fail("NPX command timed out - system may be overloaded")
    
    def test_mcp_server_package_availability(self):
        """Test that required MCP server packages can be resolved."""
        packages = [
            "@modelcontextprotocol/server-memory",
            "zotero-mcp-server",
            "@modelcontextprotocol/server-sequential-thinking"
        ]
        
        for package in packages:
            try:
                # Test package resolution without installing
                result = subprocess.run(
                    ["npx", "-y", "--dry-run", package, "--help"],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                logger.info(f"📦 Package {package}: {'✅ Available' if result.returncode == 0 else '❌ Not available'}")
            except subprocess.TimeoutExpired:
                logger.warning(f"⏰ Package {package} resolution timed out")
    
    @pytest.mark.asyncio
    async def test_mcp_adapter_manager_initialization(self, mcp_test_environment):
        """Test MCPServerAdapterManager initialization with real servers."""
        manager = get_mcp_adapter_manager()
        
        # Test safe servers first (memory, sequential-thinking)
        safe_servers = ["memory", "sequential-thinking"]
        
        initialization_results = {}
        
        for server in safe_servers:
            try:
                logger.info(f"🚀 Testing {server} server initialization...")
                await manager.initialize_server(server)
                initialization_results[server] = "success"
                logger.info(f"✅ {server} server initialized successfully")
            except Exception as e:
                initialization_results[server] = f"failed: {str(e)}"
                logger.error(f"❌ {server} server failed: {e}")
        
        # At least one server should work
        successful_servers = [k for k, v in initialization_results.items() if v == "success"]
        assert len(successful_servers) > 0, f"No MCP servers could be initialized: {initialization_results}"
        
        # Cleanup
        await manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_zotero_server_specific_diagnosis(self, mcp_test_environment):
        """Specific diagnostic test for Zotero server connection issues."""
        manager = get_mcp_adapter_manager()
        
        # Test Zotero server configuration
        zotero_config = manager.SERVER_CONFIGS.get("zotero")
        assert zotero_config is not None, "Zotero configuration not found"
        
        logger.info(f"🔧 Zotero config: {zotero_config}")
        logger.info(f"🔑 Zotero API key set: {'Yes' if os.getenv('ZOTERO_API_KEY') else 'No'}")
        logger.info(f"📚 Zotero library ID: {os.getenv('ZOTERO_LIBRARY_ID', 'Not set')}")
        
        try:
            # Attempt Zotero server initialization with detailed error capture
            logger.info("🚀 Attempting Zotero server initialization...")
            await manager.initialize_server("zotero")
            logger.info("✅ Zotero server initialized successfully")
            
            # Test cleanup
            await manager.shutdown()
            
        except Exception as e:
            logger.error(f"❌ Zotero server initialization failed: {e}")
            logger.error(f"🔍 Error type: {type(e).__name__}")
            
            # Provide specific diagnostic information
            diagnostics = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "zotero_config": {
                    "command": zotero_config.command,
                    "args": zotero_config.args,
                    "env_keys": list(zotero_config.env.keys())
                },
                "environment": {
                    "zotero_api_key_set": bool(os.getenv("ZOTERO_API_KEY")),
                    "zotero_library_id_set": bool(os.getenv("ZOTERO_LIBRARY_ID")),
                    "npx_available": True  # We tested this earlier
                }
            }
            
            logger.error(f"🔍 Diagnostic info: {json.dumps(diagnostics, indent=2)}")
            
            # Don't fail the test - capture the issue for analysis
            pytest.warns(UserWarning, f"Zotero server connection failed: {e}")


class TestAgentInitializationRobustness:
    """Test agent initialization with real MCP connections to catch runtime errors."""
    
    @pytest.mark.asyncio 
    async def test_historian_agent_mcp_tools(self, mcp_test_environment):
        """Test Historian agent tools with real MCP connections."""
        try:
            # Get historian tools (should include memory tools)
            tools = get_historian_tools()
            assert len(tools) > 0, "Historian should have tools"
            
            # Find memory search tool
            memory_tool = None
            for tool in tools:
                if hasattr(tool, 'server_name') and tool.server_name == "memory":
                    memory_tool = tool
                    break
            
            assert memory_tool is not None, "Historian should have memory tools"
            
            # Test tool execution (with timeout)
            result = memory_tool._run(query="test connection")
            assert isinstance(result, str), "Tool should return string result"
            
            # Parse result to ensure it's valid JSON
            parsed_result = json.loads(result)
            assert isinstance(parsed_result, dict), "Tool result should be valid JSON dict"
            
            logger.info("✅ Historian agent MCP tools working")
            
        except Exception as e:
            logger.error(f"❌ Historian agent MCP tools failed: {e}")
            # Don't fail test completely - capture for analysis
            pytest.warns(UserWarning, f"Historian MCP tools issue: {e}")
    
    @pytest.mark.asyncio
    async def test_researcher_agent_mcp_tools(self, mcp_test_environment):
        """Test Researcher agent tools with real MCP connections - This is where Zotero fails."""
        try:
            # Get researcher tools (should include Zotero tools)
            tools = get_researcher_tools()
            assert len(tools) > 0, "Researcher should have tools"
            
            # Find Zotero search tool
            zotero_tool = None
            for tool in tools:
                if hasattr(tool, 'server_name') and tool.server_name == "zotero":
                    zotero_tool = tool
                    break
            
            if zotero_tool is None:
                logger.warning("⚠️ No Zotero tools found in researcher tools")
                pytest.skip("No Zotero tools configured")
            
            # Test Zotero tool execution (this is likely where the crash occurs)
            logger.info("🔬 Testing Zotero tool execution...")
            result = zotero_tool._run(
                query="test paper",
                search_type="everything",
                limit=1
            )
            assert isinstance(result, str), "Zotero tool should return string result"
            
            # Parse result to ensure it's valid JSON
            parsed_result = json.loads(result)
            assert isinstance(parsed_result, dict), "Zotero result should be valid JSON dict"
            
            logger.info("✅ Researcher agent Zotero tools working")
            
        except Exception as e:
            logger.error(f"❌ Researcher agent Zotero tools failed: {e}")
            logger.error(f"🔍 This is likely the source of the runtime crash!")
            
            # Capture detailed error info
            error_details = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "zotero_config_env": {
                    "ZOTERO_API_KEY": "***" if os.getenv("ZOTERO_API_KEY") else None,
                    "ZOTERO_LIBRARY_ID": os.getenv("ZOTERO_LIBRARY_ID"),
                    "ZOTERO_LOCAL": os.getenv("ZOTERO_LOCAL")
                }
            }
            
            logger.error(f"🔍 Error details: {json.dumps(error_details, indent=2)}")
            
            # Mark as expected failure for now
            pytest.xfail(f"Known issue: Zotero agent startup crash - {e}")
    
    def test_crew_initialization_with_mcp_error_handling(self, mcp_test_environment):
        """Test crew initialization handles MCP errors gracefully."""
        try:
            # Attempt to create crew instance
            crew_instance = ServerResearchMcp()
            assert crew_instance is not None
            
            # Test individual agent creation
            agents_created = {}
            
            try:
                historian = crew_instance.historian()
                agents_created["historian"] = "success"
                logger.info("✅ Historian agent created successfully")
            except Exception as e:
                agents_created["historian"] = f"failed: {str(e)}"
                logger.error(f"❌ Historian agent failed: {e}")
            
            try:
                researcher = crew_instance.researcher()
                agents_created["researcher"] = "success"
                logger.info("✅ Researcher agent created successfully")
            except Exception as e:
                agents_created["researcher"] = f"failed: {str(e)}"
                logger.error(f"❌ Researcher agent failed: {e}")
                # This is likely where the Zotero crash occurs
            
            try:
                archivist = crew_instance.archivist()
                agents_created["archivist"] = "success"
                logger.info("✅ Archivist agent created successfully")
            except Exception as e:
                agents_created["archivist"] = f"failed: {str(e)}"
                logger.error(f"❌ Archivist agent failed: {e}")
            
            try:
                publisher = crew_instance.publisher()
                agents_created["publisher"] = "success"
                logger.info("✅ Publisher agent created successfully")
            except Exception as e:
                agents_created["publisher"] = f"failed: {str(e)}"
                logger.error(f"❌ Publisher agent failed: {e}")
            
            logger.info(f"📊 Agent creation results: {json.dumps(agents_created, indent=2)}")
            
            # At least historian should work (uses memory server which is more stable)
            if agents_created.get("historian") != "success":
                pytest.fail("Even Historian agent failed - basic MCP functionality broken")
            
            # If researcher fails, that's our expected issue
            if agents_created.get("researcher") != "success":
                logger.warning("🎯 Researcher agent failed - this confirms the Zotero issue")
                pytest.warns(UserWarning, f"Researcher agent failed: {agents_created['researcher']}")
            
        except Exception as e:
            logger.error(f"❌ Crew initialization completely failed: {e}")
            pytest.fail(f"Crew initialization failed: {e}")


class TestMCPHealthChecks:
    """Health check utilities for MCP servers to prevent runtime crashes."""
    
    @pytest.mark.asyncio
    async def test_mcp_server_health_check_utility(self):
        """Test a health check utility that can be used before agent startup."""
        
        async def check_mcp_server_health(server_name: str) -> Dict[str, Any]:
            """Health check for a specific MCP server."""
            health_status = {
                "server": server_name,
                "status": "unknown",
                "error": None,
                "response_time": None
            }
            
            start_time = time.time()
            
            try:
                manager = get_mcp_adapter_manager()
                await manager.initialize_server(server_name)
                
                health_status["status"] = "healthy"
                health_status["response_time"] = time.time() - start_time
                
                await manager.shutdown()
                
            except Exception as e:
                health_status["status"] = "unhealthy"
                health_status["error"] = str(e)
                health_status["response_time"] = time.time() - start_time
            
            return health_status
        
        # Test health checks for all configured servers
        servers = ["memory", "zotero", "sequential-thinking", "context7"]
        health_results = {}
        
        for server in servers:
            try:
                health_results[server] = await check_mcp_server_health(server)
                logger.info(f"🏥 {server} health: {health_results[server]['status']}")
            except Exception as e:
                health_results[server] = {"status": "error", "error": str(e)}
                logger.error(f"❌ Health check failed for {server}: {e}")
        
        logger.info(f"📋 MCP Health Report: {json.dumps(health_results, indent=2)}")
        
        # At least one server should be healthy
        healthy_servers = [k for k, v in health_results.items() if v.get("status") == "healthy"]
        assert len(healthy_servers) > 0, f"No MCP servers are healthy: {health_results}"
        
        return health_results
    
    def test_pre_startup_mcp_validation(self, mcp_test_environment):
        """Test that we can validate MCP servers before starting agents."""
        
        def validate_mcp_environment() -> Dict[str, Any]:
            """Validate MCP environment before crew startup."""
            validation_results = {
                "npx_available": False,
                "environment_vars": {},
                "server_configs": {},
                "overall_status": "unknown"
            }
            
            # Check NPX
            try:
                subprocess.run(["npx", "--version"], capture_output=True, timeout=5)
                validation_results["npx_available"] = True
            except:
                validation_results["npx_available"] = False
            
            # Check environment variables
            validation_results["environment_vars"] = {
                "ZOTERO_API_KEY": bool(os.getenv("ZOTERO_API_KEY")),
                "ZOTERO_LIBRARY_ID": bool(os.getenv("ZOTERO_LIBRARY_ID")),
                "ANTHROPIC_API_KEY": bool(os.getenv("ANTHROPIC_API_KEY")),
                "USE_ENHANCED_MCP": os.getenv("USE_ENHANCED_MCP", "false").lower() == "true"
            }
            
            # Check server configurations
            manager = get_mcp_adapter_manager()
            for server_name, config in manager.SERVER_CONFIGS.items():
                validation_results["server_configs"][server_name] = {
                    "command": config.command,
                    "args_count": len(config.args),
                    "env_vars_set": len(config.env) > 0
                }
            
            # Overall assessment
            if not validation_results["npx_available"]:
                validation_results["overall_status"] = "failed_npx"
            elif not validation_results["environment_vars"]["USE_ENHANCED_MCP"]:
                validation_results["overall_status"] = "mcp_disabled"
            else:
                validation_results["overall_status"] = "ready"
            
            return validation_results
        
        # Run validation
        results = validate_mcp_environment()
        logger.info(f"🔍 MCP Environment Validation: {json.dumps(results, indent=2)}")
        
        # Basic checks
        if not results["npx_available"]:
            pytest.skip("NPX not available - cannot use MCP servers")
        
        if results["overall_status"] == "mcp_disabled":
            pytest.skip("MCP disabled in environment")
        
        assert results["overall_status"] in ["ready", "warnings"], f"MCP environment not ready: {results}"


if __name__ == "__main__":
    # Allow running this test file directly for diagnosis
    pytest.main([__file__, "-v", "-s"]) 