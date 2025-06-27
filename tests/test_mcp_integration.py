"""Comprehensive MCP (Model Context Protocol) integration tests."""

import pytest
import asyncio
import json
from unittest.mock import MagicMock, patch, AsyncMock, call
from datetime import datetime
import subprocess
import os


class TestMCPServerManagement:
    """Test MCP server lifecycle management."""
    
    @pytest.mark.mcp
    def test_mcp_server_startup(self, test_environment):
        """Test MCP servers can be started successfully."""
        servers = [
            ("memory", "@modelcontextprotocol/server-memory"),
            ("context7", "@upstash/context7-mcp"),
            ("sequential-thinking", "@modelcontextprotocol/server-sequential-thinking")
        ]
        
        for server_name, package in servers:
            # Mock subprocess to simulate server startup
            with patch('subprocess.Popen') as mock_popen:
                mock_process = MagicMock()
                mock_process.poll.return_value = None  # Server is running
                mock_process.pid = 12345
                mock_popen.return_value = mock_process
                
                # Simulate server start
                process = subprocess.Popen(["npx", "-y", package])
                
                assert process.poll() is None
                assert process.pid > 0
    
    @pytest.mark.mcp
    def test_mcp_server_health_check(self, mock_mcp_manager):
        """Test MCP server health monitoring."""
        # Check server connectivity
        assert mock_mcp_manager.is_connected()
        
        # List available tools
        tools = mock_mcp_manager.list_tools()
        expected_tools = [
            "memory_search", "memory_create_entity", "memory_add_observation",
            "context7_resolve_library", "context7_get_docs",
            "sequential_thinking_append_thought", "sequential_thinking_get_thoughts"
        ]
        
        for tool in expected_tools:
            assert tool in tools
    
    @pytest.mark.mcp
    def test_mcp_server_restart(self, mock_mcp_manager):
        """Test MCP server restart capability."""
        # Simulate server failure and restart
        mock_mcp_manager.is_connected = MagicMock(side_effect=[False, False, True])
        mock_mcp_manager.restart = MagicMock(return_value=True)
        
        # Initial check - server down
        assert not mock_mcp_manager.is_connected()
        
        # Restart server
        assert mock_mcp_manager.restart()
        
        # Verify server is back up
        assert mock_mcp_manager.is_connected()
    
    @pytest.mark.mcp
    def test_mcp_server_graceful_shutdown(self, mock_mcp_manager):
        """Test graceful MCP server shutdown."""
        mock_mcp_manager.shutdown = MagicMock(return_value=True)
        
        # Shutdown server
        result = mock_mcp_manager.shutdown()
        assert result is True
        
        mock_mcp_manager.shutdown.assert_called_once()


class TestMCPToolExecution:
    """Test MCP tool execution and error handling."""
    
    def test_memory_tool_execution(self, mock_mcp_manager):
        """Test memory tool execution flow."""
        # Search memory
        search_result = mock_mcp_manager.call_tool(
            "memory_search",
            query="artificial intelligence testing"
        )
        
        assert "results" in search_result
        assert len(search_result["results"]) >= 1
        
        # Create entity
        create_result = mock_mcp_manager.call_tool(
            "memory_create_entity",
            name="ai_testing",
            entity_type="concept",
            observations=["Testing methodology for AI", "Includes validation techniques"]
        )
        
        assert create_result["success"] is True
        assert "entity_id" in create_result
        
        # Add observations
        add_result = mock_mcp_manager.call_tool(
            "memory_add_observation",
            entity_name="ai_testing",
            observations=["New observation about edge cases"]
        )
        
        assert add_result["success"] is True
    
    def test_context7_tool_execution(self, mock_mcp_manager):
        """Test Context7 tool execution flow."""
        # Resolve library
        resolve_result = mock_mcp_manager.call_tool(
            "context7_resolve_library",
            library_name="tensorflow"
        )
        
        assert "library_id" in resolve_result
        assert resolve_result["confidence"] > 0.8
        
        # Get documentation
        docs_result = mock_mcp_manager.call_tool(
            "context7_get_docs",
            library_id=resolve_result["library_id"],
            topic="neural networks",
            tokens=10000
        )
        
        assert "content" in docs_result
        assert docs_result["tokens_used"] <= 10000
        assert len(docs_result["sections"]) > 0
    
    def test_sequential_thinking_execution(self, mock_mcp_manager):
        """Test sequential thinking tool execution."""
        thoughts = []
        
        # Append multiple thoughts
        for i in range(5):
            thought_result = mock_mcp_manager.call_tool(
                "sequential_thinking_append_thought",
                thought=f"Analysis step {i+1}: Examining aspect {chr(65+i)}",
                thought_number=i+1,
                total_thoughts=5
            )
            thoughts.append(thought_result)
            assert thought_result["status"] == "recorded"
        
        # Retrieve all thoughts
        all_thoughts = mock_mcp_manager.call_tool("sequential_thinking_get_thoughts")
        assert len(all_thoughts["thoughts"]) >= 3
    
    def test_mcp_tool_error_handling(self, mock_mcp_manager):
        """Test MCP tool error handling."""
        # Test invalid tool name
        error_result = mock_mcp_manager.call_tool("invalid_tool_name")
        assert "error" in error_result
        
        # Test with missing parameters
        with patch.object(mock_mcp_manager, 'call_tool', side_effect=ValueError("Missing required parameter")):
            with pytest.raises(ValueError):
                mock_mcp_manager.call_tool("memory_search")  # Missing query parameter
    
    def test_mcp_tool_timeout_handling(self, mock_mcp_manager):
        """Test MCP tool timeout handling."""
        # Simulate timeout
        with patch.object(mock_mcp_manager, 'call_tool', side_effect=TimeoutError("Tool execution timeout")):
            with pytest.raises(TimeoutError):
                mock_mcp_manager.call_tool(
                    "context7_get_docs",
                    library_id="/large-library/docs",
                    tokens=100000  # Large request
                )


class TestMCPAsyncOperations:
    """Test asynchronous MCP operations."""
    
    @pytest.mark.asyncio
    async def test_async_tool_execution(self, async_mcp_client):
        """Test async MCP tool execution."""
        await async_mcp_client.connect()
        
        # Execute tool asynchronously
        result = await async_mcp_client.call_tool(
            "memory_search",
            query="async testing patterns"
        )
        
        assert result["status"] == "success"
        
        await async_mcp_client.disconnect()
    
    @pytest.mark.asyncio
    async def test_concurrent_tool_execution(self, async_mcp_client):
        """Test concurrent MCP tool execution."""
        await async_mcp_client.connect()
        
        # Execute multiple tools concurrently
        tasks = [
            async_mcp_client.call_tool("memory_search", query="query1"),
            async_mcp_client.call_tool("memory_search", query="query2"),
            async_mcp_client.call_tool("context7_resolve_library", library_name="numpy")
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        assert all(r["status"] == "success" for r in results)
        
        await async_mcp_client.disconnect()
    
    @pytest.mark.asyncio
    async def test_async_batch_operations(self, async_mcp_client):
        """Test batch MCP operations."""
        await async_mcp_client.connect()
        
        # Batch create entities
        entities = [
            ("concept1", "testing_concept", ["obs1", "obs2"]),
            ("concept2", "testing_concept", ["obs3", "obs4"]),
            ("concept3", "testing_concept", ["obs5", "obs6"])
        ]
        
        tasks = []
        for name, entity_type, observations in entities:
            task = async_mcp_client.call_tool(
                "memory_create_entity",
                name=name,
                entity_type=entity_type,
                observations=observations
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        assert len(results) == 3
        assert all(r["status"] == "success" for r in results)
        
        await async_mcp_client.disconnect()


class TestMCPIntegrationPatterns:
    """Test common MCP integration patterns."""
    
    def test_knowledge_graph_building(self, mock_mcp_manager):
        """Test building a knowledge graph through MCP."""
        # Create nodes
        nodes = []
        for i in range(3):
            result = mock_mcp_manager.call_tool(
                "memory_create_entity",
                name=f"node_{i}",
                entity_type="concept",
                observations=[f"Node {i} observation"]
            )
            nodes.append(result["entity_id"])
        
        # Add relationships (through observations)
        for i in range(len(nodes) - 1):
            mock_mcp_manager.call_tool(
                "memory_add_observation",
                entity_name=f"node_{i}",
                observations=[f"Connected to node_{i+1}"]
            )
        
        # Search for connected nodes
        search_result = mock_mcp_manager.call_tool(
            "memory_search",
            query="Connected to"
        )
        
        assert len(search_result["results"]) >= 2
    
    def test_research_workflow_integration(self, mock_mcp_manager):
        """Test complete research workflow using MCP tools."""
        workflow_steps = []
        
        # Step 1: Search existing knowledge
        search_result = mock_mcp_manager.call_tool(
            "memory_search",
            query="machine learning frameworks"
        )
        workflow_steps.append(("search", search_result))
        
        # Step 2: Resolve library documentation
        if not search_result["results"]:
            resolve_result = mock_mcp_manager.call_tool(
                "context7_resolve_library",
                library_name="scikit-learn"
            )
            workflow_steps.append(("resolve", resolve_result))
            
            # Step 3: Get documentation
            docs_result = mock_mcp_manager.call_tool(
                "context7_get_docs",
                library_id=resolve_result["library_id"],
                topic="classification algorithms",
                tokens=5000
            )
            workflow_steps.append(("docs", docs_result))
        
        # Step 4: Sequential analysis
        for i in range(3):
            thought_result = mock_mcp_manager.call_tool(
                "sequential_thinking_append_thought",
                thought=f"Analyzing {workflow_steps[-1][0]} results",
                thought_number=i+1,
                total_thoughts=3
            )
            workflow_steps.append(("thinking", thought_result))
        
        # Step 5: Create knowledge entity
        create_result = mock_mcp_manager.call_tool(
            "memory_create_entity",
            name="ml_frameworks_research",
            entity_type="research_topic",
            observations=[
                "Scikit-learn is primary Python ML framework",
                "Supports various classification algorithms",
                "Well-documented API"
            ]
        )
        workflow_steps.append(("create", create_result))
        
        assert len(workflow_steps) >= 5
        assert all(step[1] for step in workflow_steps)  # All steps succeeded
    
    def test_mcp_caching_strategy(self, mock_mcp_manager, mock_chromadb):
        """Test caching strategy for MCP responses."""
        cache = {}
        
        def cached_call_tool(tool_name, **kwargs):
            cache_key = f"{tool_name}:{json.dumps(kwargs, sort_keys=True)}"
            
            if cache_key in cache:
                return cache[cache_key]
            
            result = mock_mcp_manager.call_tool(tool_name, **kwargs)
            cache[cache_key] = result
            return result
        
        # First call - cache miss
        result1 = cached_call_tool("memory_search", query="test query")
        assert len(cache) == 1
        
        # Second call - cache hit
        result2 = cached_call_tool("memory_search", query="test query")
        assert result1 == result2
        assert len(cache) == 1  # No new cache entry
        
        # Different query - cache miss
        result3 = cached_call_tool("memory_search", query="different query")
        assert len(cache) == 2


class TestMCPErrorRecovery:
    """Test MCP error recovery mechanisms."""
    
    def test_connection_retry_logic(self, mock_mcp_manager):
        """Test connection retry logic."""
        retry_count = 0
        max_retries = 3
        
        def flaky_connection():
            nonlocal retry_count
            retry_count += 1
            if retry_count < max_retries:
                raise ConnectionError("Connection failed")
            return True
        
        mock_mcp_manager.connect = MagicMock(side_effect=flaky_connection)
        
        # Retry until success
        for i in range(max_retries):
            try:
                mock_mcp_manager.connect()
                break
            except ConnectionError:
                if i == max_retries - 1:
                    raise
                continue
        
        assert retry_count == max_retries
    
    def test_partial_failure_handling(self, mock_mcp_manager):
        """Test handling partial failures in batch operations."""
        results = []
        errors = []
        
        # Simulate mixed success/failure
        operations = [
            ("memory_search", {"query": "test1"}, True),
            ("memory_search", {"query": "test2"}, False),  # This will fail
            ("memory_search", {"query": "test3"}, True)
        ]
        
        for tool, params, should_succeed in operations:
            try:
                if not should_succeed:
                    raise Exception(f"Operation failed: {tool}")
                
                result = mock_mcp_manager.call_tool(tool, **params)
                results.append(result)
            except Exception as e:
                errors.append({"tool": tool, "params": params, "error": str(e)})
        
        assert len(results) == 2  # Two successful operations
        assert len(errors) == 1   # One failed operation
        assert errors[0]["tool"] == "memory_search"
    
    def test_state_recovery_after_crash(self, mock_mcp_manager, temp_workspace):
        """Test state recovery after MCP crash."""
        state_file = f"{temp_workspace}/mcp_state.json"
        
        # Save state before crash
        state = {
            "last_operation": "memory_create_entity",
            "timestamp": datetime.now().isoformat(),
            "pending_operations": [
                {"tool": "memory_add_observation", "params": {"entity_name": "test"}}
            ]
        }
        
        with open(state_file, 'w') as f:
            json.dump(state, f)
        
        # Simulate crash and recovery
        mock_mcp_manager.is_connected = MagicMock(return_value=False)
        
        # Load state and resume
        with open(state_file, 'r') as f:
            recovered_state = json.load(f)
        
        assert recovered_state["last_operation"] == "memory_create_entity"
        assert len(recovered_state["pending_operations"]) == 1
        
        # Resume pending operations
        for op in recovered_state["pending_operations"]:
            mock_mcp_manager.call_tool(op["tool"], **op["params"])


class TestMCPPerformance:
    """Test MCP performance characteristics."""
    
    @pytest.mark.performance
    def test_tool_execution_performance(self, mock_mcp_manager, performance_monitor):
        """Test MCP tool execution performance."""
        iterations = 100
        
        performance_monitor.start_timer("memory_search_batch")
        
        for i in range(iterations):
            mock_mcp_manager.call_tool("memory_search", query=f"query_{i}")
        
        elapsed = performance_monitor.stop_timer("memory_search_batch")
        avg_time = elapsed / iterations
        
        # Performance assertions
        assert avg_time < 0.1  # Average should be under 100ms
        assert elapsed < 10   # Total should be under 10 seconds
    
    @pytest.mark.performance
    def test_concurrent_performance(self, mock_mcp_manager, performance_monitor):
        """Test concurrent MCP operations performance."""
        import concurrent.futures
        
        def execute_tool(index):
            return mock_mcp_manager.call_tool("memory_search", query=f"concurrent_{index}")
        
        performance_monitor.start_timer("concurrent_execution")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(execute_tool, i) for i in range(50)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        elapsed = performance_monitor.stop_timer("concurrent_execution")
        
        assert len(results) == 50
        assert elapsed < 5  # Should complete within 5 seconds
    
    @pytest.mark.performance
    def test_memory_usage_optimization(self, mock_mcp_manager):
        """Test memory usage optimization for large operations."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Execute memory-intensive operations
        large_observations = ["x" * 1000 for _ in range(100)]  # 100KB of data
        
        for i in range(10):
            mock_mcp_manager.call_tool(
                "memory_create_entity",
                name=f"large_entity_{i}",
                entity_type="data",
                observations=large_observations
            )
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 50  # Less than 50MB increase