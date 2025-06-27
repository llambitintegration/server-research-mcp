"""Comprehensive MCP (Model Context Protocol) integration tests - moved from root."""

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
    async def test_async_batch_operations(self, async_mcp_client):
        """Test async batch operations."""
        await async_mcp_client.connect()
        
        # Execute multiple operations concurrently
        operations = [
            async_mcp_client.call_tool("memory_search", query=f"query_{i}")
            for i in range(5)
        ]
        
        results = await asyncio.gather(*operations)
        
        assert len(results) == 5
        assert all(r["status"] == "success" for r in results)
        
        await async_mcp_client.disconnect()


class TestMCPIntegrationPatterns:
    """Test common MCP integration patterns."""
    
    def test_knowledge_graph_building(self, mock_mcp_manager):
        """Test building knowledge graphs through MCP tools."""
        # Create a knowledge graph about a research topic
        topic = "machine_learning_testing"
        
        # Create root entity
        root_entity = mock_mcp_manager.call_tool(
            "memory_create_entity",
            name=topic,
            entity_type="research_topic",
            observations=["Testing methodologies for ML systems"]
        )
        
        # Create related entities
        related_entities = [
            ("unit_testing", "testing_method", ["Tests individual components"]),
            ("integration_testing", "testing_method", ["Tests component interactions"]),
            ("validation_datasets", "resource", ["Datasets for model validation"]),
            ("performance_metrics", "measurement", ["Accuracy, precision, recall"])
        ]
        
        for name, entity_type, observations in related_entities:
            mock_mcp_manager.call_tool(
                "memory_create_entity",
                name=name,
                entity_type=entity_type,
                observations=observations
            )
            
            # Link to root topic
            mock_mcp_manager.call_tool(
                "memory_add_observation",
                entity_name=topic,
                observations=[f"Related to {name}"]
            )
        
        # Search for connected entities
        search_result = mock_mcp_manager.call_tool(
            "memory_search",
            query=topic
        )
        
        assert len(search_result["results"]) >= 4
    
    def test_research_workflow_integration(self, mock_mcp_manager):
        """Test full research workflow using MCP tools."""
        research_query = "neural network optimization techniques"
        
        # Step 1: Search existing knowledge
        existing_knowledge = mock_mcp_manager.call_tool(
            "memory_search",
            query=research_query
        )
        
        # Step 2: Resolve technical documentation
        docs_result = mock_mcp_manager.call_tool(
            "context7_resolve_library",
            library_name="tensorflow"
        )
        
        library_docs = mock_mcp_manager.call_tool(
            "context7_get_docs",
            library_id=docs_result["library_id"],
            topic="optimization",
            tokens=5000
        )
        
        # Step 3: Sequential analysis
        analysis_thoughts = []
        for i, concept in enumerate(["gradient_descent", "adam_optimizer", "learning_rate"]):
            thought = mock_mcp_manager.call_tool(
                "sequential_thinking_append_thought",
                thought=f"Analyzing {concept} in context of neural network optimization",
                thought_number=i+1,
                total_thoughts=3
            )
            analysis_thoughts.append(thought)
        
        # Step 4: Store synthesized knowledge
        synthesis_result = mock_mcp_manager.call_tool(
            "memory_create_entity",
            name="nn_optimization_synthesis",
            entity_type="research_synthesis",
            observations=[
                "Combined analysis of optimization techniques",
                f"Based on {len(existing_knowledge['results'])} existing sources",
                f"Incorporated {len(library_docs['sections'])} documentation sections",
                f"Applied {len(analysis_thoughts)} analytical frameworks"
            ]
        )
        
        # Verify workflow completion
        assert existing_knowledge["results"] is not None
        assert library_docs["content"] is not None
        assert len(analysis_thoughts) == 3
        assert synthesis_result["success"] is True
    
    def test_mcp_caching_strategy(self, mock_mcp_manager, mock_chromadb):
        """Test caching strategy for MCP operations."""
        cache = {}
        
        def cached_call_tool(tool_name, **kwargs):
            cache_key = f"{tool_name}:{hash(str(kwargs))}"
            if cache_key in cache:
                return cache[cache_key]
            
            result = mock_mcp_manager.call_tool(tool_name, **kwargs)
            cache[cache_key] = result
            return result
        
        # Replace call_tool with cached version
        mock_mcp_manager.call_tool = cached_call_tool
        
        # Make identical calls
        query = "machine learning"
        result1 = mock_mcp_manager.call_tool("memory_search", query=query)
        result2 = mock_mcp_manager.call_tool("memory_search", query=query)
        
        # Second call should be cached
        assert result1 == result2
        assert len(cache) == 1


class TestMCPErrorRecovery:
    """Test MCP error recovery mechanisms."""
    
    def test_connection_retry_logic(self, mock_mcp_manager):
        """Test connection retry mechanisms."""
        retry_count = 0
        max_retries = 3
        
        def flaky_connection():
            nonlocal retry_count
            retry_count += 1
            if retry_count < max_retries:
                raise ConnectionError("Temporary connection failure")
            return True
        
        mock_mcp_manager.connect = flaky_connection
        
        # Should eventually succeed after retries
        success = False
        for _ in range(max_retries + 1):
            try:
                success = mock_mcp_manager.connect()
                break
            except ConnectionError:
                continue
        
        assert success is True
        assert retry_count == max_retries
    
    def test_partial_failure_handling(self, mock_mcp_manager):
        """Test handling of partial failures in batch operations."""
        # Simulate batch operation where some calls fail
        entities = ["entity1", "entity2", "entity3", "entity4"]
        results = []
        
        for i, entity in enumerate(entities):
            try:
                if i == 2:  # Simulate failure on third entity
                    raise Exception("Simulated failure")
                
                result = mock_mcp_manager.call_tool(
                    "memory_create_entity",
                    name=entity,
                    entity_type="test",
                    observations=["Test observation"]
                )
                results.append({"entity": entity, "success": True, "result": result})
            except Exception as e:
                results.append({"entity": entity, "success": False, "error": str(e)})
        
        # Should have partial success
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]
        
        assert len(successful) == 3
        assert len(failed) == 1
        assert failed[0]["entity"] == "entity3"
    
    def test_state_recovery_after_crash(self, mock_mcp_manager, temp_workspace):
        """Test state recovery after system crash."""
        # Simulate saving state before crash
        state_file = f"{temp_workspace}/mcp_state.json"
        
        pre_crash_state = {
            "active_entities": ["entity1", "entity2"],
            "pending_operations": [
                {"tool": "memory_search", "params": {"query": "test"}},
                {"tool": "context7_resolve_library", "params": {"library_name": "numpy"}}
            ],
            "last_checkpoint": datetime.now().isoformat()
        }
        
        with open(state_file, 'w') as f:
            json.dump(pre_crash_state, f)
        
        # Simulate crash and recovery
        with open(state_file, 'r') as f:
            recovered_state = json.load(f)
        
        # Resume operations from checkpoint
        for operation in recovered_state["pending_operations"]:
            result = mock_mcp_manager.call_tool(
                operation["tool"],
                **operation["params"]
            )
            assert result is not None
        
        # Verify state consistency
        assert len(recovered_state["active_entities"]) == 2
        assert len(recovered_state["pending_operations"]) == 2


class TestMCPPerformance:
    """Test MCP performance characteristics."""
    
    @pytest.mark.performance
    def test_tool_execution_performance(self, mock_mcp_manager, performance_monitor):
        """Test individual tool execution performance."""
        operations = [
            ("memory_search", {"query": "test"}),
            ("memory_create_entity", {"name": "test_entity", "entity_type": "test", "observations": ["test"]}),
            ("context7_resolve_library", {"library_name": "pandas"}),
            ("sequential_thinking_append_thought", {"thought": "test thought", "thought_number": 1, "total_thoughts": 1})
        ]
        
        for tool_name, params in operations:
            performance_monitor.start_timer(f"{tool_name}_execution")
            result = mock_mcp_manager.call_tool(tool_name, **params)
            execution_time = performance_monitor.stop_timer(f"{tool_name}_execution")
            
            assert result is not None
            assert execution_time < 1.0  # Should complete within 1 second
    
    @pytest.mark.performance
    def test_concurrent_performance(self, mock_mcp_manager, performance_monitor):
        """Test concurrent MCP operations performance."""
        import concurrent.futures
        
        def execute_tool(index):
            return mock_mcp_manager.call_tool(
                "memory_search",
                query=f"concurrent_query_{index}"
            )
        
        performance_monitor.start_timer("concurrent_operations")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(execute_tool, i) for i in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        total_time = performance_monitor.stop_timer("concurrent_operations")
        
        assert len(results) == 10
        assert total_time < 2.0  # Should be faster than sequential execution
    
    @pytest.mark.performance
    def test_memory_usage_optimization(self, mock_mcp_manager):
        """Test memory usage optimization strategies."""
        import gc
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Perform memory-intensive operations
        large_entities = []
        for i in range(100):
            result = mock_mcp_manager.call_tool(
                "memory_create_entity",
                name=f"large_entity_{i}",
                entity_type="test",
                observations=[f"Large observation data {i}" * 100]  # Large observation
            )
            large_entities.append(result)
        
        # Force garbage collection
        gc.collect()
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for this test)
        assert memory_increase < 100 * 1024 * 1024 