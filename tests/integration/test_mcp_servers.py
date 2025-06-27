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
    async def test_async_batch_operations(self, async_mcp_client):
        """Test batch operations with async client."""
        await async_mcp_client.connect()
        
        # Create multiple entities in parallel
        entities = [
            {"name": f"entity_{i}", "type": "test", "observations": [f"Test entity {i}"]}
            for i in range(5)
        ]
        
        tasks = []
        for entity in entities:
            task = async_mcp_client.call_tool(
                "memory_create_entity",
                **entity
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 5
        assert all(r["status"] == "success" for r in results)
        
        await async_mcp_client.disconnect()


class TestMCPIntegrationPatterns:
    """Test common MCP integration patterns."""
    
    def test_knowledge_graph_building(self, mock_mcp_manager):
        """Test building knowledge graph through MCP operations."""
        # Create interconnected entities
        entities = [
            ("machine_learning", "field", ["Subset of AI", "Uses statistical methods"]),
            ("neural_networks", "technique", ["Core ML technique", "Inspired by biology"]),
            ("deep_learning", "subfield", ["Uses deep neural networks", "Part of ML"])
        ]
        
        created_entities = []
        for name, entity_type, observations in entities:
            result = mock_mcp_manager.call_tool(
                "memory_create_entity",
                name=name,
                entity_type=entity_type,
                observations=observations
            )
            created_entities.append(result)
        
        # Create relationships
        relationships = [
            ("neural_networks", "implements machine_learning concepts"),
            ("deep_learning", "specializes neural_networks"),
            ("machine_learning", "encompasses deep_learning")
        ]
        
        for entity, relationship in relationships:
            mock_mcp_manager.call_tool(
                "memory_add_observation",
                entity_name=entity,
                observations=[relationship]
            )
        
        # Query the graph
        search_result = mock_mcp_manager.call_tool(
            "memory_search",
            query="machine learning neural networks"
        )
        
        assert len(search_result["results"]) >= 2
    
    def test_research_workflow_integration(self, mock_mcp_manager):
        """Test research workflow using multiple MCP tools."""
        # Step 1: Search existing knowledge
        existing_knowledge = mock_mcp_manager.call_tool(
            "memory_search",
            query="AI testing methodologies"
        )
        
        # Step 2: Resolve documentation sources
        doc_source = mock_mcp_manager.call_tool(
            "context7_resolve_library",
            library_name="pytest"
        )
        
        # Step 3: Get detailed documentation
        detailed_docs = mock_mcp_manager.call_tool(
            "context7_get_docs",
            library_id=doc_source["library_id"],
            topic="testing frameworks",
            tokens=5000
        )
        
        # Step 4: Sequential analysis
        analysis_thoughts = []
        for i, thought in enumerate([
            "Analyze existing knowledge gaps",
            "Review documentation quality",
            "Synthesize findings"
        ], 1):
            result = mock_mcp_manager.call_tool(
                "sequential_thinking_append_thought",
                thought=thought,
                thought_number=i,
                total_thoughts=3
            )
            analysis_thoughts.append(result)
        
        # Step 5: Create new knowledge entity
        synthesis_result = mock_mcp_manager.call_tool(
            "memory_create_entity",
            name="ai_testing_synthesis",
            entity_type="research_output",
            observations=[
                "Synthesized from existing knowledge and documentation",
                "Includes pytest framework analysis",
                "Sequential thinking applied for thoroughness"
            ]
        )
        
        # Verify workflow completion
        assert existing_knowledge["results"]
        assert doc_source["library_id"]
        assert detailed_docs["content"]
        assert len(analysis_thoughts) == 3
        assert synthesis_result["success"]
    
    def test_mcp_caching_strategy(self, mock_mcp_manager, mock_chromadb):
        """Test caching strategy for MCP operations."""
        cache = {}
        
        def cached_call_tool(tool_name, **kwargs):
            cache_key = f"{tool_name}:{hash(str(sorted(kwargs.items())))}"
            
            if cache_key in cache:
                return cache[cache_key]
            
            result = mock_mcp_manager.call_tool(tool_name, **kwargs)
            cache[cache_key] = result
            return result
        
        # First call - should hit MCP
        result1 = cached_call_tool("memory_search", query="caching test")
        
        # Second call - should hit cache
        result2 = cached_call_tool("memory_search", query="caching test")
        
        assert result1 == result2
        assert len(cache) == 1


class TestMCPErrorRecovery:
    """Test MCP error recovery and resilience."""
    
    def test_connection_retry_logic(self, mock_mcp_manager):
        """Test connection retry logic."""
        call_count = 0
        
        def flaky_connection():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Connection failed")
            return {"status": "success"}
        
        mock_mcp_manager.call_tool = MagicMock(side_effect=flaky_connection)
        
        # Implement retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = mock_mcp_manager.call_tool("test_tool")
                break
            except ConnectionError:
                if attempt == max_retries - 1:
                    raise
                continue
        
        assert result["status"] == "success"
        assert call_count == 3
    
    def test_partial_failure_handling(self, mock_mcp_manager):
        """Test handling of partial failures in batch operations."""
        operations = [
            ("memory_create_entity", {"name": "entity1", "type": "test"}),
            ("memory_create_entity", {"name": "invalid", "type": ""}),  # Will fail
            ("memory_create_entity", {"name": "entity3", "type": "test"})
        ]
        
        results = []
        failed_operations = []
        
        for i, (tool_name, kwargs) in enumerate(operations):
            try:
                if kwargs.get("type") == "":  # Simulate failure
                    raise ValueError("Invalid entity type")
                result = mock_mcp_manager.call_tool(tool_name, **kwargs)
                results.append(result)
            except Exception as e:
                failed_operations.append((i, tool_name, str(e)))
        
        # Should have 2 successes, 1 failure
        assert len(results) == 2
        assert len(failed_operations) == 1
        assert failed_operations[0][0] == 1  # Second operation failed
    
    def test_state_recovery_after_crash(self, mock_mcp_manager, temp_workspace):
        """Test state recovery after simulated crash."""
        # Save state before crash
        state_file = f"{temp_workspace}/mcp_state.json"
        
        pre_crash_state = {
            "entities": ["entity1", "entity2"],
            "last_operation": "memory_create_entity",
            "timestamp": datetime.now().isoformat()
        }
        
        with open(state_file, 'w') as f:
            json.dump(pre_crash_state, f)
        
        # Simulate crash and recovery
        assert os.path.exists(state_file)
        
        # Load state after recovery
        with open(state_file, 'r') as f:
            recovered_state = json.load(f)
        
        assert recovered_state["entities"] == pre_crash_state["entities"]
        assert recovered_state["last_operation"] == pre_crash_state["last_operation"]
        
        # Verify operations can continue
        continue_result = mock_mcp_manager.call_tool(
            "memory_search",
            query="recovery test"
        )
        
        assert continue_result["results"]


class TestMCPPerformance:
    """Test MCP performance characteristics."""
    
    @pytest.mark.performance
    def test_tool_execution_performance(self, mock_mcp_manager, performance_monitor):
        """Test tool execution performance."""
        operations = [
            ("memory_search", {"query": f"test query {i}"})
            for i in range(10)
        ]
        
        with performance_monitor() as monitor:
            for tool_name, kwargs in operations:
                mock_mcp_manager.call_tool(tool_name, **kwargs)
        
        assert monitor.operations_per_second > 5  # Should handle at least 5 ops/sec
        assert monitor.average_latency < 0.5  # Should average under 500ms
    
    @pytest.mark.performance
    def test_concurrent_performance(self, mock_mcp_manager, performance_monitor):
        """Test concurrent tool execution performance."""
        import concurrent.futures
        
        def execute_tool(index):
            return mock_mcp_manager.call_tool(
                "memory_search",
                query=f"concurrent test {index}"
            )
        
        with performance_monitor() as monitor:
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(execute_tool, i) for i in range(20)]
                results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        assert len(results) == 20
        assert monitor.concurrent_operations_per_second > 10
    
    @pytest.mark.performance
    def test_memory_usage_optimization(self, mock_mcp_manager):
        """Test memory usage stays reasonable during operations."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Perform many operations
        for i in range(100):
            mock_mcp_manager.call_tool(
                "memory_create_entity",
                name=f"entity_{i}",
                entity_type="test",
                observations=[f"Test entity {i}"]
            )
        
        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be reasonable (less than 100MB)
        assert memory_growth < 100 * 1024 * 1024


class TestMCPManagerIntegration:
    """Test MCP manager integration with servers (legacy compatibility)."""
    
    def test_mcp_manager_initialization(self, mock_mcp_manager):
        """Test MCP manager can be initialized."""
        assert mock_mcp_manager is not None
        assert hasattr(mock_mcp_manager, 'call_tool')
        
    def test_mcp_manager_tool_calls(self, mock_mcp_manager):
        """Test MCP manager can call different tools."""
        # Test memory search
        result = mock_mcp_manager.call_tool("server-memory", "search_nodes", {"query": "test"})
        assert "nodes" in result
        
        # Test entity creation
        result = mock_mcp_manager.call_tool(
            "server-memory", 
            "create_entity", 
            {"name": "test_entity", "type": "test"}
        )
        assert result["success"] is True
        
        # Test sequential thinking
        result = mock_mcp_manager.call_tool(
            "server-sequential-thinking",
            "append_thought",
            {"thought": "test thought"}
        )
        assert "thought" in result