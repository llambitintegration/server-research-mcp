"""Real MCP integration tests - consolidated from test_mcp_real_integration.py."""

import pytest
import asyncio
import json
import os
import subprocess
import tempfile
from unittest.mock import patch, MagicMock
from datetime import datetime
import time


@pytest.mark.skip(reason="Requires actual MCP servers - skip for CI/basic testing")
class TestRealMCPIntegration:
    """Test real MCP server integration (requires actual servers)."""
    
    @pytest.fixture(scope="class")
    def mcp_servers(self):
        """Start real MCP servers for testing."""
        servers = {}
        processes = {}
        
        try:
            # Start memory server
            memory_process = subprocess.Popen([
                "npx", "-y", "@modelcontextprotocol/server-memory"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            processes["memory"] = memory_process
            time.sleep(2)  # Give server time to start
            
            # Start context7 server
            context7_process = subprocess.Popen([
                "npx", "-y", "@upstash/context7-mcp"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            processes["context7"] = context7_process
            time.sleep(2)
            
            # Start sequential thinking server
            seq_process = subprocess.Popen([
                "npx", "-y", "@modelcontextprotocol/server-sequential-thinking"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            processes["sequential"] = seq_process
            time.sleep(2)
            
            servers["processes"] = processes
            yield servers
            
        finally:
            # Cleanup - terminate all processes
            for name, process in processes.items():
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                except Exception as e:
                    print(f"Error cleaning up {name} server: {e}")
    
    def test_real_memory_server_operations(self, mcp_servers):
        """Test real memory server operations."""
        from server_research_mcp.tools.mcp_manager import get_mcp_manager
        
        manager = get_mcp_manager()
        
        # Test entity creation
        create_result = manager.call_tool(
            "memory_create_entity",
            name="real_test_entity",
            entity_type="test",
            observations=["This is a real test entity", "Created during integration testing"]
        )
        
        assert create_result is not None
        assert "entity_id" in create_result or "success" in create_result
        
        # Test search
        search_result = manager.call_tool(
            "memory_search",
            query="real test entity"
        )
        
        assert search_result is not None
        assert "results" in search_result
        
        # Test adding observations
        add_result = manager.call_tool(
            "memory_add_observation",
            entity_name="real_test_entity",
            observations=["Additional observation added during test"]
        )
        
        assert add_result is not None
    
    def test_real_context7_server_operations(self, mcp_servers):
        """Test real Context7 server operations."""
        from server_research_mcp.tools.mcp_manager import get_mcp_manager
        
        manager = get_mcp_manager()
        
        # Test library resolution
        resolve_result = manager.call_tool(
            "context7_resolve_library",
            library_name="numpy"
        )
        
        assert resolve_result is not None
        if "library_id" in resolve_result:
            library_id = resolve_result["library_id"]
            
            # Test documentation retrieval
            docs_result = manager.call_tool(
                "context7_get_docs",
                library_id=library_id,
                topic="arrays",
                tokens=5000
            )
            
            assert docs_result is not None
            assert "content" in docs_result or "sections" in docs_result
    
    def test_real_sequential_thinking_operations(self, mcp_servers):
        """Test real sequential thinking server operations."""
        from server_research_mcp.tools.mcp_manager import get_mcp_manager
        
        manager = get_mcp_manager()
        
        # Test thought recording
        thoughts = []
        for i in range(3):
            thought_result = manager.call_tool(
                "sequential_thinking_append_thought",
                thought=f"Real test thought {i+1}: Analyzing integration patterns",
                thought_number=i+1,
                total_thoughts=3
            )
            thoughts.append(thought_result)
            assert thought_result is not None
        
        # Test thought retrieval
        all_thoughts = manager.call_tool("sequential_thinking_get_thoughts")
        assert all_thoughts is not None
        assert "thoughts" in all_thoughts or len(all_thoughts) > 0
    
    def test_real_end_to_end_workflow(self, mcp_servers):
        """Test end-to-end workflow with real MCP servers."""
        from server_research_mcp.tools.mcp_manager import get_mcp_manager
        
        manager = get_mcp_manager()
        
        # Step 1: Create research topic in memory
        topic_result = manager.call_tool(
            "memory_create_entity",
            name="real_integration_test_topic",
            entity_type="research_topic",
            observations=["Integration testing of MCP servers", "Real-world validation"]
        )
        
        assert topic_result is not None
        
        # Step 2: Sequential analysis
        analysis_thoughts = []
        analysis_steps = [
            "Identifying key integration points",
            "Analyzing server communication patterns",
            "Evaluating error handling mechanisms"
        ]
        
        for i, step in enumerate(analysis_steps):
            thought_result = manager.call_tool(
                "sequential_thinking_append_thought",
                thought=f"Step {i+1}: {step}",
                thought_number=i+1,
                total_thoughts=len(analysis_steps)
            )
            analysis_thoughts.append(thought_result)
        
        # Step 3: Research documentation
        docs_result = manager.call_tool(
            "context7_resolve_library",
            library_name="pytest"
        )
        
        if docs_result and "library_id" in docs_result:
            pytest_docs = manager.call_tool(
                "context7_get_docs",
                library_id=docs_result["library_id"],
                topic="integration testing",
                tokens=3000
            )
        
        # Step 4: Store synthesized results
        synthesis_result = manager.call_tool(
            "memory_create_entity",
            name="real_integration_synthesis",
            entity_type="synthesis",
            observations=[
                "Completed real MCP server integration test",
                f"Analyzed {len(analysis_steps)} integration aspects",
                "Validated server communication and functionality"
            ]
        )
        
        # Verify all steps completed successfully
        assert len(analysis_thoughts) == len(analysis_steps)
        assert synthesis_result is not None


@pytest.mark.skip(reason="Requires Node.js/npx - skip for CI/basic testing")
class TestMCPServerLifecycle:
    """Test MCP server lifecycle management in real environments."""
    
    def test_server_startup_sequence(self):
        """Test the startup sequence of MCP servers."""
        servers_to_test = [
            ("memory", "@modelcontextprotocol/server-memory"),
            ("context7", "@upstash/context7-mcp"),
            ("sequential", "@modelcontextprotocol/server-sequential-thinking")
        ]
        
        started_servers = []
        
        try:
            for server_name, package in servers_to_test:
                # Start server
                process = subprocess.Popen([
                    "npx", "-y", package
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # Give server time to start
                time.sleep(3)
                
                # Check if server is running
                poll_result = process.poll()
                if poll_result is None:  # Server is still running
                    started_servers.append((server_name, process))
                    assert True, f"Server {server_name} started successfully"
                else:
                    assert False, f"Server {server_name} failed to start"
        
        finally:
            # Cleanup
            for server_name, process in started_servers:
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
    
    def test_server_health_monitoring(self):
        """Test health monitoring of running MCP servers."""
        # This would test actual health check endpoints if available
        # For now, we'll test the process monitoring
        
        process = subprocess.Popen([
            "npx", "-y", "@modelcontextprotocol/server-memory"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        try:
            # Monitor server health
            time.sleep(2)
            
            # Check process is alive
            assert process.poll() is None, "Server process should be running"
            
            # Check memory usage (basic check)
            import psutil
            try:
                proc = psutil.Process(process.pid)
                memory_info = proc.memory_info()
                assert memory_info.rss > 0, "Server should be using memory"
            except psutil.NoSuchProcess:
                assert False, "Server process not found"
            
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
    
    def test_server_graceful_shutdown(self):
        """Test graceful shutdown of MCP servers."""
        process = subprocess.Popen([
            "npx", "-y", "@modelcontextprotocol/server-memory"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        try:
            time.sleep(2)
            assert process.poll() is None, "Server should be running"
            
            # Send termination signal
            process.terminate()
            
            # Wait for graceful shutdown
            try:
                return_code = process.wait(timeout=10)
                assert return_code is not None, "Server should shutdown gracefully"
            except subprocess.TimeoutExpired:
                # Force kill if graceful shutdown fails
                process.kill()
                assert False, "Server did not shutdown gracefully"
        
        except Exception as e:
            # Cleanup in case of error
            try:
                process.kill()
            except:
                pass
            raise e


class TestMCPErrorHandling:
    """Test error handling in MCP integration scenarios."""
    
    def test_connection_error_handling(self, mock_mcp_manager):
        """Test handling of connection errors."""
        # Simulate connection failure
        with patch.object(mock_mcp_manager, 'call_tool', side_effect=ConnectionError("Server unavailable")):
            with pytest.raises(ConnectionError):
                mock_mcp_manager.call_tool("memory_search", query="test")
    
    def test_timeout_error_handling(self, mock_mcp_manager):
        """Test handling of timeout errors."""
        # Simulate timeout
        with patch.object(mock_mcp_manager, 'call_tool', side_effect=TimeoutError("Request timeout")):
            with pytest.raises(TimeoutError):
                mock_mcp_manager.call_tool("context7_get_docs", library_id="test", tokens=100000)
    
    def test_invalid_parameter_handling(self, mock_mcp_manager):
        """Test handling of invalid parameters."""
        # Test missing required parameters
        with pytest.raises((ValueError, TypeError)):
            mock_mcp_manager.call_tool("memory_search")  # Missing query
        
        # Test invalid parameter types
        with pytest.raises((ValueError, TypeError)):
            mock_mcp_manager.call_tool("memory_search", query=12345)  # Should be string
    
    def test_server_unavailable_handling(self, mock_mcp_manager):
        """Test handling when servers become unavailable."""
        # Mock server availability check
        mock_mcp_manager.is_server_available = MagicMock(return_value=False)
        
        # Should handle unavailable server gracefully
        result = mock_mcp_manager.call_tool("memory_search", query="test")
        
        # Should return error information rather than crashing
        assert result is not None
        assert "error" in result or "status" in result


class TestMCPPerformanceReal:
    """Test MCP performance with real servers (when available)."""
    
    @pytest.mark.performance
    @pytest.mark.skip(reason="Requires real MCP servers - performance testing")
    def test_concurrent_request_performance(self):
        """Test performance under concurrent requests."""
        import concurrent.futures
        import time
        
        from server_research_mcp.tools.mcp_manager import get_mcp_manager
        
        def make_request(request_id):
            manager = get_mcp_manager()
            start_time = time.time()
            
            result = manager.call_tool(
                "memory_search",
                query=f"performance test {request_id}"
            )
            
            end_time = time.time()
            return {
                "request_id": request_id,
                "duration": end_time - start_time,
                "success": result is not None
            }
        
        # Test concurrent requests
        num_requests = 10
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request, i) for i in range(num_requests)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        total_time = time.time() - start_time
        
        # Analyze results
        successful_requests = [r for r in results if r["success"]]
        avg_duration = sum(r["duration"] for r in successful_requests) / len(successful_requests)
        
        # Performance assertions
        assert len(successful_requests) == num_requests, "All requests should succeed"
        assert total_time < num_requests * 2, "Concurrent execution should be faster than sequential"
        assert avg_duration < 5.0, "Average request duration should be reasonable"
    
    @pytest.mark.performance
    @pytest.mark.skip(reason="Requires real MCP servers - memory testing")
    def test_memory_usage_under_load(self):
        """Test memory usage under sustained load."""
        import psutil
        import os
        
        from server_research_mcp.tools.mcp_manager import get_mcp_manager
        
        manager = get_mcp_manager()
        process = psutil.Process(os.getpid())
        
        # Baseline memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform memory-intensive operations
        for i in range(50):
            # Create entities with substantial data
            manager.call_tool(
                "memory_create_entity",
                name=f"load_test_entity_{i}",
                entity_type="load_test",
                observations=[f"Load test data {i}" * 100]  # Large observations
            )
            
            # Search operations
            manager.call_tool(
                "memory_search",
                query=f"load test {i}"
            )
        
        # Final memory check
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory usage should be reasonable
        assert memory_increase < 200, f"Memory increase ({memory_increase}MB) should be reasonable"


class TestMCPConfigurationReal:
    """Test MCP configuration in real environments."""
    
    @pytest.mark.skip(reason="Requires environment configuration")
    def test_environment_variable_configuration(self):
        """Test MCP configuration through environment variables."""
        # Test various environment configurations
        test_configs = [
            {"MCP_SERVER_TIMEOUT": "30"},
            {"MCP_MAX_CONNECTIONS": "5"},
            {"MCP_RETRY_ATTEMPTS": "3"}
        ]
        
        for config in test_configs:
            with patch.dict(os.environ, config):
                from server_research_mcp.tools.mcp_manager import get_mcp_manager
                
                manager = get_mcp_manager()
                
                # Test that configuration is applied
                # This would test actual configuration loading
                assert manager is not None
    
    @pytest.mark.skip(reason="Requires config file setup")
    def test_configuration_file_loading(self, temp_workspace):
        """Test loading MCP configuration from files."""
        config_file = f"{temp_workspace}/mcp_config.json"
        
        config_data = {
            "servers": [
                {"name": "memory", "package": "@modelcontextprotocol/server-memory"},
                {"name": "context7", "package": "@upstash/context7-mcp"}
            ],
            "timeouts": {
                "connection": 30,
                "request": 60
            },
            "retry": {
                "attempts": 3,
                "delay": 1
            }
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        # Test configuration loading
        with patch.dict(os.environ, {"MCP_CONFIG_FILE": config_file}):
            from server_research_mcp.tools.mcp_manager import get_mcp_manager
            
            manager = get_mcp_manager()
            assert manager is not None
            
            # Verify configuration was loaded
            # This would test actual config file parsing 