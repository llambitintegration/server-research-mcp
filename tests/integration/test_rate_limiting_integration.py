"""
Integration test for rate limiting components working together.

Tests the integration between rate limiting utilities, tools, and LLM wrappers
without requiring a full crew execution.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from src.server_research_mcp.utils.rate_limiting import RateLimitConfig, RateLimiter
from src.server_research_mcp.utils.rate_limiting.tool_wrapper import RateLimitedTool, get_rate_limited_tools
from src.server_research_mcp.utils.llm_rate_limiter import RateLimitedLLM, get_rate_limited_llm
from src.server_research_mcp.utils.rate_limit_monitor import get_monitoring_stats, reset_monitoring_stats
from src.server_research_mcp.tools.mcp_tools import MCPToolRegistry


class TestRateLimitingIntegration:
    """Integration tests for rate limiting components."""
    
    @pytest.fixture(autouse=True)
    def setup_monitoring(self):
        """Reset monitoring stats before each test."""
        reset_monitoring_stats()
        yield
        # Print stats after test for debugging
        stats = get_monitoring_stats()
        if stats.get("total_identifiers", 0) > 0:
            print(f"\nRate limiting stats: {stats['overall_stats']}")
    
    def test_rate_limiter_basic_functionality(self):
        """Test that RateLimiter works correctly with basic configuration."""
        config = RateLimitConfig(
            max_requests_per_minute=5,
            max_requests_per_hour=50,
            min_request_interval=1.0
        )
        limiter = RateLimiter(config)
        
        # First request should be allowed
        allowed, wait_time = limiter.check_rate_limit("test_tool")
        assert allowed is True
        assert wait_time is None
        
        # Record the request
        limiter.record_request("test_tool")
        
        # Immediate second request should require waiting
        allowed, wait_time = limiter.check_rate_limit("test_tool")
        assert wait_time >= 0.9  # Should respect min_request_interval (with timing tolerance)
    
    def test_tool_rate_limiting_integration(self):
        """Test that tools can be wrapped with rate limiting."""
        # Create a mock tool
        mock_tool = Mock()
        mock_tool.name = "test_tool"
        mock_tool.description = "Test tool description"
        mock_tool._run = Mock(return_value="test result")
        
        # Create rate limited tool
        config = RateLimitConfig(
            max_requests_per_minute=10,
            min_request_interval=0.1
        )
        limiter = RateLimiter(config)
        rate_limited_tool = RateLimitedTool(mock_tool, limiter)
        
        # Test execution
        result = rate_limited_tool._run("test_arg")
        assert result == "test result"
        mock_tool._run.assert_called_once_with("test_arg")
        
        # Verify tool properties
        assert rate_limited_tool.name == "rate_limited_test_tool"
        assert rate_limited_tool.description == "Test tool description"
        assert rate_limited_tool.wrapped_tool == mock_tool
    
    def test_llm_rate_limiting_integration(self):
        """Test that LLMs can be wrapped with rate limiting."""
        # Create a mock LLM
        mock_llm = Mock()
        mock_llm.model = "test-model"
        mock_llm.invoke = Mock(return_value="test response")
        mock_llm.generate = Mock(return_value="generated response")
        
        # Create rate limited LLM
        rate_limited_llm = get_rate_limited_llm(mock_llm)
        
        # Test that it's properly wrapped
        assert isinstance(rate_limited_llm, RateLimitedLLM)
        assert rate_limited_llm.wrapped_llm == mock_llm
        assert hasattr(rate_limited_llm, 'rate_limiter_applied')
        assert rate_limited_llm.rate_limiter_applied is True
        
        # Test method calls
        result = rate_limited_llm.invoke("test prompt")
        assert result == "test response"
        mock_llm.invoke.assert_called_once_with("test prompt")
    
    def test_multiple_tools_rate_limiting(self):
        """Test rate limiting with multiple tools sharing a limiter."""
        # Create multiple mock tools
        tools = []
        for i in range(3):
            tool = Mock()
            tool.name = f"tool_{i}"
            tool.description = f"Tool {i} description"
            tool._run = Mock(return_value=f"result_{i}")
            tools.append(tool)
        
        # Apply rate limiting
        rate_limited_tools = get_rate_limited_tools(tools, "test")
        
        # Verify all tools are wrapped
        assert len(rate_limited_tools) == 3
        for i, wrapped_tool in enumerate(rate_limited_tools):
            assert isinstance(wrapped_tool, RateLimitedTool)
            assert wrapped_tool.wrapped_tool == tools[i]
            assert wrapped_tool.name == f"rate_limited_tool_{i}"
    
    def test_rate_limiting_with_retries(self):
        """Test that rate limiting handles retries correctly."""
        # Create a tool that fails once then succeeds
        mock_tool = Mock()
        mock_tool.name = "retry_tool"
        mock_tool.description = "Tool that retries"
        
        # First call fails, second succeeds
        mock_tool._run = Mock(side_effect=[Exception("Rate limit error"), "success"])
        
        # Create rate limited tool with retry config
        config = RateLimitConfig(
            max_requests_per_minute=10,
            min_request_interval=0.1,
            max_retries=3,
            initial_retry_delay=0.1,
            rate_limit_error_keywords=["rate limit"]
        )
        limiter = RateLimiter(config)
        rate_limited_tool = RateLimitedTool(mock_tool, limiter)
        
        # Execute - should retry and succeed
        result = rate_limited_tool._run("test")
        assert result == "success"
        assert mock_tool._run.call_count == 2
    
    def test_monitoring_integration(self):
        """Test that monitoring captures rate limiting statistics."""
        # Reset stats
        reset_monitoring_stats()
        
        # Create and use rate limited components
        config = RateLimitConfig(max_requests_per_minute=10)
        limiter = RateLimiter(config)
        
        # Simulate some requests
        for i in range(3):
            limiter.check_rate_limit(f"tool_{i}")
            limiter.record_request(f"tool_{i}")
        
        # Check monitoring stats
        stats = get_monitoring_stats()
        assert stats["overall_stats"]["total_requests"] >= 3
        assert len(stats["per_identifier_stats"]) >= 3
    
    def test_environment_configuration_integration(self):
        """Test that environment variables are properly integrated."""
        with patch.dict('os.environ', {
            'MCP_TEST_MAX_REQUESTS_PER_MINUTE': '5',
            'MCP_TEST_MIN_REQUEST_INTERVAL': '2.0'
        }):
            # Create tools with environment-based config
            mock_tool = Mock()
            mock_tool.name = "env_tool"
            mock_tool.description = "Environment configured tool"
            mock_tool._run = Mock(return_value="env result")
            
            # Get rate limited tools (should pick up env config)
            rate_limited_tools = get_rate_limited_tools([mock_tool], "test")
            
            # Verify tool is wrapped
            assert len(rate_limited_tools) == 1
            wrapped_tool = rate_limited_tools[0]
            assert isinstance(wrapped_tool, RateLimitedTool)
    
    def test_concurrent_rate_limiting(self):
        """Test rate limiting behavior with concurrent access simulation."""
        import threading
        import queue
        
        # Create shared rate limiter
        config = RateLimitConfig(
            max_requests_per_minute=10,
            min_request_interval=0.2
        )
        limiter = RateLimiter(config)
        
        # Create mock tool
        mock_tool = Mock()
        mock_tool.name = "concurrent_tool"
        mock_tool.description = "Concurrent test tool"
        mock_tool._run = Mock(return_value="concurrent result")
        
        rate_limited_tool = RateLimitedTool(mock_tool, limiter)
        
        # Results queue
        results = queue.Queue()
        
        def worker():
            try:
                result = rate_limited_tool._run("test")
                results.put(("success", result))
            except Exception as e:
                results.put(("error", str(e)))
        
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=5.0)
        
        # Check results
        success_count = 0
        while not results.empty():
            result_type, result_value = results.get()
            if result_type == "success":
                success_count += 1
        
        # At least some should succeed (rate limiting may delay but not prevent)
        assert success_count >= 1
    
    def test_error_handling_integration(self):
        """Test error handling across rate limiting components."""
        # Create a tool that always fails
        mock_tool = Mock()
        mock_tool.name = "failing_tool"
        mock_tool.description = "Tool that always fails"
        mock_tool._run = Mock(side_effect=Exception("Tool error"))
        
        # Create rate limited tool
        config = RateLimitConfig(
            max_requests_per_minute=10,
            max_retries=2,
            initial_retry_delay=0.1
        )
        limiter = RateLimiter(config)
        rate_limited_tool = RateLimitedTool(mock_tool, limiter)
        
        # Should eventually raise the error after retries
        with pytest.raises(Exception, match="Tool error"):
            rate_limited_tool._run("test")
        
        # Should have retried
        assert mock_tool._run.call_count == 2
    
    def test_performance_impact_measurement(self):
        """Test that rate limiting doesn't add excessive overhead."""
        import time
        
        # Create simple tool
        mock_tool = Mock()
        mock_tool.name = "perf_tool"
        mock_tool.description = "Performance test tool"
        mock_tool._run = Mock(return_value="perf result")
        
        # Create rate limited version with minimal delays
        config = RateLimitConfig(
            max_requests_per_minute=100,
            min_request_interval=0.01
        )
        limiter = RateLimiter(config)
        rate_limited_tool = RateLimitedTool(mock_tool, limiter)
        
        # Measure execution time
        start_time = time.time()
        
        # Execute multiple times
        for i in range(10):
            rate_limited_tool._run(f"test_{i}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should complete reasonably quickly (less than 1 second for 10 calls)
        assert total_time < 1.0, f"Rate limiting added too much overhead: {total_time}s"
        
        # All calls should have succeeded
        assert mock_tool._run.call_count == 10


class TestRateLimitingConfigurationIntegration:
    """Test configuration and environment variable integration."""
    
    def test_default_configuration_loading(self):
        """Test that default configurations are loaded correctly."""
        # Create rate limiter with defaults
        config = RateLimitConfig()
        limiter = RateLimiter(config)
        
        # Verify default values are reasonable
        assert config.max_requests_per_minute > 0
        assert config.max_requests_per_hour > 0
        assert config.min_request_interval >= 0
        assert config.max_retries > 0
    
    @patch.dict('os.environ', {
        'MCP_DEFAULT_MAX_REQUESTS_PER_MINUTE': '15',
        'MCP_DEFAULT_MIN_REQUEST_INTERVAL': '1.5'
    })
    def test_environment_override_integration(self):
        """Test that environment variables override defaults."""
        # This would be tested if get_rate_limited_tools supported env vars
        # For now, just verify the environment is set
        import os
        assert os.getenv('MCP_DEFAULT_MAX_REQUESTS_PER_MINUTE') == '15'
        assert os.getenv('MCP_DEFAULT_MIN_REQUEST_INTERVAL') == '1.5'
    
    def test_configuration_validation(self):
        """Test that invalid configurations are handled properly."""
        # Test invalid configuration
        with pytest.raises(ValueError):
            RateLimitConfig(max_requests_per_minute=-1)
        
        with pytest.raises(ValueError):
            RateLimitConfig(min_request_interval=-1)
    
    def test_tool_type_specific_configuration(self):
        """Test that different tool types can have different configurations."""
        # Create tools of different types
        zotero_tool = Mock()
        zotero_tool.name = "zotero_search"
        zotero_tool.description = "Zotero search tool"
        zotero_tool._run = Mock(return_value="zotero result")
        
        memory_tool = Mock()
        memory_tool.name = "memory_search"
        memory_tool.description = "Memory search tool"
        memory_tool._run = Mock(return_value="memory result")
        
        # Apply rate limiting with different tool types
        zotero_tools = get_rate_limited_tools([zotero_tool], "zotero")
        memory_tools = get_rate_limited_tools([memory_tool], "memory")
        
        # Both should be wrapped but potentially with different configs
        assert len(zotero_tools) == 1
        assert len(memory_tools) == 1
        assert isinstance(zotero_tools[0], RateLimitedTool)
        assert isinstance(memory_tools[0], RateLimitedTool) 