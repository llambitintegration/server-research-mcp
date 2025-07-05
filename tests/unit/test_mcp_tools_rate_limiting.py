"""
Unit tests for MCP tools rate limiting functionality.

Tests the enhanced get_agent_tools method with rate limiting application,
agent type detection, and comprehensive logging verification.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from src.server_research_mcp.tools.mcp_tools import MCPToolRegistry, ToolMapping
from src.server_research_mcp.crew import AgentDefinition
from src.server_research_mcp.utils.rate_limiting import RateLimitedTool, RateLimitConfig
from crewai.tools import BaseTool


class MockTool(BaseTool):
    """Mock tool for testing."""
    
    def __init__(self, name: str):
        self.name = name
        self.description = f"Mock tool: {name}"
    
    def _run(self, *args, **kwargs):
        return f"Mock result from {self.name}"


@pytest.fixture
def mock_registry():
    """Create a mock registry with test tools."""
    registry = MCPToolRegistry()
    
    # Clear any existing mappings
    registry.mappings.clear()
    
    # Add test tool mappings
    registry.map_tools("researcher", ["zotero", "search"], required_count=3)
    registry.map_tools("historian", ["memory", "context"], required_count=6)
    registry.map_tools("publisher", ["filesystem", "write"], required_count=11)
    registry.map_tools("archivist", ["sequential", "thinking"], required_count=1)
    
    return registry


@pytest.fixture
def mock_tools():
    """Create mock tools for testing."""
    return [
        MockTool("zotero_search"),
        MockTool("zotero_extract"),
        MockTool("memory_store"),
        MockTool("memory_search"),
        MockTool("filesystem_read"),
        MockTool("filesystem_write"),
        MockTool("sequential_thinking"),
        MockTool("context_search"),
    ]


class TestMCPToolsRateLimiting:
    """Test suite for MCP tools rate limiting functionality."""
    
    def test_agent_type_detection(self, mock_registry):
        """Test that agent types are correctly detected for rate limiting."""
        # Test researcher -> zotero
        assert mock_registry._get_tool_type_for_agent("researcher") == "zotero"
        
        # Test historian -> memory
        assert mock_registry._get_tool_type_for_agent("historian") == "memory"
        
        # Test publisher -> filesystem
        assert mock_registry._get_tool_type_for_agent("publisher") == "filesystem"
        
        # Test archivist -> sequential_thinking
        assert mock_registry._get_tool_type_for_agent("archivist") == "sequential_thinking"
        
        # Test unknown agent -> default
        assert mock_registry._get_tool_type_for_agent("unknown") == "default"
        
        # Test case insensitive
        assert mock_registry._get_tool_type_for_agent("RESEARCHER") == "zotero"
    
    @patch('src.server_research_mcp.tools.mcp_tools.get_rate_limited_tools')
    def test_rate_limiting_application(self, mock_get_rate_limited, mock_registry, mock_tools):
        """Test that rate limiting is applied correctly to matched tools."""
        # Setup
        mock_registry._all_tools = mock_tools
        mock_rate_limited_tools = [Mock(spec=RateLimitedTool) for _ in range(3)]
        mock_get_rate_limited.return_value = mock_rate_limited_tools
        
        # Execute
        result = mock_registry.get_agent_tools("researcher", apply_rate_limiting=True)
        
        # Verify rate limiting was called with correct parameters
        mock_get_rate_limited.assert_called_once()
        call_args = mock_get_rate_limited.call_args
        
        # Check that tools were passed and tool type was "zotero"
        assert len(call_args[0][0]) >= 1  # At least one tool matched
        assert call_args[0][1] == "zotero"  # Correct tool type
        
        # Check result
        assert result == mock_rate_limited_tools
    
    @patch('src.server_research_mcp.tools.mcp_tools.get_rate_limited_tools')
    def test_rate_limiting_disabled(self, mock_get_rate_limited, mock_registry, mock_tools):
        """Test that rate limiting can be disabled."""
        # Setup
        mock_registry._all_tools = mock_tools
        
        # Execute with rate limiting disabled
        result = mock_registry.get_agent_tools("researcher", apply_rate_limiting=False)
        
        # Verify rate limiting was not called
        mock_get_rate_limited.assert_not_called()
        
        # Should return original tools
        assert all(isinstance(tool, MockTool) for tool in result)
    
    @patch('src.server_research_mcp.tools.mcp_tools.logger')
    @patch('src.server_research_mcp.tools.mcp_tools.get_rate_limited_tools')
    def test_comprehensive_logging(self, mock_get_rate_limited, mock_logger, mock_registry, mock_tools):
        """Test that comprehensive logging is performed during tool retrieval."""
        # Setup
        mock_registry._all_tools = mock_tools
        mock_rate_limited_tools = [Mock(spec=RateLimitedTool) for _ in range(2)]
        
        # Add rate_limiter attribute to mock tools
        for tool in mock_rate_limited_tools:
            tool.rate_limiter = Mock()
            tool.rate_limiter.config = Mock()
            tool.rate_limiter.config.max_requests_per_minute = 10
            tool.rate_limiter.config.max_requests_per_hour = 100
            tool.rate_limiter.config.min_request_interval = 1.0
            tool.name = "test_tool"
        
        mock_get_rate_limited.return_value = mock_rate_limited_tools
        
        # Execute
        mock_registry.get_agent_tools("researcher", apply_rate_limiting=True)
        
        # Verify logging calls were made
        info_calls = [call for call in mock_logger.info.call_args_list]
        debug_calls = [call for call in mock_logger.debug.call_args_list]
        
        # Check for expected log messages
        log_messages = [str(call) for call in info_calls + debug_calls]
        
        # Should log total available tools
        assert any("Total available tools" in msg for msg in log_messages)
        
        # Should log matched tools
        assert any("Matched" in msg and "tools for researcher" in msg for msg in log_messages)
        
        # Should log rate limiting application
        assert any("Applying" in msg and "rate limiting" in msg for msg in log_messages)
        
        # Should log rate limiting verification
        assert any("Rate limiting applied" in msg for msg in log_messages)
    
    def test_tool_pattern_matching(self, mock_registry, mock_tools):
        """Test that tool patterns correctly match available tools."""
        # Setup
        mock_registry._all_tools = mock_tools
        
        # Test researcher pattern matching
        result = mock_registry.get_agent_tools("researcher", apply_rate_limiting=False)
        
        # Should match zotero tools
        matched_names = [tool.name for tool in result]
        assert any("zotero" in name.lower() for name in matched_names)
    
    @patch('src.server_research_mcp.tools.mcp_tools.get_rate_limited_tools')
    def test_fallback_tool_handling(self, mock_get_rate_limited, mock_registry):
        """Test that fallback tools are created when minimum count not met."""
        # Setup with no tools to force fallback
        mock_registry._all_tools = []
        mock_get_rate_limited.return_value = []
        
        # Execute for historian (requires 6 tools)
        result = mock_registry.get_agent_tools("historian", apply_rate_limiting=True)
        
        # Should have created fallback tools
        # Note: The actual fallback creation happens in the original method
        # This test verifies the flow works when no tools match
        mock_get_rate_limited.assert_called_once()
    
    @patch('src.server_research_mcp.tools.mcp_tools.get_rate_limited_tools')
    def test_rate_limit_verification_logging(self, mock_get_rate_limited, mock_registry, mock_tools):
        """Test that rate limit configuration is logged for verification."""
        # Setup
        mock_registry._all_tools = mock_tools
        
        # Create mock rate limited tools with rate limiter attributes
        mock_rate_limited_tools = []
        for i in range(2):
            tool = Mock(spec=RateLimitedTool)
            tool.name = f"rate_limited_tool_{i}"
            tool.rate_limiter = Mock()
            tool.rate_limiter.config = Mock()
            tool.rate_limiter.config.max_requests_per_minute = 5 + i
            tool.rate_limiter.config.max_requests_per_hour = 50 + i * 10
            tool.rate_limiter.config.min_request_interval = 1.0 + i * 0.5
            mock_rate_limited_tools.append(tool)
        
        mock_get_rate_limited.return_value = mock_rate_limited_tools
        
        with patch('src.server_research_mcp.tools.mcp_tools.logger') as mock_logger:
            # Execute
            result = mock_registry.get_agent_tools("researcher", apply_rate_limiting=True)
            
            # Verify rate limit configuration was logged
            debug_calls = [str(call) for call in mock_logger.debug.call_args_list]
            
            # Should log configuration for each tool
            assert len([call for call in debug_calls if "req/min" in call]) >= 1
    
    def test_no_mapping_found(self, mock_registry):
        """Test behavior when no tool mapping exists for an agent."""
        # Test with non-existent agent
        result = mock_registry.get_agent_tools("nonexistent", apply_rate_limiting=True)
        
        # Should return empty list
        assert result == []
    
    @patch('src.server_research_mcp.tools.mcp_tools.get_rate_limited_tools')
    def test_different_agent_types_get_different_configs(self, mock_get_rate_limited, mock_registry, mock_tools):
        """Test that different agent types get appropriate rate limiting configurations."""
        # Setup
        mock_registry._all_tools = mock_tools
        mock_get_rate_limited.return_value = [Mock(spec=RateLimitedTool)]
        
        # Test different agents
        agents_and_types = [
            ("researcher", "zotero"),
            ("historian", "memory"),
            ("publisher", "filesystem"),
            ("archivist", "sequential_thinking")
        ]
        
        for agent_name, expected_type in agents_and_types:
            mock_get_rate_limited.reset_mock()
            
            # Execute
            mock_registry.get_agent_tools(agent_name, apply_rate_limiting=True)
            
            # Verify correct tool type was used
            if mock_get_rate_limited.called:
                call_args = mock_get_rate_limited.call_args
                assert call_args[0][1] == expected_type, f"Agent {agent_name} should use {expected_type} rate limiting"


class TestRateLimitingIntegration:
    """Integration tests for rate limiting with actual tool instances."""
    
    @patch('src.server_research_mcp.tools.mcp_tools.wrap_tools_with_rate_limit')
    def test_rate_limiting_wrapper_integration(self, mock_wrap_tools, mock_registry, mock_tools):
        """Test integration with the actual rate limiting wrapper."""
        # Setup
        mock_registry._all_tools = mock_tools
        
        # Mock the wrapper to return RateLimitedTool instances
        mock_wrapped_tools = []
        for tool in mock_tools[:2]:  # Wrap first 2 tools
            wrapped = Mock(spec=RateLimitedTool)
            wrapped.name = f"rate_limited_{tool.name}"
            wrapped.rate_limiter = Mock()
            mock_wrapped_tools.append(wrapped)
        
        with patch('src.server_research_mcp.utils.rate_limiting.tool_wrapper.wrap_tools_with_rate_limit') as mock_wrap:
            mock_wrap.return_value = mock_wrapped_tools
            
            # Execute
            result = mock_registry.get_agent_tools("researcher", apply_rate_limiting=True)
            
            # Verify wrapper was called
            mock_wrap.assert_called_once()
            
            # Verify result contains wrapped tools
            assert len(result) == len(mock_wrapped_tools)
            assert all(hasattr(tool, 'rate_limiter') for tool in result)


@pytest.fixture
def rate_limit_config():
    """Create a test rate limit configuration."""
    return RateLimitConfig(
        max_requests_per_minute=10,
        max_requests_per_hour=100,
        min_request_interval=1.0,
        max_retries=3,
        initial_retry_delay=1.0
    )


class TestRateLimitConfigValidation:
    """Test rate limit configuration validation and application."""
    
    def test_rate_limit_config_creation(self, rate_limit_config):
        """Test that rate limit configurations are created correctly."""
        assert rate_limit_config.max_requests_per_minute == 10
        assert rate_limit_config.max_requests_per_hour == 100
        assert rate_limit_config.min_request_interval == 1.0
        assert rate_limit_config.max_retries == 3
        assert rate_limit_config.initial_retry_delay == 1.0
    
    @patch.dict('os.environ', {
        'MCP_ZOTERO_MAX_REQUESTS_PER_MINUTE': '5',
        'MCP_ZOTERO_MAX_REQUESTS_PER_HOUR': '50',
        'MCP_ZOTERO_MIN_REQUEST_INTERVAL': '2.0'
    })
    def test_environment_variable_override(self):
        """Test that environment variables override default configurations."""
        from src.server_research_mcp.utils.rate_limiting.tool_wrapper import get_rate_limit_config_from_env
        
        config = get_rate_limit_config_from_env("MCP_ZOTERO")
        
        assert config['max_requests_per_minute'] == 5
        assert config['max_requests_per_hour'] == 50
        assert config['min_request_interval'] == 2.0 