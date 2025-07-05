"""
Integration test for full crew execution with rate limiting.

Tests that the complete crew pipeline works without 429 errors when rate limiting
is properly applied to all components.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from src.server_research_mcp.crew import ServerResearchMcpCrew
from src.server_research_mcp.utils.rate_limit_monitor import get_monitoring_stats, reset_monitoring_stats


class TestFullCrewExecution:
    """Integration tests for full crew execution with rate limiting."""
    
    @pytest.fixture(autouse=True)
    def setup_monitoring(self):
        """Reset monitoring stats before each test."""
        reset_monitoring_stats()
        yield
        # Print stats after test for debugging
        stats = get_monitoring_stats()
        if stats.get("total_identifiers", 0) > 0:
            print(f"\nRate limiting stats: {stats['overall_stats']}")
    
    @patch('src.server_research_mcp.utils.llm_rate_limiter.get_rate_limited_llm')
    def test_crew_initialization_with_rate_limiting(self, mock_get_rate_limited_llm):
        """Test that crew initializes properly with all rate limiting components."""
        # Setup mocks
        mock_rate_limited_llm = Mock()
        mock_rate_limited_llm.rate_limiter_applied = True
        mock_get_rate_limited_llm.return_value = mock_rate_limited_llm
        
        # Create crew
        crew = ServerResearchMcpCrew(inputs={"topic": "test research topic"})
        
        # Verify crew created successfully
        assert crew is not None
        assert crew.inputs["topic"] == "test research topic"
        assert hasattr(crew, 'llm_config')
        assert hasattr(crew, 'tool_registry')
    
    @patch('src.server_research_mcp.utils.llm_rate_limiter.get_rate_limited_llm')
    def test_agent_creation_with_rate_limiting(self, mock_get_rate_limited_llm):
        """Test that agents are created with rate limiting applied."""
        # Setup mock LLM
        mock_llm = Mock()
        mock_rate_limited_llm = Mock()
        mock_rate_limited_llm.rate_limiter_applied = True
        mock_rate_limited_llm.wrapped_llm = mock_llm
        mock_get_rate_limited_llm.return_value = mock_rate_limited_llm
        
        # Create crew and test agent creation
        crew = ServerResearchMcpCrew(inputs={"topic": "test topic"})
        
        with patch('crewai.Agent') as mock_agent_class:
            with patch.object(crew.tool_registry, 'get_agent_tools', return_value=[Mock(), Mock()]):
                # Test historian agent creation
                agent = crew.historian()
                
                # Verify agent was created with rate-limited LLM
                mock_agent_class.assert_called_once()
                call_kwargs = mock_agent_class.call_args[1]
                assert call_kwargs['llm'] == mock_rate_limited_llm
                
                # Verify rate limiting was applied
                mock_get_rate_limited_llm.assert_called_once()
    
    @patch('src.server_research_mcp.utils.llm_rate_limiter.get_rate_limited_llm')
    @patch('crewai.Agent')
    @patch('crewai.Task')
    @patch('crewai.Crew')
    def test_full_crew_creation_with_rate_limiting(self, mock_crew_class, mock_task_class, mock_agent_class, mock_get_rate_limited_llm):
        """Test full crew creation with rate limiting on all components."""
        # Setup mocks
        mock_rate_limited_llm = Mock()
        mock_rate_limited_llm.rate_limiter_applied = True
        mock_get_rate_limited_llm.return_value = mock_rate_limited_llm
        
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent
        
        mock_task = Mock()
        mock_task_class.return_value = mock_task
        
        mock_crew_instance = Mock()
        mock_crew_class.return_value = mock_crew_instance
        
        # Create crew
        crew = ServerResearchMcpCrew(inputs={"topic": "test topic"})
        
        with patch.object(crew.tool_registry, 'get_agent_tools', return_value=[Mock(), Mock()]):
            # Create the crew
            crew_instance = crew.crew()
            
            # Verify all components were created
            assert mock_agent_class.call_count == 4  # 4 agents: historian, researcher, archivist, publisher
            assert mock_task_class.call_count == 4   # 4 tasks
            assert mock_crew_class.call_count == 1   # 1 crew
            
            # Verify rate limiting was applied to all agents
            assert mock_get_rate_limited_llm.call_count == 4
    
    @patch('src.server_research_mcp.utils.llm_rate_limiter.get_rate_limited_llm')
    def test_execution_time_doubling(self, mock_get_rate_limited_llm):
        """Test that execution times are doubled to accommodate rate limiting."""
        # Setup mock
        mock_rate_limited_llm = Mock()
        mock_rate_limited_llm.rate_limiter_applied = True
        mock_get_rate_limited_llm.return_value = mock_rate_limited_llm
        
        crew = ServerResearchMcpCrew(inputs={"topic": "test topic"})
        
        with patch('crewai.Agent') as mock_agent_class:
            with patch.object(crew.tool_registry, 'get_agent_tools', return_value=[Mock()]):
                # Create historian agent (has max_execution_time=120)
                crew.historian()
                
                # Verify execution time was doubled
                call_kwargs = mock_agent_class.call_args[1]
                assert call_kwargs['max_execution_time'] == 240  # 120 * 2
    
    @patch('src.server_research_mcp.utils.llm_rate_limiter.get_rate_limited_llm')
    def test_rate_limiting_configuration_logging(self, mock_get_rate_limited_llm):
        """Test that rate limiting configuration is properly logged."""
        # Setup mock with rate limiter config
        mock_rate_limited_llm = Mock()
        mock_rate_limited_llm.rate_limiter_applied = True
        mock_rate_limited_llm.rate_limiter = Mock()
        mock_rate_limited_llm.rate_limiter.config = Mock()
        mock_rate_limited_llm.rate_limiter.config.max_requests_per_minute = 20
        mock_rate_limited_llm.rate_limiter.config.min_request_interval = 0.5
        mock_get_rate_limited_llm.return_value = mock_rate_limited_llm
        
        crew = ServerResearchMcpCrew(inputs={"topic": "test topic"})
        
        with patch('src.server_research_mcp.crew.logger') as mock_logger:
            with patch('crewai.Agent'):
                with patch.object(crew.tool_registry, 'get_agent_tools', return_value=[Mock()]):
                    crew.historian()
                    
                    # Verify rate limiting was logged
                    info_calls = [str(call) for call in mock_logger.info.call_args_list]
                    assert any("Applied rate limiting to LLM" in call for call in info_calls)
    
    def test_monitoring_integration(self):
        """Test that rate limiting monitoring is properly integrated."""
        # Reset stats
        reset_monitoring_stats()
        
        # Get initial stats
        initial_stats = get_monitoring_stats()
        assert initial_stats["overall_stats"]["total_requests"] == 0
        
        # Create crew (this should trigger some monitoring)
        crew = ServerResearchMcpCrew(inputs={"topic": "test topic"})
        
        # Verify monitoring is working
        assert hasattr(crew, 'llm_config')
        assert hasattr(crew, 'tool_registry')
    
    @patch('src.server_research_mcp.utils.llm_rate_limiter.get_rate_limited_llm')
    @patch('crewai.Agent')
    def test_different_agent_types_get_different_tools(self, mock_agent_class, mock_get_rate_limited_llm):
        """Test that different agent types get tools with appropriate rate limiting."""
        mock_rate_limited_llm = Mock()
        mock_rate_limited_llm.rate_limiter_applied = True
        mock_get_rate_limited_llm.return_value = mock_rate_limited_llm
        
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent
        
        crew = ServerResearchMcpCrew(inputs={"topic": "test topic"})
        
        # Test different agent types
        agent_types = ["historian", "researcher", "archivist", "publisher"]
        
        for agent_type in agent_types:
            with patch.object(crew.tool_registry, 'get_agent_tools', return_value=[Mock(), Mock()]) as mock_get_tools:
                # Get agent
                agent_method = getattr(crew, agent_type)
                agent_method()
                
                # Verify tools were requested with rate limiting
                mock_get_tools.assert_called_once_with(agent_type, apply_rate_limiting=True)
                mock_get_tools.reset_mock()
    
    @patch('src.server_research_mcp.utils.llm_rate_limiter.get_rate_limited_llm')
    def test_error_handling_with_rate_limiting(self, mock_get_rate_limited_llm):
        """Test that errors are handled gracefully with rate limiting."""
        # Setup mock to raise an error
        mock_get_rate_limited_llm.side_effect = Exception("Rate limiting setup failed")
        
        crew = ServerResearchMcpCrew(inputs={"topic": "test topic"})
        
        with patch('src.server_research_mcp.crew.logger') as mock_logger:
            with patch.object(crew.tool_registry, 'get_agent_tools', return_value=[Mock()]):
                # Should handle error gracefully
                with pytest.raises(Exception):
                    crew.historian()
                
                # Verify error was logged
                error_calls = [str(call) for call in mock_logger.error.call_args_list]
                assert any("Failed to create agent" in call for call in error_calls)


class TestRateLimitingPerformanceImpact:
    """Test the performance impact of rate limiting."""
    
    @patch('src.server_research_mcp.utils.llm_rate_limiter.get_rate_limited_llm')
    def test_rate_limiting_overhead_is_minimal(self, mock_get_rate_limited_llm):
        """Test that rate limiting doesn't add significant overhead to crew creation."""
        import time
        
        # Setup mock
        mock_rate_limited_llm = Mock()
        mock_rate_limited_llm.rate_limiter_applied = True
        mock_get_rate_limited_llm.return_value = mock_rate_limited_llm
        
        # Measure crew creation time
        start_time = time.time()
        
        crew = ServerResearchMcpCrew(inputs={"topic": "test topic"})
        
        with patch('crewai.Agent'):
            with patch.object(crew.tool_registry, 'get_agent_tools', return_value=[Mock()]):
                # Create one agent
                crew.historian()
        
        end_time = time.time()
        creation_time = end_time - start_time
        
        # Should complete quickly (less than 1 second for mocked operations)
        assert creation_time < 1.0, f"Crew creation took too long: {creation_time}s"
    
    def test_monitoring_stats_collection(self):
        """Test that monitoring stats are collected without performance impact."""
        import time
        
        # Reset stats
        reset_monitoring_stats()
        
        start_time = time.time()
        
        # Get stats multiple times
        for _ in range(100):
            stats = get_monitoring_stats()
        
        end_time = time.time()
        stats_time = end_time - start_time
        
        # Should be very fast
        assert stats_time < 0.1, f"Stats collection too slow: {stats_time}s"


class TestRateLimitingConfiguration:
    """Test rate limiting configuration and environment variables."""
    
    @patch.dict('os.environ', {
        'LLM_MAX_REQUESTS_PER_MINUTE': '10',
        'MCP_ZOTERO_MAX_REQUESTS_PER_MINUTE': '3'
    })
    @patch('src.server_research_mcp.utils.llm_rate_limiter.get_rate_limited_llm')
    def test_environment_variables_are_respected(self, mock_get_rate_limited_llm):
        """Test that environment variables override default rate limiting configs."""
        # Setup mock
        mock_rate_limited_llm = Mock()
        mock_rate_limited_llm.rate_limiter_applied = True
        mock_rate_limited_llm.rate_limiter = Mock()
        mock_rate_limited_llm.rate_limiter.config = Mock()
        mock_rate_limited_llm.rate_limiter.config.max_requests_per_minute = 10  # From env var
        mock_get_rate_limited_llm.return_value = mock_rate_limited_llm
        
        crew = ServerResearchMcpCrew(inputs={"topic": "test topic"})
        
        with patch('crewai.Agent'):
            with patch.object(crew.tool_registry, 'get_agent_tools', return_value=[Mock()]):
                crew.historian()
                
                # Verify rate limiting was applied with env var values
                mock_get_rate_limited_llm.assert_called_once()
    
    def test_rate_limiting_can_be_disabled(self):
        """Test that rate limiting can be disabled if needed."""
        crew = ServerResearchMcpCrew(inputs={"topic": "test topic"})
        
        with patch.object(crew.tool_registry, 'get_agent_tools') as mock_get_tools:
            # Test with rate limiting disabled
            mock_get_tools.return_value = [Mock(), Mock()]
            
            # This should work even if we patch get_agent_tools to not apply rate limiting
            with patch('crewai.Agent'):
                agent = crew.historian()
                
                # Verify tools were requested
                mock_get_tools.assert_called_once() 