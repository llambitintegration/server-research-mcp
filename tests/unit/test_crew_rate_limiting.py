"""
Unit tests for crew rate limiting integration.

Tests the integration of rate limiting in crew.py, including LLM wrapping,
agent creation with rate limiting, and timeout/retry configuration.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from src.server_research_mcp.crew import ServerResearchMcpCrew, AgentDefinition, TaskDefinition
from src.server_research_mcp.utils.llm_rate_limiter import RateLimitedLLM
from src.server_research_mcp.utils.rate_limiting import RateLimitConfig
from src.server_research_mcp.schemas.research_paper import ResearchPaperSchema


@pytest.fixture
def mock_llm_config():
    """Create a mock LLM configuration."""
    config = Mock()
    config.get_llm.return_value = Mock()
    config.get_llm.return_value.model_name = "test-model"
    config.get_llm.return_value.temperature = 0.7
    config.get_llm.return_value.max_tokens = 1000
    config.get_llm.return_value.__class__.__name__ = "MockLLM"
    return config


@pytest.fixture
def mock_mcp_registry():
    """Create a mock MCP registry."""
    registry = Mock()
    registry.get_agent_tools.return_value = [Mock(), Mock()]
    return registry


@pytest.fixture
def sample_agent_definition():
    """Create a sample agent definition for testing."""
    agent_def = AgentDefinition(
        name="test_agent",
        schema=ResearchPaperSchema,
        tools_pattern="test",
        min_tools=1,
        max_iter=5,
        max_execution_time=300
    )
    # Manually set the role, goal, and backstory
    agent_def.role = "Test Agent Role for {topic}"
    agent_def.goal = "Test goal for {topic}"
    agent_def.backstory = "Test backstory for {topic}"
    return agent_def


@pytest.fixture
def sample_task_definition():
    """Create a sample task definition for testing."""
    return TaskDefinition(
        name="test_task",
        description="Test task description for {topic}",
        agent_name="test_agent",
        expected_output="Test output for {topic}"
    )


class TestCrewRateLimiting:
    """Test suite for crew rate limiting integration."""
    
    @patch('src.server_research_mcp.crew.get_rate_limited_llm')
    def test_llm_rate_limiting_integration(self, mock_get_rate_limited_llm, mock_llm_config, mock_mcp_registry):
        """Test that LLMs are wrapped with rate limiting during agent creation."""
        # Setup
        mock_rate_limited_llm = Mock(spec=RateLimitedLLM)
        mock_get_rate_limited_llm.return_value = mock_rate_limited_llm
        
        crew = ServerResearchMcpCrew(inputs={"topic": "test topic"})
        crew.llm_config = mock_llm_config
        crew.tool_registry = mock_mcp_registry
        
        # Create agent definition
        agent_def = AgentDefinition(
            name="test_agent",
            schema=ResearchPaperSchema,
            tools_pattern="test",
            min_tools=1
        )
        agent_def.role = "Test Role"
        agent_def.goal = "Test Goal"
        agent_def.backstory = "Test Backstory"
        
        # Execute agent creation
        with patch.object(crew, '_create_fallback_tools', return_value=[]):
            agent = crew._create_agent_from_definition(agent_def)
        
        # Verify LLM was wrapped with rate limiting
        mock_get_rate_limited_llm.assert_called_once()
        call_args = mock_get_rate_limited_llm.call_args
        assert call_args[0][0] == mock_llm_config.get_llm.return_value
    
    def test_max_execution_time_doubling(self, mock_llm_config, mock_mcp_registry, sample_agent_definition):
        """Test that max_execution_time is doubled for rate limiting accommodation."""
        crew = ServerResearchMcpCrew(
            llm_config=mock_llm_config,
            mcp_registry=mock_mcp_registry,
            topic="test topic"
        )
        
        # Agent with max_execution_time of 300
        agent_def = sample_agent_definition
        agent_def.max_execution_time = 300
        
        with patch.object(crew, '_create_fallback_tools', return_value=[]):
            with patch('src.server_research_mcp.crew.get_rate_limited_llm'):
                with patch('crewai.Agent') as mock_agent_class:
                    crew._create_agent_from_definition(agent_def)
                    
                    # Verify Agent was called with doubled max_execution_time
                    mock_agent_class.assert_called_once()
                    call_kwargs = mock_agent_class.call_args[1]
                    assert call_kwargs['max_execution_time'] == 600  # 300 * 2
    
    def test_max_execution_time_none_handling(self, mock_llm_config, mock_mcp_registry, sample_agent_definition):
        """Test that None max_execution_time is handled correctly."""
        crew = ServerResearchMcpCrew(
            llm_config=mock_llm_config,
            mcp_registry=mock_mcp_registry,
            topic="test topic"
        )
        
        # Agent with None max_execution_time
        agent_def = sample_agent_definition
        agent_def.max_execution_time = None
        
        with patch.object(crew, '_create_fallback_tools', return_value=[]):
            with patch('src.server_research_mcp.crew.get_rate_limited_llm'):
                with patch('crewai.Agent') as mock_agent_class:
                    crew._create_agent_from_definition(agent_def)
                    
                    # Verify Agent was called without max_execution_time
                    mock_agent_class.assert_called_once()
                    call_kwargs = mock_agent_class.call_args[1]
                    assert 'max_execution_time' not in call_kwargs
    
    @patch('src.server_research_mcp.crew.logger')
    def test_rate_limiting_logging(self, mock_logger, mock_llm_config, mock_mcp_registry, sample_agent_definition):
        """Test that rate limiting integration is properly logged."""
        crew = ServerResearchMcpCrew(
            llm_config=mock_llm_config,
            mcp_registry=mock_mcp_registry,
            topic="test topic"
        )
        
        with patch.object(crew, '_create_fallback_tools', return_value=[]):
            with patch('src.server_research_mcp.crew.get_rate_limited_llm') as mock_get_rate_limited:
                mock_rate_limited_llm = Mock(spec=RateLimitedLLM)
                mock_rate_limited_llm.rate_limiter = Mock()
                mock_rate_limited_llm.rate_limiter.config = Mock()
                mock_rate_limited_llm.rate_limiter.config.max_requests_per_minute = 20
                mock_rate_limited_llm.rate_limiter.config.min_request_interval = 0.5
                mock_get_rate_limited.return_value = mock_rate_limited_llm
                
                crew._create_agent_from_definition(sample_agent_definition)
                
                # Verify logging calls
                info_calls = [str(call) for call in mock_logger.info.call_args_list]
                
                # Should log rate limiting application
                assert any("Rate limiting applied to LLM" in call for call in info_calls)
                
                # Should log rate limiting configuration
                assert any("20 req/min" in call for call in info_calls)
                assert any("0.5s interval" in call for call in info_calls)
    
    def test_agent_tools_rate_limiting_integration(self, mock_llm_config, mock_mcp_registry, sample_agent_definition):
        """Test that agent tools are retrieved with rate limiting enabled."""
        crew = ServerResearchMcpCrew(
            llm_config=mock_llm_config,
            mcp_registry=mock_mcp_registry,
            topic="test topic"
        )
        
        with patch.object(crew, '_create_fallback_tools', return_value=[]):
            with patch('src.server_research_mcp.crew.get_rate_limited_llm'):
                crew._create_agent_from_definition(sample_agent_definition)
                
                # Verify tools were requested with rate limiting enabled
                mock_mcp_registry.get_agent_tools.assert_called_once_with(
                    sample_agent_definition.name,
                    apply_rate_limiting=True
                )
    
    def test_retry_configuration_enhancement(self, mock_llm_config, mock_mcp_registry, sample_agent_definition):
        """Test that retry configuration is enhanced for rate limiting."""
        crew = ServerResearchMcpCrew(
            llm_config=mock_llm_config,
            mcp_registry=mock_mcp_registry,
            topic="test topic"
        )
        
        # Agent with max_iter
        agent_def = sample_agent_definition
        agent_def.max_iter = 5
        
        with patch.object(crew, '_create_fallback_tools', return_value=[]):
            with patch('src.server_research_mcp.crew.get_rate_limited_llm'):
                with patch('crewai.Agent') as mock_agent_class:
                    crew._create_agent_from_definition(agent_def)
                    
                    # Verify Agent was called with correct max_iter
                    mock_agent_class.assert_called_once()
                    call_kwargs = mock_agent_class.call_args[1]
                    assert call_kwargs['max_iter'] == 5
    
    @patch('src.server_research_mcp.crew.get_rate_limited_llm')
    def test_llm_provider_detection(self, mock_get_rate_limited_llm, mock_llm_config, mock_mcp_registry):
        """Test that LLM provider is correctly detected for rate limiting."""
        # Setup different LLM types
        openai_llm = Mock()
        openai_llm.__class__.__name__ = "OpenAI"
        openai_llm.model_name = "gpt-4"
        
        anthropic_llm = Mock()
        anthropic_llm.__class__.__name__ = "AnthropicLLM"
        anthropic_llm.model_name = "claude-3"
        
        mock_llm_config.get_llm.side_effect = [openai_llm, anthropic_llm]
        
        crew = ServerResearchMcpCrew(
            llm_config=mock_llm_config,
            mcp_registry=mock_mcp_registry,
            topic="test topic"
        )
        
        agent_def = AgentDefinition(
            name="test_agent",
            role="Test Role",
            goal="Test Goal",
            backstory="Test Backstory", 
            schema=ResearchPaperSchema
        )
        
        # Test OpenAI LLM
        with patch.object(crew, '_create_fallback_tools', return_value=[]):
            crew._create_agent_from_definition(agent_def)
            
            # Verify rate limiting was applied to OpenAI LLM
            assert mock_get_rate_limited_llm.call_count == 1
            call_args = mock_get_rate_limited_llm.call_args
            assert call_args[0][0] == openai_llm
        
        # Reset mock
        mock_get_rate_limited_llm.reset_mock()
        
        # Test Anthropic LLM
        with patch.object(crew, '_create_fallback_tools', return_value=[]):
            crew._create_agent_from_definition(agent_def)
            
            # Verify rate limiting was applied to Anthropic LLM
            assert mock_get_rate_limited_llm.call_count == 1
            call_args = mock_get_rate_limited_llm.call_args
            assert call_args[0][0] == anthropic_llm


class TestCrewRateLimitingConfiguration:
    """Test crew rate limiting configuration and environment integration."""
    
    @patch.dict('os.environ', {
        'LLM_MAX_REQUESTS_PER_MINUTE': '15',
        'LLM_MIN_REQUEST_INTERVAL': '1.0'
    })
    def test_environment_variable_integration(self, mock_llm_config, mock_mcp_registry, sample_agent_definition):
        """Test that environment variables are respected in crew rate limiting."""
        crew = ServerResearchMcpCrew(
            llm_config=mock_llm_config,
            mcp_registry=mock_mcp_registry,
            topic="test topic"
        )
        
        with patch.object(crew, '_create_fallback_tools', return_value=[]):
            with patch('src.server_research_mcp.crew.get_rate_limited_llm') as mock_get_rate_limited:
                mock_rate_limited_llm = Mock(spec=RateLimitedLLM)
                mock_rate_limited_llm.rate_limiter = Mock()
                mock_rate_limited_llm.rate_limiter.config = Mock()
                mock_rate_limited_llm.rate_limiter.config.max_requests_per_minute = 15
                mock_rate_limited_llm.rate_limiter.config.min_request_interval = 1.0
                mock_get_rate_limited.return_value = mock_rate_limited_llm
                
                crew._create_agent_from_definition(sample_agent_definition)
                
                # Verify the rate limited LLM was created
                mock_get_rate_limited.assert_called_once()
    
    def test_error_handling_in_rate_limiting(self, mock_llm_config, mock_mcp_registry, sample_agent_definition):
        """Test error handling during rate limiting setup."""
        crew = ServerResearchMcpCrew(
            llm_config=mock_llm_config,
            mcp_registry=mock_mcp_registry,
            topic="test topic"
        )
        
        # Mock get_rate_limited_llm to raise an exception
        with patch('src.server_research_mcp.crew.get_rate_limited_llm') as mock_get_rate_limited:
            mock_get_rate_limited.side_effect = Exception("Rate limiting setup failed")
            
            with patch.object(crew, '_create_fallback_tools', return_value=[]):
                with patch('src.server_research_mcp.crew.logger') as mock_logger:
                    # Should handle the error gracefully
                    with pytest.raises(Exception):
                        crew._create_agent_from_definition(sample_agent_definition)
                    
                    # Should log the error
                    error_calls = [str(call) for call in mock_logger.error.call_args_list]
                    assert any("Rate limiting setup failed" in call for call in error_calls)
    
    def test_rate_limiting_disabled_fallback(self, mock_llm_config, mock_mcp_registry, sample_agent_definition):
        """Test fallback behavior when rate limiting is disabled or fails."""
        crew = ServerResearchMcpCrew(
            llm_config=mock_llm_config,
            mcp_registry=mock_mcp_registry,
            topic="test topic"
        )
        
        # Mock get_rate_limited_llm to return None (disabled)
        with patch('src.server_research_mcp.crew.get_rate_limited_llm') as mock_get_rate_limited:
            mock_get_rate_limited.return_value = None
            
            with patch.object(crew, '_create_fallback_tools', return_value=[]):
                with patch('crewai.Agent') as mock_agent_class:
                    crew._create_agent_from_definition(sample_agent_definition)
                    
                    # Should use original LLM
                    mock_agent_class.assert_called_once()
                    call_kwargs = mock_agent_class.call_args[1]
                    assert call_kwargs['llm'] == mock_llm_config.get_llm.return_value


class TestCrewRateLimitingIntegration:
    """Integration tests for crew rate limiting with other components."""
    
    def test_full_crew_creation_with_rate_limiting(self, mock_llm_config, mock_mcp_registry):
        """Test full crew creation with rate limiting enabled."""
        # Setup agent definitions
        # Create agent definitions
        researcher_def = AgentDefinition(
            name="researcher",
            schema=ResearchPaperSchema,
            tools_pattern="test",
            min_tools=1
        )
        researcher_def.role = "Research Agent"
        researcher_def.goal = "Research goal"
        researcher_def.backstory = "Research backstory"
        
        historian_def = AgentDefinition(
            name="historian",
            schema=ResearchPaperSchema,
            tools_pattern="test",
            min_tools=1
        )
        historian_def.role = "History Agent"
        historian_def.goal = "History goal"
        historian_def.backstory = "History backstory"
        
        agent_defs = [researcher_def, historian_def]
        
        # Setup task definitions
        task_defs = [
            TaskDefinition(
                name="research_task",
                description="Research task",
                agent_name="researcher",
                expected_output="Research output"
            ),
            TaskDefinition(
                name="history_task",
                description="History task", 
                agent_name="historian",
                expected_output="History output"
            )
        ]
        
        crew = ServerResearchMcpCrew(
            llm_config=mock_llm_config,
            mcp_registry=mock_mcp_registry,
            topic="test topic"
        )
        
        # Override definitions
        crew.agent_definitions = agent_defs
        crew.task_definitions = task_defs
        
        with patch.object(crew, '_create_fallback_tools', return_value=[]):
            with patch('src.server_research_mcp.crew.get_rate_limited_llm') as mock_get_rate_limited:
                with patch('crewai.Agent'):
                    with patch('crewai.Task'):
                        with patch('crewai.Crew'):
                            crew_instance = crew.create_crew()
                            
                            # Verify rate limiting was applied to all agents
                            assert mock_get_rate_limited.call_count == len(agent_defs)
    
    def test_rate_limiting_with_different_agent_types(self, mock_llm_config, mock_mcp_registry):
        """Test that different agent types get appropriate rate limiting."""
        agent_types = ["researcher", "historian", "publisher", "archivist"]
        
        for agent_type in agent_types:
            crew = ServerResearchMcpCrew(
                llm_config=mock_llm_config,
                mcp_registry=mock_mcp_registry,
                topic="test topic"
            )
            
            agent_def = AgentDefinition(
                name=agent_type,
                schema=ResearchPaperSchema,
                tools_pattern="test",
                min_tools=1
            )
            agent_def.role = f"{agent_type} Role"
            agent_def.goal = f"{agent_type} Goal"
            agent_def.backstory = f"{agent_type} Backstory"
            
            with patch.object(crew, '_create_fallback_tools', return_value=[]):
                with patch('src.server_research_mcp.crew.get_rate_limited_llm'):
                    crew._create_agent_from_definition(agent_def)
                    
                    # Verify tools were requested for the specific agent type
                    mock_mcp_registry.get_agent_tools.assert_called_with(
                        agent_type,
                        apply_rate_limiting=True
                    )
            
            # Reset mock for next iteration
            mock_mcp_registry.reset_mock()
    
    @patch('src.server_research_mcp.crew.logger')
    def test_comprehensive_rate_limiting_logging(self, mock_logger, mock_llm_config, mock_mcp_registry):
        """Test comprehensive logging throughout the rate limiting process."""
        crew = ServerResearchMcpCrew(
            llm_config=mock_llm_config,
            mcp_registry=mock_mcp_registry,
            topic="test topic"
        )
        
        agent_def = AgentDefinition(
            name="test_agent",
            schema=ResearchPaperSchema,
            tools_pattern="test",
            min_tools=1,
            max_execution_time=300
        )
        agent_def.role = "Test Role"
        agent_def.goal = "Test Goal"
        agent_def.backstory = "Test Backstory"
        
        with patch.object(crew, '_create_fallback_tools', return_value=[]):
            with patch('src.server_research_mcp.crew.get_rate_limited_llm') as mock_get_rate_limited:
                mock_rate_limited_llm = Mock(spec=RateLimitedLLM)
                mock_rate_limited_llm.rate_limiter = Mock()
                mock_rate_limited_llm.rate_limiter.config = Mock()
                mock_rate_limited_llm.rate_limiter.config.max_requests_per_minute = 20
                mock_rate_limited_llm.rate_limiter.config.min_request_interval = 0.5
                mock_get_rate_limited.return_value = mock_rate_limited_llm
                
                crew._create_agent_from_definition(agent_def)
                
                # Verify comprehensive logging
                info_calls = [str(call) for call in mock_logger.info.call_args_list]
                
                # Should log LLM rate limiting
                assert any("Rate limiting applied to LLM" in call for call in info_calls)
                
                # Should log timeout doubling
                assert any("Doubled max_execution_time" in call for call in info_calls)
                
                # Should log rate limiting configuration
                assert any("20 req/min" in call for call in info_calls)
                assert any("0.5s interval" in call for call in info_calls) 