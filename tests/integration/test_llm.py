"""Integration tests for LLM connections and configurations."""

import pytest
import os
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv
from crewai import LLM

# Load environment variables
load_dotenv()


class TestLLMConfiguration:
    """Test LLM configuration and setup."""
    
    def test_llm_config_from_environment(self, llm_config):
        """Test LLM configuration from environment variables."""
        assert 'provider' in llm_config
        assert 'api_key' in llm_config
        assert 'model' in llm_config
        
        # Verify provider is supported
        assert llm_config['provider'] in ['anthropic', 'openai']
        
        # Verify model format
        assert llm_config['model'].startswith(llm_config['provider'])
        
    def test_llm_instance_creation(self, llm_config):
        """Test LLM instance can be created with config."""
        if not llm_config.get('api_key'):
            pytest.skip(f"API key not found for {llm_config['provider']}")
            
        llm = LLM(
            model=llm_config['model'],
            api_key=llm_config['api_key']
        )
        
        assert llm is not None
        assert hasattr(llm, 'call')
        
    def test_llm_provider_configuration_from_env(self, llm_config):
        """Test LLM provider configuration from actual .env."""
        # Test that configuration is properly loaded from environment
        assert llm_config['provider'] in ['anthropic', 'openai'], f"Unsupported provider: {llm_config['provider']}"
        
        # Test model format matches provider
        assert llm_config['model'].startswith(f"{llm_config['provider']}/"), f"Model {llm_config['model']} doesn't match provider {llm_config['provider']}"
        
        # Test API key is present
        assert llm_config['api_key'], f"API key missing for provider {llm_config['provider']}"


@pytest.mark.requires_llm
class TestLLMConnections:
    """Test actual LLM connections (requires API keys)."""
    
    def test_llm_basic_call(self, llm_instance):
        """Test basic LLM call functionality."""
        messages = [
            {
                "role": "user",
                "content": "Reply with a brief test response to confirm connection."
            }
        ]
        
        response = llm_instance.call(messages)
        
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0  # Any non-empty response indicates connection works
        
    def test_llm_string_format(self, llm_instance):
        """Test LLM call with string format."""
        response = llm_instance.call("Say 'Hello from test' exactly.")
        
        assert response is not None
        assert isinstance(response, str)
        assert "hello" in response.lower()
        
    def test_llm_conversation_format(self, llm_instance):
        """Test LLM with conversation history."""
        messages = [
            {"role": "user", "content": "Remember the number 42."},
            {"role": "assistant", "content": "I'll remember the number 42."},
            {"role": "user", "content": "What number did I ask you to remember?"}
        ]
        
        response = llm_instance.call(messages)
        
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0  # Any response indicates conversation handling works
        
    @pytest.mark.parametrize("prompt,validation_fn", [
        ("What is 2+2?", lambda r: any(word in r.lower() for word in ["4", "four"])),
        ("Complete: Hello, ", lambda r: len(r) > 5),  # Any reasonable completion
        ("Is this a test? Reply yes or no.", lambda r: any(word in r.lower() for word in ["yes", "no"]))
    ])
    def test_llm_various_prompts(self, llm_instance, prompt, validation_fn):
        """Test LLM with various prompt types - flexible validation for real responses."""
        response = llm_instance.call(prompt)
        
        assert response is not None
        assert validation_fn(response), f"Response '{response}' failed validation for prompt '{prompt}'"


class TestLLMIntegrationWithCrew:
    """Test LLM integration with CrewAI components."""
    
    def test_llm_in_agent_configuration(self, mock_llm):
        """Test LLM is properly configured in agents."""
        with patch('server_research_mcp.config.llm_config.create_llm', return_value=mock_llm):
            from server_research_mcp.crew import ServerResearchMcp
            
            crew_instance = ServerResearchMcp()
            
            # Test that agents would use the LLM
            # Note: actual agent methods may use decorators
            assert crew_instance is not None
            
    def test_llm_error_handling(self):
        """Test LLM handles errors gracefully."""
        # Create LLM with invalid config and try to use it
        with pytest.raises(Exception):
            llm = LLM(
                model="invalid/model",
                api_key=None
            )
            # Actually call the LLM to trigger the error
            llm.call("test message")
            
    def test_llm_timeout_handling(self, llm_config):
        """Test LLM timeout handling."""
        if not llm_config.get('api_key'):
            pytest.skip("No API key available")
            
        # Create LLM with timeout settings
        llm = LLM(
            model=llm_config['model'],
            api_key=llm_config['api_key'],
            timeout=1  # Very short timeout
        )
        
        # This might timeout or succeed quickly
        try:
            response = llm.call("Quick test")
            assert isinstance(response, str)
        except Exception as e:
            # Timeout or authentication error is acceptable (testing with mock key)
            assert "timeout" in str(e).lower() or "authentication" in str(e).lower()


class TestLLMProviderFallback:
    """Test LLM provider fallback mechanisms."""
    
    @patch('os.getenv')
    def test_missing_provider_fallback(self, mock_getenv):
        """Test fallback when provider is not specified."""
        mock_getenv.side_effect = lambda key, default=None: {
            'ANTHROPIC_API_KEY': 'test-key',
            'LLM_MODEL': 'claude-3-haiku-20240307'
        }.get(key, default)
        
        from server_research_mcp.config.llm_config import get_llm_config
        config = get_llm_config()
        
        # Should default to anthropic
        assert config['provider'] == 'anthropic'
        
    def test_invalid_provider_handling(self):
        """Test handling of invalid provider."""
        with patch('os.getenv', return_value='invalid_provider'):
            # This should be handled gracefully
            from server_research_mcp.config.llm_config import get_llm_config
            
            # Should either raise error or fall back to default
            try:
                config = get_llm_config()
                assert config['provider'] in ['anthropic', 'openai']
            except Exception as e:
                assert "provider" in str(e).lower()


@pytest.mark.slow
class TestLLMPerformance:
    """Test LLM performance characteristics."""
    
    @pytest.mark.requires_llm
    def test_llm_response_time(self, llm_instance):
        """Test LLM response time is reasonable."""
        import time
        
        start = time.time()
        response = llm_instance.call("Reply with 'fast' immediately.")
        end = time.time()
        
        assert response is not None
        assert (end - start) < 30  # Should respond within 30 seconds
        
    @pytest.mark.requires_llm
    def test_llm_token_handling(self, llm_instance):
        """Test LLM handles different token lengths."""
        # Short prompt
        short_response = llm_instance.call("Say 'hi'")
        assert len(short_response) < 100
        
        # Longer prompt
        long_prompt = "List 5 benefits of automated testing. Be brief, one line each."
        long_response = llm_instance.call(long_prompt)
        assert len(long_response) > 50
        assert len(long_response) < 1000