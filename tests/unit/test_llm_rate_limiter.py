"""
Unit tests for LLM rate limiting functionality.

Tests the RateLimitedLLM wrapper, provider detection, environment variable
overrides, and rate limiting for invoke/generate/predict methods.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from src.server_research_mcp.utils.llm_rate_limiter import (
    RateLimitedLLM, 
    wrap_llm_with_rate_limit, 
    get_rate_limited_llm
)
from src.server_research_mcp.utils.rate_limiting import RateLimitConfig, RateLimitError


class MockLLM:
    """Mock LLM for testing."""
    
    def __init__(self, model_name: str = "test-model", provider: str = "openai"):
        self.model_name = model_name
        self.model = model_name
        self.temperature = 0.7
        self.max_tokens = 1000
        self.__class__.__name__ = f"{provider}LLM"
    
    def invoke(self, prompt: str):
        return f"Response to: {prompt}"
    
    def generate(self, prompts):
        return [f"Generated: {p}" for p in prompts]
    
    def predict(self, text: str):
        return f"Predicted: {text}"
    
    def __call__(self, prompt: str):
        return self.invoke(prompt)


@pytest.fixture
def mock_openai_llm():
    """Create a mock OpenAI LLM."""
    return MockLLM("gpt-4", "openai")


@pytest.fixture
def mock_anthropic_llm():
    """Create a mock Anthropic LLM."""
    llm = MockLLM("claude-3", "anthropic")
    llm.__class__.__name__ = "AnthropicLLM"
    return llm


@pytest.fixture
def mock_unknown_llm():
    """Create a mock unknown LLM."""
    llm = MockLLM("unknown-model", "unknown")
    llm.__class__.__name__ = "UnknownLLM"
    return llm


class TestRateLimitedLLM:
    """Test suite for RateLimitedLLM wrapper."""
    
    def test_llm_wrapper_initialization(self, mock_openai_llm):
        """Test that LLM wrapper initializes correctly."""
        wrapper = RateLimitedLLM(mock_openai_llm)
        
        # Check wrapper attributes
        assert wrapper.wrapped_llm == mock_openai_llm
        assert wrapper.rate_limiter_applied is True
        assert wrapper.rate_limiter is not None
        assert wrapper.identifier == "llm_gpt-4"
        
        # Check copied attributes
        assert wrapper.model_name == "gpt-4"
        assert wrapper.temperature == 0.7
        assert wrapper.max_tokens == 1000
    
    def test_provider_detection_openai(self, mock_openai_llm):
        """Test OpenAI provider detection."""
        wrapper = RateLimitedLLM(mock_openai_llm)
        provider = wrapper._detect_provider(mock_openai_llm)
        assert provider == "openai"
    
    def test_provider_detection_anthropic(self, mock_anthropic_llm):
        """Test Anthropic provider detection."""
        wrapper = RateLimitedLLM(mock_anthropic_llm)
        provider = wrapper._detect_provider(mock_anthropic_llm)
        assert provider == "anthropic"
    
    def test_provider_detection_unknown(self, mock_unknown_llm):
        """Test unknown provider detection."""
        wrapper = RateLimitedLLM(mock_unknown_llm)
        provider = wrapper._detect_provider(mock_unknown_llm)
        assert provider == "unknown"
    
    def test_openai_default_config(self, mock_openai_llm):
        """Test OpenAI default rate limit configuration."""
        wrapper = RateLimitedLLM(mock_openai_llm)
        config = wrapper.rate_limiter.config
        
        assert config.max_requests_per_minute == 20
        assert config.max_requests_per_hour == 200
        assert config.min_request_interval == 0.5
        assert config.max_retries == 5
    
    def test_anthropic_default_config(self, mock_anthropic_llm):
        """Test Anthropic default rate limit configuration."""
        wrapper = RateLimitedLLM(mock_anthropic_llm)
        config = wrapper.rate_limiter.config
        
        assert config.max_requests_per_minute == 50
        assert config.max_requests_per_hour == 500
        assert config.min_request_interval == 0.2
        assert config.max_retries == 5
    
    @patch.dict('os.environ', {
        'LLM_MAX_REQUESTS_PER_MINUTE': '10',
        'LLM_MAX_REQUESTS_PER_HOUR': '100',
        'LLM_MIN_REQUEST_INTERVAL': '1.0'
    })
    def test_environment_variable_override(self, mock_openai_llm):
        """Test that environment variables override default configurations."""
        wrapper = RateLimitedLLM(mock_openai_llm)
        config = wrapper.rate_limiter.config
        
        assert config.max_requests_per_minute == 10
        assert config.max_requests_per_hour == 100
        assert config.min_request_interval == 1.0
    
    def test_invoke_method_rate_limiting(self, mock_openai_llm):
        """Test that invoke method is rate limited."""
        wrapper = RateLimitedLLM(mock_openai_llm)
        
        # Mock rate limiter to allow request
        wrapper.rate_limiter.check_rate_limit = Mock(return_value=(True, None))
        wrapper.rate_limiter.record_request = Mock()
        
        result = wrapper.invoke("test prompt")
        
        # Verify rate limiting was checked
        wrapper.rate_limiter.check_rate_limit.assert_called_once()
        wrapper.rate_limiter.record_request.assert_called_once()
        
        # Verify original method was called
        assert result == "Response to: test prompt"
    
    def test_generate_method_rate_limiting(self, mock_openai_llm):
        """Test that generate method is rate limited."""
        wrapper = RateLimitedLLM(mock_openai_llm)
        
        # Mock rate limiter to allow request
        wrapper.rate_limiter.check_rate_limit = Mock(return_value=(True, None))
        wrapper.rate_limiter.record_request = Mock()
        
        result = wrapper.generate(["prompt1", "prompt2"])
        
        # Verify rate limiting was checked
        wrapper.rate_limiter.check_rate_limit.assert_called_once()
        wrapper.rate_limiter.record_request.assert_called_once()
        
        # Verify original method was called
        assert result == ["Generated: prompt1", "Generated: prompt2"]
    
    def test_predict_method_rate_limiting(self, mock_openai_llm):
        """Test that predict method is rate limited."""
        wrapper = RateLimitedLLM(mock_openai_llm)
        
        # Mock rate limiter to allow request
        wrapper.rate_limiter.check_rate_limit = Mock(return_value=(True, None))
        wrapper.rate_limiter.record_request = Mock()
        
        result = wrapper.predict("test text")
        
        # Verify rate limiting was checked
        wrapper.rate_limiter.check_rate_limit.assert_called_once()
        wrapper.rate_limiter.record_request.assert_called_once()
        
        # Verify original method was called
        assert result == "Predicted: test text"
    
    def test_call_method_rate_limiting(self, mock_openai_llm):
        """Test that __call__ method is rate limited."""
        wrapper = RateLimitedLLM(mock_openai_llm)
        
        # Mock rate limiter to allow request
        wrapper.rate_limiter.check_rate_limit = Mock(return_value=(True, None))
        wrapper.rate_limiter.record_request = Mock()
        
        result = wrapper("test prompt")
        
        # Verify rate limiting was checked
        wrapper.rate_limiter.check_rate_limit.assert_called_once()
        wrapper.rate_limiter.record_request.assert_called_once()
        
        # Verify original method was called
        assert result == "Response to: test prompt"
    
    def test_rate_limit_wait(self, mock_openai_llm):
        """Test that wrapper waits when rate limited."""
        wrapper = RateLimitedLLM(mock_openai_llm)
        
        # Mock rate limiter to require waiting
        wrapper.rate_limiter.check_rate_limit = Mock(side_effect=[
            (False, 0.1),  # First check: wait required
            (True, None)   # Second check: allowed
        ])
        wrapper.rate_limiter.record_request = Mock()
        
        with patch('time.sleep') as mock_sleep:
            result = wrapper.invoke("test prompt")
            
            # Verify sleep was called with wait time
            mock_sleep.assert_called_once_with(0.1)
            
            # Verify rate limiting was checked twice
            assert wrapper.rate_limiter.check_rate_limit.call_count == 2
    
    def test_retry_on_rate_limit_error(self, mock_openai_llm):
        """Test retry behavior on rate limit errors."""
        wrapper = RateLimitedLLM(mock_openai_llm)
        
        # Mock rate limiter
        wrapper.rate_limiter.check_rate_limit = Mock(return_value=(True, None))
        wrapper.rate_limiter.record_error = Mock(return_value=0.1)
        wrapper.rate_limiter.record_request = Mock()
        wrapper.rate_limiter.config.max_retries = 2
        
        # Mock LLM to raise rate limit error then succeed
        mock_openai_llm.invoke = Mock(side_effect=[
            Exception("rate limit exceeded"),  # First call fails
            "Success"  # Second call succeeds
        ])
        
        with patch('time.sleep') as mock_sleep:
            result = wrapper.invoke("test prompt")
            
            # Verify retry occurred
            assert mock_openai_llm.invoke.call_count == 2
            wrapper.rate_limiter.record_error.assert_called_once()
            mock_sleep.assert_called_once_with(0.1)
            
            # Verify success
            assert result == "Success"
    
    def test_max_retries_exceeded(self, mock_openai_llm):
        """Test behavior when max retries are exceeded."""
        wrapper = RateLimitedLLM(mock_openai_llm)
        
        # Mock rate limiter
        wrapper.rate_limiter.check_rate_limit = Mock(return_value=(True, None))
        wrapper.rate_limiter.record_error = Mock(return_value=0.1)
        wrapper.rate_limiter.config.max_retries = 2
        
        # Mock LLM to always fail
        error = Exception("persistent error")
        mock_openai_llm.invoke = Mock(side_effect=error)
        
        with patch('time.sleep'):
            with pytest.raises(Exception) as exc_info:
                wrapper.invoke("test prompt")
            
            # Verify the original error is raised
            assert exc_info.value == error
            
            # Verify retries were attempted
            assert mock_openai_llm.invoke.call_count == 2
    
    def test_attribute_proxy(self, mock_openai_llm):
        """Test that unknown attributes are proxied to wrapped LLM."""
        wrapper = RateLimitedLLM(mock_openai_llm)
        
        # Add a custom attribute to the original LLM
        mock_openai_llm.custom_attribute = "test_value"
        
        # Verify it's accessible through the wrapper
        assert wrapper.custom_attribute == "test_value"
    
    def test_identifier_generation(self):
        """Test LLM identifier generation for different LLM types."""
        # Test with model_name
        llm1 = Mock()
        llm1.model_name = "gpt-4"
        wrapper1 = RateLimitedLLM(llm1)
        assert wrapper1.identifier == "llm_gpt-4"
        
        # Test with model attribute
        llm2 = Mock()
        del llm2.model_name  # Remove model_name
        llm2.model = "claude-3"
        wrapper2 = RateLimitedLLM(llm2)
        assert wrapper2.identifier == "llm_claude-3"
        
        # Test with class name
        llm3 = Mock()
        del llm3.model_name
        del llm3.model
        llm3.__class__.__name__ = "CustomLLM"
        wrapper3 = RateLimitedLLM(llm3)
        assert wrapper3.identifier == "llm_CustomLLM"


class TestLLMRateLimitingHelpers:
    """Test helper functions for LLM rate limiting."""
    
    def test_wrap_llm_with_rate_limit(self, mock_openai_llm):
        """Test wrap_llm_with_rate_limit function."""
        result = wrap_llm_with_rate_limit(mock_openai_llm)
        
        assert isinstance(result, RateLimitedLLM)
        assert result.wrapped_llm == mock_openai_llm
        assert result.rate_limiter_applied is True
    
    def test_wrap_already_rate_limited_llm(self, mock_openai_llm):
        """Test wrapping an already rate-limited LLM."""
        # First wrap
        wrapped_once = wrap_llm_with_rate_limit(mock_openai_llm)
        
        # Try to wrap again
        wrapped_twice = wrap_llm_with_rate_limit(wrapped_once)
        
        # Should return the same instance
        assert wrapped_twice == wrapped_once
    
    def test_get_rate_limited_llm(self, mock_openai_llm):
        """Test get_rate_limited_llm function."""
        result = get_rate_limited_llm(mock_openai_llm)
        
        assert isinstance(result, RateLimitedLLM)
        assert result.wrapped_llm == mock_openai_llm
    
    def test_get_rate_limited_llm_with_custom_config(self, mock_openai_llm):
        """Test get_rate_limited_llm with custom configuration."""
        custom_config = {
            "max_requests_per_minute": 5,
            "min_request_interval": 2.0
        }
        
        result = get_rate_limited_llm(mock_openai_llm, custom_config=custom_config)
        
        assert isinstance(result, RateLimitedLLM)
        assert result.rate_limiter.config.max_requests_per_minute == 5
        assert result.rate_limiter.config.min_request_interval == 2.0
    
    def test_custom_rate_limit_config(self, mock_openai_llm):
        """Test custom rate limit configuration."""
        config = RateLimitConfig(
            max_requests_per_minute=5,
            max_requests_per_hour=50,
            min_request_interval=2.0
        )
        
        result = wrap_llm_with_rate_limit(mock_openai_llm, config=config)
        
        assert result.rate_limiter.config.max_requests_per_minute == 5
        assert result.rate_limiter.config.max_requests_per_hour == 50
        assert result.rate_limiter.config.min_request_interval == 2.0


class TestLLMRateLimitingIntegration:
    """Integration tests for LLM rate limiting."""
    
    def test_real_rate_limiting_behavior(self, mock_openai_llm):
        """Test actual rate limiting behavior with timing."""
        # Create wrapper with very low limits for testing
        config = RateLimitConfig(
            max_requests_per_minute=2,
            min_request_interval=0.1
        )
        wrapper = RateLimitedLLM(mock_openai_llm, rate_limiter=None)
        wrapper.rate_limiter.config = config
        
        # Make rapid requests
        start_time = time.time()
        
        # First request should be immediate
        wrapper.invoke("request 1")
        
        # Second request should wait due to min_request_interval
        wrapper.invoke("request 2")
        
        end_time = time.time()
        
        # Should have taken at least the min_request_interval
        assert end_time - start_time >= 0.1
    
    def test_error_classification(self, mock_openai_llm):
        """Test that rate limit errors are properly classified."""
        wrapper = RateLimitedLLM(mock_openai_llm)
        
        # Mock rate limiter methods
        wrapper.rate_limiter.check_rate_limit = Mock(return_value=(True, None))
        wrapper.rate_limiter.record_error = Mock(return_value=0.1)
        wrapper.rate_limiter.config.max_retries = 1
        
        # Test rate limit error keywords
        rate_limit_errors = [
            Exception("rate limit exceeded"),
            Exception("429 Too Many Requests"),
            Exception("quota exceeded"),
            Exception("too many requests")
        ]
        
        for error in rate_limit_errors:
            wrapper.rate_limiter.record_error.reset_mock()
            mock_openai_llm.invoke = Mock(side_effect=error)
            
            with patch('time.sleep'):
                with pytest.raises(Exception):
                    wrapper.invoke("test")
            
            # Verify error was recorded
            wrapper.rate_limiter.record_error.assert_called()
    
    @patch('src.server_research_mcp.utils.llm_rate_limiter.logger')
    def test_logging_integration(self, mock_logger, mock_openai_llm):
        """Test that rate limiting events are properly logged."""
        wrapper = RateLimitedLLM(mock_openai_llm)
        
        # Verify initialization logging
        mock_logger.info.assert_called_with(f"Created rate-limited LLM wrapper for {wrapper.identifier}")
        
        # Test rate limit wait logging
        wrapper.rate_limiter.check_rate_limit = Mock(side_effect=[
            (False, 0.5),  # Rate limited
            (True, None)   # Allowed
        ])
        wrapper.rate_limiter.record_request = Mock()
        
        with patch('time.sleep'):
            wrapper.invoke("test")
        
        # Verify wait logging
        info_calls = [str(call) for call in mock_logger.info.call_args_list]
        assert any("Rate limit for" in call and "waiting" in call for call in info_calls)


class TestProviderSpecificBehavior:
    """Test provider-specific rate limiting behavior."""
    
    def test_openai_error_keywords(self, mock_openai_llm):
        """Test OpenAI-specific error keyword detection."""
        wrapper = RateLimitedLLM(mock_openai_llm)
        
        # OpenAI rate limit errors should be detected
        openai_errors = [
            "Rate limit reached for requests",
            "You exceeded your current quota",
            "Too many requests in 1 minute"
        ]
        
        for error_msg in openai_errors:
            assert any(keyword in error_msg.lower() 
                      for keyword in wrapper.rate_limiter.config.rate_limit_error_keywords)
    
    def test_anthropic_error_keywords(self, mock_anthropic_llm):
        """Test Anthropic-specific error keyword detection."""
        wrapper = RateLimitedLLM(mock_anthropic_llm)
        
        # Anthropic rate limit errors should be detected
        anthropic_errors = [
            "rate_limit_error",
            "Your credit balance is too low",
            "Request rate limit exceeded"
        ]
        
        for error_msg in anthropic_errors:
            # Should detect "rate" keyword at minimum
            assert any(keyword in error_msg.lower() 
                      for keyword in wrapper.rate_limiter.config.rate_limit_error_keywords)
    
    def test_provider_specific_defaults(self):
        """Test that different providers get appropriate default configurations."""
        # OpenAI LLM
        openai_llm = MockLLM("gpt-4", "openai")
        openai_wrapper = RateLimitedLLM(openai_llm)
        
        # Anthropic LLM  
        anthropic_llm = MockLLM("claude-3", "anthropic")
        anthropic_llm.__class__.__name__ = "AnthropicLLM"
        anthropic_wrapper = RateLimitedLLM(anthropic_llm)
        
        # Anthropic should have higher limits than OpenAI
        assert anthropic_wrapper.rate_limiter.config.max_requests_per_minute > openai_wrapper.rate_limiter.config.max_requests_per_minute
        assert anthropic_wrapper.rate_limiter.config.min_request_interval < openai_wrapper.rate_limiter.config.min_request_interval 