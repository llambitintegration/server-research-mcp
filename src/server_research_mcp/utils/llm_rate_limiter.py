"""
LLM Rate Limiting Module

Provides rate limiting wrappers for LangChain LLM instances to prevent API rate limit errors.
Supports auto-detection of LLM providers and environment variable configuration.
"""

import os
import time
import logging
from typing import Any, Dict, List, Optional, Union
from functools import wraps

from .rate_limiting import RateLimiter, RateLimitConfig, RateLimitError

logger = logging.getLogger(__name__)


class RateLimitedLLM:
    """
    Rate-limited wrapper for LangChain LLM instances.
    
    Wraps invoke, generate, and predict methods with rate limiting to prevent
    API rate limit errors. Auto-detects provider for appropriate defaults.
    """
    
    def __init__(
        self,
        llm: Any,
        rate_limiter: Optional[RateLimiter] = None,
        identifier: Optional[str] = None
    ):
        """
        Initialize rate-limited LLM wrapper.
        
        Args:
            llm: LangChain LLM instance to wrap
            rate_limiter: Optional rate limiter instance
            identifier: Optional identifier for rate limiting
        """
        self.wrapped_llm = llm
        self.identifier = identifier or self._get_llm_identifier(llm)
        
        # Set up rate limiting
        if rate_limiter is None:
            config = self._get_rate_limit_config_for_llm(llm)
            self.rate_limiter = RateLimiter(config)
        else:
            self.rate_limiter = rate_limiter
        
        # Copy essential attributes from wrapped LLM
        self._copy_llm_attributes(llm)
        
        # Mark as rate limited to prevent double-wrapping
        self.rate_limiter_applied = True
        
        logger.info(f"Created rate-limited LLM wrapper for {self.identifier}")
    
    def _get_llm_identifier(self, llm: Any) -> str:
        """Get identifier for the LLM instance."""
        if hasattr(llm, 'model_name'):
            return f"llm_{llm.model_name}"
        elif hasattr(llm, 'model'):
            return f"llm_{llm.model}"
        elif hasattr(llm, '__class__'):
            return f"llm_{llm.__class__.__name__}"
        else:
            return "llm_unknown"
    
    def _get_rate_limit_config_for_llm(self, llm: Any) -> RateLimitConfig:
        """Get rate limit configuration based on LLM provider."""
        # Detect provider
        provider = self._detect_provider(llm)
        
        # Get environment overrides
        env_config = self._get_env_config()
        
        # Provider-specific defaults
        if provider == "openai":
            base_config = {
                "max_requests_per_minute": 20,
                "max_requests_per_hour": 200,
                "min_request_interval": 0.5,
                "max_retries": 5,
                "initial_retry_delay": 1.0,
                "backoff_factor": 2.0,
                "rate_limit_error_keywords": [
                    "rate limit", "rate_limit", "429", "too many requests",
                    "quota exceeded", "api limit", "requests per minute"
                ]
            }
        elif provider == "anthropic":
            base_config = {
                "max_requests_per_minute": 50,
                "max_requests_per_hour": 500,
                "min_request_interval": 0.2,
                "max_retries": 5,
                "initial_retry_delay": 1.0,
                "backoff_factor": 2.0,
                "rate_limit_error_keywords": [
                    "rate limit", "rate_limit", "429", "too many requests",
                    "quota exceeded", "api limit", "requests per minute"
                ]
            }
        else:
            # Default configuration
            base_config = {
                "max_requests_per_minute": 30,
                "max_requests_per_hour": 300,
                "min_request_interval": 0.5,
                "max_retries": 4,
                "initial_retry_delay": 1.0,
                "backoff_factor": 2.0
            }
        
        # Merge with environment overrides
        final_config = {**base_config, **env_config}
        
        return RateLimitConfig(**final_config)
    
    def _detect_provider(self, llm: Any) -> str:
        """Detect LLM provider from class name or attributes."""
        llm_class = llm.__class__.__name__.lower()
        
        if "openai" in llm_class or "gpt" in llm_class:
            return "openai"
        elif "anthropic" in llm_class or "claude" in llm_class:
            return "anthropic"
        elif hasattr(llm, 'model_name'):
            model_name = llm.model_name.lower()
            if "gpt" in model_name or "openai" in model_name:
                return "openai"
            elif "claude" in model_name or "anthropic" in model_name:
                return "anthropic"
        
        return "unknown"
    
    def _get_env_config(self) -> Dict[str, Any]:
        """Get rate limit configuration from environment variables."""
        config = {}
        
        env_vars = {
            "LLM_MAX_REQUESTS_PER_MINUTE": "max_requests_per_minute",
            "LLM_MAX_REQUESTS_PER_HOUR": "max_requests_per_hour",
            "LLM_MIN_REQUEST_INTERVAL": "min_request_interval",
            "LLM_MAX_RETRIES": "max_retries",
            "LLM_INITIAL_RETRY_DELAY": "initial_retry_delay",
            "LLM_MAX_RETRY_DELAY": "max_retry_delay",
            "LLM_BACKOFF_FACTOR": "backoff_factor",
        }
        
        for env_var, config_key in env_vars.items():
            value = os.getenv(env_var)
            if value:
                try:
                    if config_key in ["max_requests_per_minute", "max_requests_per_hour", "max_retries"]:
                        config[config_key] = int(value)
                    else:
                        config[config_key] = float(value)
                except ValueError:
                    logger.warning(f"Invalid value for {env_var}: {value}")
        
        return config
    
    def _copy_llm_attributes(self, llm: Any):
        """Copy essential attributes from the wrapped LLM."""
        # Common LLM attributes to copy
        attributes_to_copy = [
            'model', 'model_name', 'temperature', 'max_tokens', 'top_p',
            'frequency_penalty', 'presence_penalty', 'stop', 'streaming',
            'callbacks', 'tags', 'metadata', 'verbose'
        ]
        
        for attr in attributes_to_copy:
            if hasattr(llm, attr):
                setattr(self, attr, getattr(llm, attr))
    
    def _rate_limited_call(self, method_name: str, *args, **kwargs):
        """Execute a rate-limited call to the wrapped LLM."""
        last_error = None
        attempt = 0
        
        while attempt < self.rate_limiter.config.max_retries:
            attempt += 1
            
            try:
                # Check rate limit
                allowed, wait_time = self.rate_limiter.check_rate_limit(self.identifier)
                if not allowed and wait_time:
                    logger.info(f"Rate limit for {self.identifier}: waiting {wait_time:.2f}s")
                    time.sleep(wait_time)
                    continue
                
                # Get the method from wrapped LLM
                method = getattr(self.wrapped_llm, method_name)
                
                # Execute the method
                result = method(*args, **kwargs)
                
                # Record success
                self.rate_limiter.record_request(self.identifier)
                
                if attempt > 1:
                    logger.info(f"{self.identifier} {method_name} succeeded after {attempt} attempts")
                
                return result
                
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                
                # Check if it's a rate limit error
                is_rate_limit = any(
                    keyword in error_str 
                    for keyword in self.rate_limiter.config.rate_limit_error_keywords
                )
                
                if is_rate_limit:
                    logger.warning(f"Rate limit error for {self.identifier} {method_name}: {e}")
                else:
                    logger.error(f"Error in {self.identifier} {method_name}: {e}")
                
                # Calculate retry delay
                delay = self.rate_limiter.record_error(self.identifier, e)
                
                # Check if we should retry
                if attempt >= self.rate_limiter.config.max_retries:
                    logger.error(f"Max retries ({attempt}) exceeded for {self.identifier} {method_name}")
                    break
                
                logger.info(f"Retrying {self.identifier} {method_name} in {delay:.2f}s (attempt {attempt}/{self.rate_limiter.config.max_retries})")
                time.sleep(delay)
        
        # Raise the last error if all retries failed
        if last_error:
            raise last_error
        else:
            raise RateLimitError(f"Failed to execute {self.identifier} {method_name} after {attempt} attempts")
    
    def invoke(self, *args, **kwargs):
        """Rate-limited invoke method."""
        return self._rate_limited_call('invoke', *args, **kwargs)
    
    def generate(self, *args, **kwargs):
        """Rate-limited generate method."""
        return self._rate_limited_call('generate', *args, **kwargs)
    
    def predict(self, *args, **kwargs):
        """Rate-limited predict method."""
        return self._rate_limited_call('predict', *args, **kwargs)
    
    def __call__(self, *args, **kwargs):
        """Rate-limited call method."""
        return self._rate_limited_call('__call__', *args, **kwargs)
    
    def __getattr__(self, name):
        """Proxy other attributes to the wrapped LLM."""
        return getattr(self.wrapped_llm, name)


def wrap_llm_with_rate_limit(
    llm: Any,
    config: Optional[Union[RateLimitConfig, Dict[str, Any]]] = None,
    rate_limiter: Optional[RateLimiter] = None,
    identifier: Optional[str] = None
) -> RateLimitedLLM:
    """
    Wrap an LLM with rate limiting.
    
    Args:
        llm: LangChain LLM instance to wrap
        config: Rate limit configuration (dict or RateLimitConfig)
        rate_limiter: Existing rate limiter to use
        identifier: Custom identifier for the LLM
    
    Returns:
        Rate-limited LLM wrapper
    """
    # Check if already rate limited
    if hasattr(llm, 'rate_limiter_applied') and llm.rate_limiter_applied:
        logger.info(f"LLM {identifier or 'unknown'} already rate limited, returning as-is")
        return llm
    
    # Create rate limiter if not provided
    if rate_limiter is None and config is not None:
        if isinstance(config, dict):
            config = RateLimitConfig(**config)
        rate_limiter = RateLimiter(config)
    
    return RateLimitedLLM(llm, rate_limiter, identifier)


def get_rate_limited_llm(
    llm: Any,
    provider: Optional[str] = None,
    custom_config: Optional[Dict[str, Any]] = None
) -> RateLimitedLLM:
    """
    Get a rate-limited version of an LLM with provider-specific defaults.
    
    Args:
        llm: LangChain LLM instance
        provider: Optional provider override ("openai", "anthropic")
        custom_config: Optional custom configuration overrides
    
    Returns:
        Rate-limited LLM
    """
    # Check if already rate limited
    if hasattr(llm, 'rate_limiter_applied') and llm.rate_limiter_applied:
        logger.info("LLM already rate limited, returning as-is")
        return llm
    
    # Create wrapper (will auto-detect provider if not specified)
    wrapper = RateLimitedLLM(llm)
    
    # Apply custom config if provided
    if custom_config:
        for key, value in custom_config.items():
            if hasattr(wrapper.rate_limiter.config, key):
                setattr(wrapper.rate_limiter.config, key, value)
    
    return wrapper 