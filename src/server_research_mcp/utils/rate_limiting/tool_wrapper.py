"""Rate-limited wrapper for CrewAI tools."""

import time
import logging
from typing import Any, Dict, List, Optional, Union
from functools import wraps
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import os

from .rate_limiter import RateLimiter, RateLimitConfig, RateLimitError, create_rate_limiter
from .backoff_strategies import ExponentialBackoff

logger = logging.getLogger(__name__)


class RateLimitedTool(BaseTool):
    """Wrapper that adds rate limiting to any CrewAI tool."""
    
    # Declare fields that will be set in __init__
    wrapped_tool: Optional[BaseTool] = None
    rate_limiter: Optional[RateLimiter] = None
    identifier: Optional[str] = None
    
    def __init__(
        self,
        tool: BaseTool,
        rate_limiter: Optional[RateLimiter] = None,
        identifier: Optional[str] = None,
        **kwargs
    ):
        # Handle args_schema
        if hasattr(tool, 'args_schema') and tool.args_schema:
            args_schema = tool.args_schema
        else:
            # Create a simple schema if none exists
            class GenericArgs(BaseModel):
                query: str = Field(default="", description="Input query")
            args_schema = GenericArgs
        
        # Set up rate limiting
        rate_limiter = rate_limiter or create_rate_limiter(
            f"tool_{tool.name}",
            RateLimitConfig(
                max_requests_per_minute=20,
                max_requests_per_hour=200,
                min_request_interval=0.5,
                max_retries=5,
                initial_retry_delay=1.0,
                backoff_factor=2.0
            )
        )
        identifier = identifier or tool.name
        
        # Initialize parent with required fields
        super().__init__(
            name=f"rate_limited_{tool.name}",
            description=tool.description,
            args_schema=args_schema,
            wrapped_tool=tool,
            rate_limiter=rate_limiter,
            identifier=identifier,
            **kwargs
        )
    
    def _run(self, *args, **kwargs) -> Any:
        """Execute the wrapped tool with rate limiting."""
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
                
                # Log the attempt
                logger.debug(f"Executing {self.identifier} (attempt {attempt})")
                
                # Execute the wrapped tool
                if hasattr(self.wrapped_tool, '_run'):
                    result = self.wrapped_tool._run(*args, **kwargs)
                elif callable(self.wrapped_tool):
                    result = self.wrapped_tool(*args, **kwargs)
                else:
                    # Handle other tool types
                    result = str(self.wrapped_tool)
                
                # Record success
                self.rate_limiter.record_request(self.identifier)
                
                # Reset retry counter on success
                if attempt > 1:
                    logger.info(f"{self.identifier} succeeded after {attempt} attempts")
                
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
                    logger.warning(f"Rate limit error for {self.identifier}: {e}")
                else:
                    logger.error(f"Error in {self.identifier}: {e}")
                
                # Calculate retry delay
                delay = self.rate_limiter.record_error(self.identifier, e)
                
                # Check if we should retry
                if attempt >= self.rate_limiter.config.max_retries:
                    logger.error(f"Max retries ({attempt}) exceeded for {self.identifier}")
                    break
                
                logger.info(f"Retrying {self.identifier} in {delay:.2f}s (attempt {attempt}/{self.rate_limiter.config.max_retries})")
                time.sleep(delay)
        
        # Raise the last error if all retries failed
        if last_error:
            raise last_error
        else:
            raise RateLimitError(f"Failed to execute {self.identifier} after {attempt} attempts")
    
    async def _arun(self, *args, **kwargs) -> Any:
        """Async version - falls back to sync for now."""
        # TODO: Implement proper async rate limiting
        return self._run(*args, **kwargs)


def wrap_tool_with_rate_limit(
    tool: BaseTool,
    config: Optional[Union[RateLimitConfig, Dict[str, Any]]] = None,
    rate_limiter: Optional[RateLimiter] = None,
    identifier: Optional[str] = None
) -> RateLimitedTool:
    """
    Wrap a CrewAI tool with rate limiting.
    
    Args:
        tool: The tool to wrap
        config: Rate limit configuration (dict or RateLimitConfig)
        rate_limiter: Existing rate limiter to use
        identifier: Custom identifier for the tool
    
    Returns:
        Rate-limited tool wrapper
    """
    # Create rate limiter if not provided
    if rate_limiter is None:
        if config is None:
            # Default configuration
            config = RateLimitConfig(
                max_requests_per_minute=30,
                max_requests_per_hour=300,
                min_request_interval=0.2,
                max_retries=5,
                initial_retry_delay=1.0,
                backoff_factor=2.0
            )
        elif isinstance(config, dict):
            config = RateLimitConfig(**config)
        
        rate_limiter = RateLimiter(config)
    
    return RateLimitedTool(tool, rate_limiter, identifier)


def wrap_tools_with_rate_limit(
    tools: List[BaseTool],
    shared_limiter: bool = False,
    config: Optional[Union[RateLimitConfig, Dict[str, Any]]] = None,
    per_tool_config: Optional[Dict[str, Union[RateLimitConfig, Dict[str, Any]]]] = None
) -> List[RateLimitedTool]:
    """
    Wrap multiple tools with rate limiting.
    
    Args:
        tools: List of tools to wrap
        shared_limiter: If True, all tools share the same rate limiter
        config: Default configuration for all tools
        per_tool_config: Per-tool configuration overrides
    
    Returns:
        List of rate-limited tools
    """
    wrapped_tools = []
    
    # Create shared limiter if requested
    if shared_limiter:
        if config is None:
            config = RateLimitConfig()
        elif isinstance(config, dict):
            config = RateLimitConfig(**config)
        
        shared_rate_limiter = RateLimiter(config)
    else:
        shared_rate_limiter = None
    
    # Wrap each tool
    for tool in tools:
        # Get tool-specific config if available
        tool_config = None
        if per_tool_config and tool.name in per_tool_config:
            tool_config = per_tool_config[tool.name]
            if isinstance(tool_config, dict):
                tool_config = RateLimitConfig(**tool_config)
        else:
            tool_config = config
        
        # Use shared limiter or create individual one
        if shared_limiter:
            rate_limiter = shared_rate_limiter
        else:
            if tool_config is None:
                tool_config = RateLimitConfig()
            elif isinstance(tool_config, dict):
                tool_config = RateLimitConfig(**tool_config)
            
            rate_limiter = RateLimiter(tool_config)
        
        wrapped_tool = RateLimitedTool(tool, rate_limiter)
        wrapped_tools.append(wrapped_tool)
    
    return wrapped_tools


def get_rate_limit_config_from_env(prefix: str = "MCP") -> Dict[str, Any]:
    """Get rate limit configuration from environment variables."""
    config = {}
    
    # Check for rate limiting env vars
    env_vars = {
        f"{prefix}_MAX_REQUESTS_PER_MINUTE": "max_requests_per_minute",
        f"{prefix}_MAX_REQUESTS_PER_HOUR": "max_requests_per_hour",
        f"{prefix}_MIN_REQUEST_INTERVAL": "min_request_interval",
        f"{prefix}_MAX_RETRIES": "max_retries",
        f"{prefix}_INITIAL_RETRY_DELAY": "initial_retry_delay",
        f"{prefix}_MAX_RETRY_DELAY": "max_retry_delay",
        f"{prefix}_BACKOFF_FACTOR": "backoff_factor",
    }
    
    for env_var, config_key in env_vars.items():
        value = os.getenv(env_var)
        if value:
            try:
                # Convert to appropriate type
                if config_key in ["max_requests_per_minute", "max_requests_per_hour", "max_retries"]:
                    config[config_key] = int(value)
                else:
                    config[config_key] = float(value)
            except ValueError:
                logger.warning(f"Invalid value for {env_var}: {value}")
    
    return config


def get_rate_limited_tools(
    tools: List[BaseTool],
    tool_type: str = "default"
) -> List[BaseTool]:
    """
    Get rate-limited versions of tools based on tool type.
    
    Args:
        tools: Original tools
        tool_type: Type of tools (e.g., "zotero", "memory", "filesystem")
    
    Returns:
        Rate-limited tools
    """
    # Get environment configuration
    env_config = get_rate_limit_config_from_env(f"MCP_{tool_type.upper()}")
    
    # Define default configurations for different tool types
    default_configs = {
        "zotero": {
            "max_requests_per_minute": 10,  # Zotero API limit
            "max_requests_per_hour": 100,
            "min_request_interval": 1.0,
            "max_retries": 5,
            "initial_retry_delay": 2.0,
            "backoff_factor": 2.0,
            "rate_limit_error_keywords": [
                "rate limit", "429", "too many requests",
                "api limit", "quota exceeded"
            ]
        },
        "memory": {
            "max_requests_per_minute": 60,
            "max_requests_per_hour": 600,
            "min_request_interval": 0.1,
            "max_retries": 3,
            "initial_retry_delay": 0.5
        },
        "filesystem": {
            "max_requests_per_minute": 100,
            "max_requests_per_hour": 1000,
            "min_request_interval": 0.05,
            "max_retries": 3,
            "initial_retry_delay": 0.2
        },
        "default": {
            "max_requests_per_minute": 30,
            "max_requests_per_hour": 300,
            "min_request_interval": 0.5,
            "max_retries": 4,
            "initial_retry_delay": 1.0
        }
    }
    
    # Merge default config with environment overrides
    base_config = default_configs.get(tool_type, default_configs["default"])
    final_config = {**base_config, **env_config}
    
    # Create RateLimitConfig instance
    config = RateLimitConfig(**final_config)
    
    return wrap_tools_with_rate_limit(tools, shared_limiter=True, config=config)
