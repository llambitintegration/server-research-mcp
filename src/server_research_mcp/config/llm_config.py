"""LLM Configuration Module - Centralized LLM setup for the application."""

import os
from typing import Optional
from dotenv import load_dotenv
from crewai import LLM

# Load environment variables
load_dotenv()

# Utility helpers -----------------------------------------------------------

def _parse_int_env(value: str | None, var_name: str) -> int:
    """Parse an integer environment variable allowing inline comments.

    The value may be something like "120  # longer timeout". This helper will
    strip off anything after a '#' or whitespace and convert the first token to
    an int.
    """
    if value is None:
        raise ValueError(f"{var_name} environment variable is required")
    # Remove inline comment if present
    cleaned = value.split('#', 1)[0].strip()
    try:
        return int(cleaned)
    except ValueError as e:
        raise ValueError(f"{var_name} must be an integer, got '{value}'") from e


class LLMConfig:
    """Centralized LLM configuration management - entirely environment-driven."""
    
    def __init__(self, test_mode: bool = False):
        self.provider = None
        self.model = None
        self.api_key = None
        self.test_mode = test_mode
        self._configure()
    
    def _configure(self):
        """Configure LLM based purely on environment variables."""
        # In test mode, provide default configuration
        if self.test_mode:
            self.provider = 'test'
            self.model = 'test/mock-model'
            self.api_key = 'test-key-for-validation'
            return
        
        # Provider must be explicitly set in environment
        self.provider = os.getenv('LLM_PROVIDER')
        if not self.provider:
            raise ValueError("LLM_PROVIDER environment variable is required. Set to 'anthropic', 'openai', or other supported provider")
        
        self.provider = self.provider.lower()
        
        # Model must be explicitly set in environment
        self.model = os.getenv('LLM_MODEL')
        if not self.model:
            raise ValueError("LLM_MODEL environment variable is required. Examples: 'claude-3-haiku-20240307', 'gpt-4o-mini', etc.")
        
        # API key must be explicitly set in environment
        if self.provider == 'anthropic':
            self.api_key = os.getenv('ANTHROPIC_API_KEY')
            if not self.api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable is required when using Anthropic provider")
        elif self.provider == 'openai':
            self.api_key = os.getenv('OPENAI_API_KEY')
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY environment variable is required when using OpenAI provider")
        else:
            # For other providers, use generic LLM_API_KEY
            self.api_key = os.getenv('LLM_API_KEY')
            if not self.api_key:
                raise ValueError(f"LLM_API_KEY environment variable is required for provider '{self.provider}'")
    
    def get_llm(self) -> LLM:
        """Get configured LLM instance with all settings from environment."""
        # All timeout and retry settings must come from environment
        timeout = os.getenv('LLM_REQUEST_TIMEOUT')
        max_retries = os.getenv('LLM_MAX_RETRIES')
        streaming = os.getenv('LLM_STREAMING')
        max_tokens = os.getenv('LLM_MAX_TOKENS', '8192')  # Default to 8192 for research papers
        
        if streaming is None:
            raise ValueError("LLM_STREAMING environment variable is required (set to 'true' or 'false')")

        # Sanitize numeric values to allow inline comments (e.g., "120  # note")
        timeout_int = _parse_int_env(timeout, 'LLM_REQUEST_TIMEOUT')
        max_retries_int = _parse_int_env(max_retries, 'LLM_MAX_RETRIES')
        max_tokens_int = _parse_int_env(max_tokens, 'LLM_MAX_TOKENS')

        # Disable streaming for Anthropic provider due to LiteLLM compatibility issues
        effective_streaming = streaming.lower() == 'true'
        if self.provider == 'anthropic' and effective_streaming:
            effective_streaming = False

        # Log LLM configuration before creation
        import logging
        logger = logging.getLogger(__name__)
        
        llm_config = {
            "model": f"{self.provider}/{self.model}",
            "timeout": timeout_int,
            "max_retries": max_retries_int,
            "max_tokens": max_tokens_int,
            "stream": effective_streaming,
            "provider": self.provider
        }
        
        logger.info("Creating LLM instance", extra=llm_config)
        
        try:
            llm_instance = LLM(
                model=f"{self.provider}/{self.model}",
                api_key=self.api_key,
                timeout=timeout_int,
                max_retries=max_retries_int,
                max_tokens=max_tokens_int,
                stream=effective_streaming
            )
            
            logger.info("LLM instance created successfully", extra={
                "model": f"{self.provider}/{self.model}",
                "instance_type": type(llm_instance).__name__
            })
            
            return llm_instance
            
        except Exception as e:
            logger.error("Failed to create LLM instance", extra={
                "model": f"{self.provider}/{self.model}",
                "error": str(e),
                "error_type": type(e).__name__,
                "config": llm_config
            }, exc_info=True)
            raise
    
    def check_configuration(self) -> tuple[bool, Optional[str]]:
        """Check if LLM is properly configured."""
        try:
            required_vars = ['LLM_PROVIDER', 'LLM_MODEL', 'LLM_REQUEST_TIMEOUT', 'LLM_MAX_RETRIES', 'LLM_STREAMING']
            missing_vars = []
            
            for var in required_vars:
                if not os.getenv(var):
                    missing_vars.append(var)
            
            # Check provider-specific API key
            if self.provider == 'anthropic' and not os.getenv('ANTHROPIC_API_KEY'):
                missing_vars.append('ANTHROPIC_API_KEY')
            elif self.provider == 'openai' and not os.getenv('OPENAI_API_KEY'):
                missing_vars.append('OPENAI_API_KEY')
            elif self.provider not in ['anthropic', 'openai'] and not os.getenv('LLM_API_KEY'):
                missing_vars.append('LLM_API_KEY')
            
            if missing_vars:
                return False, f"Missing required environment variables: {', '.join(missing_vars)}"
            
            return True, None
        except Exception as e:
            return False, str(e)


# Global instance
llm_config = LLMConfig()


def get_configured_llm() -> LLM:
    """Get the configured LLM instance."""
    return llm_config.get_llm()


def check_llm_config() -> tuple[bool, Optional[str]]:
    """Check if LLM configuration is valid."""
    return llm_config.check_configuration()


def get_llm_config() -> dict:
    """Get LLM configuration as dictionary (for tests)."""
    return {
        'provider': llm_config.provider,
        'model': f"{llm_config.provider}/{llm_config.model}",
        'api_key': llm_config.api_key
    }


def create_llm(config: dict = None) -> LLM:
    """Create LLM instance from configuration (for tests)."""
    if config is None:
        return llm_config.get_llm()
    else:
        # Add validation for test compatibility
        if not config.get('model'):
            raise ValueError("Model configuration is required")
        if not config.get('api_key'):
            raise ValueError("API key is required")
        
        return LLM(
            model=config['model'],
            api_key=config['api_key']
        )


def create_test_llm_config() -> dict:
    """Create a test-compatible LLM configuration with shorter responses."""
    return {
        'provider': 'test',
        'model': 'test/mock-model',
        'api_key': 'test-api-key-12345',
        'max_response_length': 80,  # Ensure responses are under 100 chars for tests
        'timeout': 30,
        'max_retries': 2
    }


def validate_test_config(config: dict) -> bool:
    """Validate test configuration and raise exceptions for invalid configs."""
    if not config:
        raise ValueError("Configuration cannot be empty")
    
    required_fields = ['model', 'api_key']
    for field in required_fields:
        if not config.get(field):
            raise ValueError(f"Missing required field: {field}")
    
    # Validate model format
    if '/' not in config['model']:
        raise ValueError("Model must be in format 'provider/model'")
    
    return True