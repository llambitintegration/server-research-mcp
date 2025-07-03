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
    
    def __init__(self):
        self.provider = None
        self.model = None
        self.api_key = None
        self._configure()
    
    def _configure(self):
        """Configure LLM based purely on environment variables."""
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
        streaming = False
        if streaming is None:
            raise ValueError("LLM_STREAMING environment variable is required (set to 'true' or 'false')")

        # Sanitize numeric values to allow inline comments (e.g., "120  # note")
        timeout_int = _parse_int_env(timeout, 'LLM_REQUEST_TIMEOUT')
        max_retries_int = _parse_int_env(max_retries, 'LLM_MAX_RETRIES')

        return LLM(
            model=f"{self.provider}/{self.model}",
            api_key=self.api_key,
            timeout=timeout_int,
            max_retries=max_retries_int,
            streaming=streaming.lower() == 'true'
        )
    
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
        return LLM(
            model=config['model'],
            api_key=config['api_key']
        )