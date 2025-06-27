"""LLM Configuration Module - Centralized LLM setup for the application."""

import os
from typing import Optional
from dotenv import load_dotenv
from crewai import LLM

# Load environment variables
load_dotenv()


class LLMConfig:
    """Centralized LLM configuration management."""
    
    def __init__(self):
        self.provider = os.getenv('LLM_PROVIDER', 'anthropic').lower()
        self.model = None
        self.api_key = None
        self._configure()
    
    def _configure(self):
        """Configure LLM based on provider."""
        if self.provider == 'anthropic':
            self.api_key = os.getenv('ANTHROPIC_API_KEY')
            if not self.api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable is required when using Anthropic provider")
            self.model = os.getenv('LLM_MODEL', 'claude-3-haiku-20240307')
            
        elif self.provider == 'openai':
            self.api_key = os.getenv('OPENAI_API_KEY')
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY environment variable is required when using OpenAI provider")
            self.model = os.getenv('LLM_MODEL', 'gpt-4o-mini')
            
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}. Use 'anthropic' or 'openai'")
    
    def get_llm(self) -> LLM:
        """Get configured LLM instance."""
        return LLM(
            model=f"{self.provider}/{self.model}",
            api_key=self.api_key
        )
    
    def check_configuration(self) -> tuple[bool, Optional[str]]:
        """Check if LLM is properly configured."""
        try:
            if not self.api_key:
                return False, f"{self.provider.upper()}_API_KEY not found"
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