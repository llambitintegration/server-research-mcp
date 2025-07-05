# Rate Limiting Bug Fix Implementation Guide

## Quick Start Commands

```bash
# Set environment variable first
export RATE_LIMITING_DISABLED=true

# Run tests by cluster
pytest tests/unit/test_crew_decorators.py -v  # Cluster 1
pytest tests/unit/test_rate_limiting.py -v    # Cluster 2  
pytest tests/unit/test_mcp_registry.py -v     # Cluster 3
```

## Detailed Implementation Steps

### Step 1: Fix RateLimitedLLM Attribute Forwarding

**File**: `src/server_research_mcp/utils/llm_rate_limiter.py`

**Current Issue**: The wrapper doesn't expose `supports_stop_words` and other LLM attributes.

**Implementation**:
```python
class RateLimitedLLM:
    """Wrapper that adds rate limiting to LLM calls."""
    
    def __init__(self, llm: BaseLLM, config: Optional[RateLimitConfig] = None):
        self.llm = llm
        self.config = config or self._get_default_config()
        self.rate_limiter = RateLimiter(self.config)
        
        # List of attributes to explicitly forward
        FORWARD_ATTRS = [
            'model', 'temperature', 'max_tokens', 'supports_stop_words',
            'model_name', 'streaming', 'callbacks', 'cache', 'verbose',
            'metadata', 'tags', 'run_name', 'request_timeout'
        ]
        
        # Forward known attributes
        for attr in FORWARD_ATTRS:
            if hasattr(llm, attr):
                setattr(self, attr, getattr(llm, attr))
        
        # Also forward any model-specific attributes
        for attr in dir(llm):
            if not attr.startswith('_') and not hasattr(self, attr):
                try:
                    value = getattr(llm, attr)
                    # Only copy simple attributes, not methods
                    if not callable(value):
                        setattr(self, attr, value)
                except (AttributeError, TypeError):
                    pass
```

### Step 2: Add Test Mode Support

**File**: `src/server_research_mcp/utils/llm_rate_limiter.py`

**Add at module level**:
```python
import os
import logging

logger = logging.getLogger(__name__)

def is_rate_limiting_disabled() -> bool:
    """Check if rate limiting is disabled for testing."""
    return os.getenv('RATE_LIMITING_DISABLED', '').lower() == 'true'
```

**Update wrap function**:
```python
def wrap_llm_with_rate_limit(llm: BaseLLM, config: Optional[RateLimitConfig] = None) -> Union[BaseLLM, RateLimitedLLM]:
    """Wrap an LLM with rate limiting."""
    if is_rate_limiting_disabled():
        logger.debug("Rate limiting disabled for testing - returning unwrapped LLM")
        return llm
    
    if isinstance(llm, RateLimitedLLM):
        logger.debug("LLM already wrapped with rate limiting")
        return llm
    
    return RateLimitedLLM(llm, config)
```

### Step 3: Fix Tool Rate Limiting

**File**: `src/server_research_mcp/utils/rate_limiting/tool_wrapper.py`

**Add test mode check**:
```python
def get_rate_limited_tools(tools: List[BaseTool], tool_type: str = "default") -> List[BaseTool]:
    """Get rate limited versions of tools."""
    if os.getenv('RATE_LIMITING_DISABLED', '').lower() == 'true':
        logger.debug(f"Rate limiting disabled for {tool_type} tools")
        return tools
    
    # ... existing implementation ...
```

### Step 4: Restore mcp_manager Backward Compatibility

**File**: `src/server_research_mcp/tools/mcp_tools.py`

**Add at end of file**:
```python
# Backward compatibility support
import warnings

class _MCPManagerCompatibility:
    """Compatibility wrapper for deprecated mcp_manager."""
    
    def __init__(self):
        self._registry = None
    
    @property
    def registry(self):
        if self._registry is None:
            self._registry = MCPToolRegistry()
        return self._registry
    
    def __getattr__(self, name):
        warnings.warn(
            f"mcp_manager.{name} is deprecated. Use MCPToolRegistry directly.",
            DeprecationWarning,
            stacklevel=2
        )
        return getattr(self.registry, name)

# Create module-level instance
mcp_manager = _MCPManagerCompatibility()

# Also expose it in __all__ for imports
__all__ = ['MCPToolRegistry', 'mcp_manager']  # Add to existing __all__ if present
```

### Step 5: Update Test Fixtures

**File**: `tests/conftest.py`

**Add at the beginning**:
```python
import os
import pytest
from unittest.mock import Mock, patch

# Ensure rate limiting is disabled for all tests
os.environ['RATE_LIMITING_DISABLED'] = 'true'

@pytest.fixture(autouse=True, scope="session")
def disable_rate_limiting():
    """Ensure rate limiting is disabled for entire test session."""
    os.environ['RATE_LIMITING_DISABLED'] = 'true'
    yield
    # Don't remove - other tests might need it

@pytest.fixture
def mock_llm():
    """Provide a properly mocked LLM for tests."""
    from langchain_core.language_models import BaseLLM
    
    mock = Mock(spec=BaseLLM)
    mock.model = "gpt-4"
    mock.temperature = 0.7
    mock.max_tokens = 4000
    mock.supports_stop_words = True
    mock.invoke.return_value = "Mocked response"
    mock.generate.return_value = Mock(generations=[[Mock(text="Mocked response")]])
    
    return mock

@pytest.fixture  
def no_rate_limit_env():
    """Explicitly disable rate limiting for a specific test."""
    original = os.environ.get('RATE_LIMITING_DISABLED')
    os.environ['RATE_LIMITING_DISABLED'] = 'true'
    yield
    if original is None:
        os.environ.pop('RATE_LIMITING_DISABLED', None)
    else:
        os.environ['RATE_LIMITING_DISABLED'] = original
```

### Step 6: Update crew.py for Better Error Handling

**File**: `src/server_research_mcp/crew.py`

**In _create_agent_from_definition method**:
```python
# Get rate-limited LLM instance
llm_instance = self.llm_config.get_llm()

# Check if we should wrap with rate limiting
should_wrap = (
    not hasattr(llm_instance, 'rate_limiter') and
    os.getenv('RATE_LIMITING_DISABLED', '').lower() != 'true'
)

if should_wrap:
    try:
        from .utils.llm_rate_limiter import wrap_llm_with_rate_limit
        llm_instance = wrap_llm_with_rate_limit(llm_instance)
        logger.info(f"Wrapped LLM with rate limiting for {definition.name}")
    except Exception as e:
        logger.warning(f"Failed to wrap LLM with rate limiting: {e}")
        # Continue with unwrapped LLM
```

## Testing Strategy

### 1. Unit Test Updates

For tests that were mocking LLMs directly:

```python
# OLD
@patch('langchain_openai.ChatOpenAI')
def test_something(mock_llm):
    mock_llm.return_value = Mock(supports_stop_words=True)
    
# NEW  
def test_something(mock_llm):  # Use fixture from conftest
    # mock_llm already has supports_stop_words=True
```

### 2. Integration Test Updates

```python
# Add to integration test files
import os
os.environ['RATE_LIMITING_DISABLED'] = 'true'

# Or use the fixture
def test_integration(no_rate_limit_env):
    # Test code here
    pass
```

### 3. Verify No Real API Calls

Add to test files:
```python
@patch('requests.post')
@patch('httpx.post')  
def test_no_api_calls(mock_httpx, mock_requests):
    # Run test
    crew.kickoff()
    
    # Verify no real API calls
    mock_requests.assert_not_called()
    mock_httpx.assert_not_called()
```

## Rollback Plan

If fixes cause issues:

1. **Immediate**: Set `RATE_LIMITING_DISABLED=true` in production
2. **Revert commits**: Git revert the rate limiting changes
3. **Hotfix**: Deploy version with rate limiting removed
4. **Re-implement**: Fix issues and re-deploy with better testing

## Monitoring

Add logging to track:
- When rate limiting is disabled
- When wrappers are created
- When deprecated APIs are used

```python
logger.info(f"Rate limiting status: {'disabled' if is_rate_limiting_disabled() else 'enabled'}")
logger.info(f"Creating {'wrapped' if should_wrap else 'unwrapped'} LLM for {agent_name}")
logger.warning(f"Deprecated API usage: {api_name}")
```
