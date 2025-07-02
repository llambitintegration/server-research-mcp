# Crew Freeze Issue - Diagnosis and Fix

## Issue Summary
The crew was freezing during LLM API calls to Anthropic, specifically during streaming response handling. The freeze occurred in the HTTP layer (`httpx`/`httpcore`), not in tool execution.

## Root Causes Identified

### 1. LLM API Timeout Issues
- **Problem**: No request timeouts configured for LLM API calls
- **Symptom**: Crew freezes during `httpx._models.py:929 in iter_lines`
- **Impact**: Indefinite hangs waiting for Anthropic API responses

### 2. Parameter Processing Bug  
- **Problem**: MCPToolWrapper incorrectly converted legitimate parameters to 'query'
- **Example**: `{'path': 'README.md'}` became `{'query': 'README.md'}`
- **Impact**: Tools received wrong parameters, potentially confusing LLM

## Fixes Applied

### 1. LLM Timeout Configuration
**Files:** `src/server_research_mcp/config/llm_config.py`, `src/server_research_mcp/crew.py`

Added configurable timeouts and consolidated LLM configuration:

```python
# Environment variables (optional)
LLM_REQUEST_TIMEOUT=60    # Request timeout in seconds (default: 60)
LLM_MAX_RETRIES=3         # Number of retries (default: 3)
LLM_STREAMING=false       # Enable/disable streaming (default: false)
```

**Key Changes:**
- Consolidated duplicate `get_configured_llm()` functions
- Applied timeout settings consistently across all agents
- Disabled streaming by default to prevent hangs

### 2. Agent Execution Limits
**File:** `src/server_research_mcp/crew.py`

Reduced iteration limits and added execution timeouts for all agents:
- `max_iter=1` (reduced from 2) to prevent iteration loops
- `execution_timeout=90` seconds per agent execution
- Applied to: historian, researcher, archivist, publisher

### 3. Parameter Processing Fix
**File:** `src/server_research_mcp/tools/mcp_tools.py`

Updated MCPToolWrapper to preserve legitimate parameter names:
- `path`, `limit`, `count`, `size`, `max_results`, `num_results`
- `qmode`, `tag`, `filename`, `directory`, `content`, `data`

## Verification
- ✅ All MCP tools respond quickly (<1s)
- ✅ No hanging tools detected
- ✅ Parameter processing working correctly
- ✅ LLM timeout configuration active

## Usage
The fixes are automatic. Optionally, you can configure timeouts:

```bash
# Set longer timeout for complex operations
export LLM_REQUEST_TIMEOUT=120

# Enable streaming if preferred (may reintroduce hanging risk)
export LLM_STREAMING=true

# Adjust retry behavior
export LLM_MAX_RETRIES=5
```

## Recommendations
1. **Keep streaming disabled** unless specifically needed
2. **Monitor LLM response times** - adjust timeout if needed
3. **Use shorter timeouts** for development (30-60s)
4. **Use longer timeouts** for production with complex operations (120s+)

## Next Steps
If freezing still occurs:
1. Check network connectivity to Anthropic API
2. Verify API key and rate limits
3. Consider switching to OpenAI provider temporarily
4. Monitor system resources (memory/CPU) 