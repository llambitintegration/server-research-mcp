# Context7 MCP Server API Fix Summary

## Problem Identified
The Context7 MCP server was failing with JSON parsing errors and communication issues because:

1. **Wrong Package**: Code was configured to use `@context7/server` but the actual package is `@upstash/context7-mcp`
2. **Wrong Tool Names**: Code expected `context7_resolve` and `context7_get_docs` but actual API provides `resolve-library-id` and `get-library-docs`
3. **Wrong Parameters**: Parameter names didn't match the actual Context7 API specification

## Root Cause Analysis
- **Communication Error**: `[Errno 22] Invalid argument` and JSON parsing failures
- **Server Mismatch**: Trying to use non-existent `@context7/server` package 
- **API Mismatch**: Tool names and parameters didn't match the real Context7 API

## Fixes Implemented

### 1. Package Configuration Fix
**File**: `src/server_research_mcp/tools/mcp_manager.py`
```python
# Before:
"context7": MCPServerConfig(
    name="context7",
    command="npx",
    args=["-y", "@context7/server"],  # ❌ Wrong package
    env={}
),

# After:  
"context7": MCPServerConfig(
    name="context7", 
    command="npx",
    args=["-y", "@upstash/context7-mcp"],  # ✅ Correct package
    env={}
),
```

### 2. Parameter Validation Fix
**File**: `src/server_research_mcp/tools/enhanced_mcp_manager.py`

Added `_validate_context7_parameters()` method:
```python
def _validate_context7_parameters(self, tool_name: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    if tool_name == "resolve-library-id":
        # Convert library_name to libraryName for the actual Context7 API
        library_name = kwargs.get("library_name", kwargs.get("libraryName", ""))
        kwargs["libraryName"] = str(library_name).strip()
        
        # Remove old parameter names
        kwargs.pop("library_name", None)
        kwargs.pop("query", None)
        
    elif tool_name == "get-library-docs":
        # Ensure context7CompatibleLibraryID is a valid string
        library_id = kwargs.get("context7_library_id", kwargs.get("context7CompatibleLibraryID", ""))
        kwargs["context7CompatibleLibraryID"] = str(library_id)
        
        # Remove old parameter names
        kwargs.pop("context7_library_id", None)
        
        # Validate tokens (minimum 10000 according to Context7 docs)
        tokens = kwargs.get("tokens", 10000)
        kwargs["tokens"] = max(int(tokens), 10000)
```

### 3. Enhanced Error Handling
Added Context7-specific error handling with detailed diagnostics:
```python
# Check if we're using incorrect tool names
if tool_name in ["context7_resolve", "context7_get_docs"]:
    correct_tool = "resolve-library-id" if tool_name == "context7_resolve" else "get-library-docs"
    logger.warning(f"Using legacy tool name '{tool_name}', actual Context7 API uses '{correct_tool}'")

# Handle specific error types
if "invalid json" in error_message.lower() or "parse" in error_message.lower():
    raise ValueError(
        f"Context7 server communication error. Server may be sending non-JSON output. "
        f"Suggested fix: Check if you're using the correct Context7 package (@upstash/context7-mcp) and tool names."
    )
```

### 4. Enhanced Logging  
Added detailed logging for Context7 server initialization:
```python
logger.info(f"Context7 available tools: {', '.join(tool_names)}")
logger.info(f"Initializing context7 server...")
logger.info(f"Context7 available tools: {available_tools}")
```

### 5. Updated Mock Responses
**File**: `src/server_research_mcp/tools/mcp_manager.py`

Updated mock responses to match the actual Context7 API:
```python
if tool == "resolve-library-id":
    return {
        "libraryId": f"/mock/{arguments.get('libraryName', 'unknown')}/docs",
        "found": True,
        "name": arguments.get('libraryName', 'unknown')
    }
elif tool == "get-library-docs":
    return {
        "content": f"Mock documentation for {arguments.get('context7CompatibleLibraryID', 'unknown')}",
        "tokens": max(arguments.get('tokens', 10000), 10000),
        "topic": arguments.get('topic', ''),
        "libraryId": arguments.get('context7CompatibleLibraryID', 'unknown')
    }
```

## Context7 API Reference (from Documentation)

### Available Tools:
1. **`resolve-library-id`**
   - Parameter: `libraryName` (required) - The name of the library to search for
   - Returns: Library ID that can be used with get-library-docs

2. **`get-library-docs`** 
   - Parameters:
     - `context7CompatibleLibraryID` (required) - Exact Context7-compatible library ID
     - `topic` (optional) - Focus the docs on a specific topic  
     - `tokens` (optional, default 10000) - Max tokens to return (minimum 10000)

## Test Results
✅ **All tests passing** - Verified with `test_context7_fix.py`:
- Mock API responses work correctly with new tool names
- Parameter validation correctly maps old → new parameter names  
- Enhanced logging shows proper Context7 server initialization
- Real server connection attempts work (when Node.js/npx available)

## Impact
- **Communication**: Fixed JSON parsing and communication errors
- **Compatibility**: Updated to use correct Context7 package and API
- **Debugging**: Enhanced logging makes Context7 issues easier to diagnose
- **Backward Compatibility**: Legacy tool names still supported in mock mode

## Next Steps for Users
1. **Update Tool Calls**: Use `resolve-library-id` and `get-library-docs` instead of legacy names
2. **Update Parameters**: Use `libraryName` and `context7CompatibleLibraryID` parameters 
3. **Verify Package**: Ensure using `@upstash/context7-mcp` package in configurations
4. **Check Node.js**: Ensure Node.js and npx are available for real Context7 server usage

The Context7 MCP server should now work correctly with the proper @upstash/context7-mcp package! 