# Rate Limiting Configuration

This document describes the rate limiting features added to prevent CrewAI failures from API rate limits.

## Overview

The rate limiting system provides:
- Request throttling per minute/hour
- Exponential backoff with jitter for retries
- Configurable rate limits via environment variables
- Tool-specific rate limit configurations
- Automatic retry on rate limit errors

## Architecture

### Components

1. **RateLimiter**: Core rate limiting logic with request tracking
2. **RateLimitedTool**: Wrapper for CrewAI tools that adds rate limiting
3. **BackoffStrategies**: Different retry delay strategies (exponential, linear, constant)
4. **RateLimitConfig**: Configuration dataclass for rate limit parameters

### Integration Points

- **mcp_tools.py**: Tools are wrapped with rate limiting before being assigned to agents
- **tool_wrapper.py**: Provides the wrapping logic and error detection
- **rate_limiter.py**: Manages request counts and timing

## Configuration

### Environment Variables

Rate limits can be configured per tool type using environment variables:

#### Global Settings
```bash
# Default rate limits for all tools
MCP_MAX_REQUESTS_PER_MINUTE=30
MCP_MAX_REQUESTS_PER_HOUR=300
MCP_MIN_REQUEST_INTERVAL=0.5
MCP_MAX_RETRIES=5
MCP_INITIAL_RETRY_DELAY=1.0
MCP_MAX_RETRY_DELAY=60.0
MCP_BACKOFF_FACTOR=2.0
```

#### Tool-Specific Settings
```bash
# Zotero API limits
MCP_ZOTERO_MAX_REQUESTS_PER_MINUTE=10
MCP_ZOTERO_MAX_REQUESTS_PER_HOUR=100
MCP_ZOTERO_MIN_REQUEST_INTERVAL=1.0
MCP_ZOTERO_MAX_RETRIES=5
MCP_ZOTERO_INITIAL_RETRY_DELAY=2.0

# Memory tool limits
MCP_MEMORY_MAX_REQUESTS_PER_MINUTE=60
MCP_MEMORY_MAX_REQUESTS_PER_HOUR=600
MCP_MEMORY_MIN_REQUEST_INTERVAL=0.1

# Filesystem tool limits
MCP_FILESYSTEM_MAX_REQUESTS_PER_MINUTE=100
MCP_FILESYSTEM_MAX_REQUESTS_PER_HOUR=1000
MCP_FILESYSTEM_MIN_REQUEST_INTERVAL=0.05
```

### Default Configurations

If not specified via environment variables, these defaults apply:

| Tool Type | Requests/Min | Requests/Hour | Min Interval | Max Retries |
|-----------|--------------|---------------|--------------|-------------|
| Zotero    | 10           | 100           | 1.0s         | 5           |
| Memory    | 60           | 600           | 0.1s         | 3           |
| Filesystem| 100          | 1000          | 0.05s        | 3           |
| Default   | 30           | 300           | 0.5s         | 4           |

## Error Detection

The system automatically detects rate limit errors by looking for these keywords:
- "rate limit"
- "rate_limit"
- "too many requests"
- "429"
- "quota exceeded"
- "throttled"
- "try again later"

## Retry Logic

When a rate limit is hit:

1. **Initial Detection**: Error is identified as rate limit error
2. **Backoff Calculation**: Delay = min(initial_delay * (backoff_factor ^ attempt), max_delay)
3. **Jitter Addition**: Random jitter added to prevent thundering herd
4. **Retry**: Request retried after calculated delay
5. **Max Retries**: Fails after max_retries attempts

## Usage Example

```python
from server_research_mcp.utils.rate_limiting import RateLimitConfig, wrap_tools_with_rate_limit

# Create custom rate limit config
config = RateLimitConfig(
    max_requests_per_minute=20,
    max_requests_per_hour=200,
    min_request_interval=1.0,
    max_retries=5,
    initial_retry_delay=2.0,
    backoff_factor=2.0
)

# Wrap tools with rate limiting
rate_limited_tools = wrap_tools_with_rate_limit(tools, config=config)
```

## Monitoring

Rate limiting events are logged at various levels:
- **DEBUG**: Request recorded, current counts
- **INFO**: Rate limit reached, waiting
- **WARNING**: Errors and retries
- **ERROR**: Max retries exceeded

## Best Practices

1. **Set Conservative Limits**: Better to be under API limits than hit them
2. **Monitor Logs**: Watch for rate limit warnings to tune settings
3. **Use Tool-Specific Configs**: Different APIs have different limits
4. **Test Under Load**: Verify rate limiting works before production
5. **Handle Failures Gracefully**: Always have fallback for when retries fail

## Troubleshooting

### Common Issues

1. **Still hitting rate limits**: Decrease MAX_REQUESTS_PER_MINUTE or increase MIN_REQUEST_INTERVAL
2. **Too slow**: Increase limits if well under API thresholds
3. **Retries failing**: Increase MAX_RETRIES or INITIAL_RETRY_DELAY
4. **Thundering herd**: Ensure jitter is enabled (default)

### Debug Mode

Enable detailed logging:
```python
import logging
logging.getLogger("server_research_mcp.utils.rate_limiting").setLevel(logging.DEBUG)
```
