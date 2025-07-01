# MCP Server Setup Guide

This guide explains how to set up and run the MCP (Model Context Protocol) servers for the Research Paper Parser system.

## Overview

The Research Paper Parser uses six MCP servers to provide advanced functionality:

1. **Memory Server** - Knowledge graph operations
2. **Sequential Thinking** - Multi-step reasoning
3. **Context7** - Library documentation
4. **Filesystem** - File operations
5. **Zotero** - Paper discovery and extraction
6. **Obsidian** - Note creation and management

## Prerequisites

- Node.js 18+ (for npm/npx)
- Python 3.10+
- Zotero account with API access
- Obsidian vault

## Installation

### 1. Install Node.js Dependencies

The following MCP servers require Node.js:

```bash
# Install globally (recommended for development)
npm install -g @modelcontextprotocol/server-filesystem
npm install -g @modelcontextprotocol/server-memory
npm install -g @modelcontextprotocol/server-sequential-thinking
npm install -g @upstash/context7-mcp
npm install -g obsidian-mcp
```

### 2. Install Python Dependencies

```bash
# Install the project
pip install -e .

# Install Zotero MCP server
pip install zotero-mcp
```

### 3. Configure Environment Variables

Create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
# LLM Configuration
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your_key_here

# Zotero Configuration
ZOTERO_API_KEY=your_zotero_api_key
ZOTERO_LIBRARY_ID=your_library_id
ZOTERO_LIBRARY_TYPE=user

# Obsidian Configuration
OBSIDIAN_VAULT_PATH=/path/to/your/vault

# MCP Configuration
MCP_TEST_MODE=false  # Set to true for testing
```

## Running MCP Servers

### Option 1: Manual Server Startup (Development)

Start each server individually in separate terminals:

```bash
# Terminal 1: Memory Server
npx @modelcontextprotocol/server-memory

# Terminal 2: Sequential Thinking
npx @modelcontextprotocol/server-sequential-thinking

# Terminal 3: Context7
npx @upstash/context7-mcp

# Terminal 4: Filesystem (adjust path for your system)
npx @modelcontextprotocol/server-filesystem C:\0_repos\mcp

# Terminal 5: Obsidian (adjust path for your vault)
npx obsidian-mcp C:\0_repos\mcp\Obsidian

# Terminal 6: Zotero
uvx zotero-mcp
```

### Option 2: Using MCP Configuration (Recommended)

Create an MCP configuration file `mcp_config.json`:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "C:\\0_repos\\mcp"]
    },
    "memory": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"],
      "env": {
        "DEBUG": "*"
      }
    },
    "sequential-thinking": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"]
    },
    "context7": {
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp"]
    },
    "obsidian-mcp-tools": {
      "command": "npx",
      "args": ["-y", "obsidian-mcp", "C:\\0_repos\\mcp\\Obsidian"]
    },
    "zotero": {
      "command": "uvx",
      "args": ["zotero-mcp"],
      "env": {
        "ZOTERO_LOCAL": "false",
        "ZOTERO_API_KEY": "",
        "ZOTERO_LIBRARY_ID": ""
      }
    }
  }
}
```

### Option 3: Test Mode

For testing without real MCP servers:

```bash
export MCP_TEST_MODE=true
python main.py "test paper"
```

## Testing the Setup

### 1. Run Unit Tests

```bash
# Run all tests
pytest

# Run MCP integration tests
pytest tests/test_mcp_integration.py -v

# Run with coverage
pytest --cov=server_research_mcp
```

### 2. Test Individual Tools

```python
# Test memory search
from server_research_mcp.tools.mcp_tools import MemorySearchTool
tool = MemorySearchTool()
result = tool._run(query="transformers")
print(result)
```

### 3. Test Full Pipeline

```bash
# Dry run (no files created)
python main.py "Attention is All You Need" --dry-run

# Real run
python main.py "Attention is All You Need" --topic "AI"
```

## Troubleshooting

### Common Issues

1. **MCP Server Connection Failed**
   ```
   Error: Failed to start memory: Connection refused
   ```
   - Solution: Ensure the MCP server is running
   - Check ports are not blocked
   - Verify server installation

2. **Zotero Authentication Error**
   ```
   Error: Zotero API key invalid
   ```
   - Solution: Check ZOTERO_API_KEY in .env
   - Verify library ID and type
   - Test API key at https://api.zotero.org

3. **Obsidian Path Not Found**
   ```
   Error: Obsidian vault path does not exist
   ```
   - Solution: Use absolute path in .env
   - Create the Papers folder in vault
   - Check permissions

4. **Node.js Command Not Found**
   ```
   Error: npx command not found
   ```
   - Solution: Install Node.js 18+
   - Add npm to PATH
   - Restart terminal

### Debug Mode

Enable debug logging:

```bash
export DEBUG=*
export MCP_DEBUG=true
python main.py "test paper" --verbose
```

### Server Health Check

Test each server individually:

```python
import asyncio
from server_research_mcp.tools.mcp_manager import get_mcp_manager

async def test_servers():
    manager = get_mcp_manager()
    
    # Test each server
    servers = ["memory", "zotero", "sequential-thinking", "context7", "filesystem", "obsidian-mcp-tools"]
    
    for server in servers:
        try:
            await manager.initialize([server])
            print(f"✅ {server} - Connected")
        except Exception as e:
            print(f"❌ {server} - Failed: {e}")
    
    await manager.shutdown()

asyncio.run(test_servers())
```

## Advanced Configuration

### Custom Server Paths

Modify paths in the MCP manager:

```python
from server_research_mcp.tools.mcp_manager import MCPManager, MCPServerConfig

# Update filesystem path
MCPManager.SERVER_CONFIGS["filesystem"].args[-1] = "/your/custom/path"

# Update Obsidian vault
MCPManager.SERVER_CONFIGS["obsidian-mcp-tools"].args[-1] = "/your/vault/path"
```

### Server Timeout Settings

Configure timeouts in environment:

```bash
export MCP_CONNECT_TIMEOUT=30
export MCP_CALL_TIMEOUT=60
```

### Parallel Server Initialization

Initialize all servers at once:

```python
manager = get_mcp_manager()
await manager.initialize()  # Initializes all servers
```

## Production Deployment

### Using Docker

```dockerfile
FROM node:18-python3.10

# Install Node dependencies
RUN npm install -g \
    @modelcontextprotocol/server-filesystem \
    @modelcontextprotocol/server-memory \
    @modelcontextprotocol/server-sequential-thinking \
    @upstash/context7-mcp \
    obsidian-mcp

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY . /app
WORKDIR /app

# Run servers
CMD ["python", "main.py"]
```

### Using Process Manager

Create `ecosystem.config.js` for PM2:

```javascript
module.exports = {
  apps: [
    {
      name: 'mcp-memory',
      script: 'npx',
      args: '@modelcontextprotocol/server-memory',
      env: { DEBUG: '*' }
    },
    {
      name: 'mcp-zotero',
      script: 'uvx',
      args: 'zotero-mcp',
      env: {
        ZOTERO_API_KEY: process.env.ZOTERO_API_KEY,
        ZOTERO_LIBRARY_ID: process.env.ZOTERO_LIBRARY_ID
      }
    }
    // Add other servers...
  ]
};
```

Run with PM2:

```bash
pm2 start ecosystem.config.js
pm2 status
pm2 logs
```

## Schema Propagation

### Overview

The MCP tool wrapper system implements comprehensive schema propagation to ensure proper parameter validation and tool interface consistency. This system handles the complex task of maintaining Pydantic schema information as tools are wrapped and adapted between different execution contexts.

### Implementation Details

#### Schema Sources

Tools can obtain their schema from multiple sources, in order of preference:

1. **Original Tool Schema** - If the original MCP tool has an `args_schema` attribute, it is preserved
2. **Signature Analysis** - Schema is built dynamically from the `_run` method signature using inspection
3. **Fallback Schema** - A basic schema with common parameters is created as a last resort

#### Wrapper Preservation

The `MCPToolWrapper` class ensures schema propagation through:

```python
class MCPToolWrapper(BaseTool):
    args_schema: Type[BaseModel]  # Explicit class attribute exposure
    
    def __init__(self, original_tool: BaseTool):
        # Step 1a: Read original_tool.args_schema if exists
        if hasattr(original_tool, 'args_schema') and original_tool.args_schema is not None:
            args_schema = original_tool.args_schema
        else:
            # Step 1b: Build from _run signature
            args_schema = self._build_schema_from_run_signature(original_tool)
        
        super().__init__(..., args_schema=args_schema)
```

#### Async Tool Patching

When async tools are patched with `_patch_tool`, schema preservation is guaranteed:

```python
def _patch_tool(tool: BaseTool) -> BaseTool:
    # Preserve args_schema before patching
    original_args_schema = getattr(tool, 'args_schema', None)
    
    # Perform async-to-sync wrapping...
    
    # Restore args_schema after patching
    if original_args_schema is not None:
        tool.args_schema = original_args_schema
    
    return tool
```

### Runtime Signature Enforcement

#### Kwargs-Only Interface

All wrapped tools enforce a kwargs-only interface for consistency:

```python
def _run(self, **kwargs) -> str:  # Step 2a: kwargs-only signature
    # Convert kwargs to positional only for tools that require it
    if self._original_tool_needs_positional_args():
        args, remaining_kwargs = self._convert_kwargs_to_positional(kwargs)
        return self.original_tool._run(*args, **remaining_kwargs)
    else:
        return self.original_tool._run(**kwargs)
```

#### Parameter Conversion

Tools that originally required positional arguments are automatically detected and handled:

- **Signature Analysis** - Inspect original `_run` method to determine positional requirements
- **Smart Conversion** - Convert kwargs to positional args based on parameter order
- **Backward Compatibility** - Maintain support for existing tool calls

### Name Filtering Enhancement

Tool name matching uses normalized comparison for robust filtering:

```python
def normalize_tool_name(name: str) -> str:
    """Normalize tool name by removing all non-alphanumeric characters."""
    return re.sub(r'[^a-z0-9]', '', name.lower())

def filter_tools_by_keywords(tools, keywords):
    normalized_keywords = [normalize_tool_name(k) for k in keywords]
    # Match using normalized names...
```

### Quality Guarantees

#### Zotero Tool Availability

The system guarantees Zotero tool availability when credentials are provided:

```python
# In _initialise method
zotero_credentials_exist = bool(os.getenv("ZOTERO_API_KEY")) and bool(os.getenv("ZOTERO_LIBRARY_ID"))
zotero_tools = [tool for tool in wrapped_tools if 'zotero' in normalize_tool_name(tool.name)]

if zotero_credentials_exist and not zotero_tools:
    raise Exception("Zotero credentials provided but no Zotero tools loaded - aborting fallback")
```

#### Schema Validation

Every wrapped tool is guaranteed to have a valid schema:

- Non-None `args_schema` attribute
- Valid Pydantic BaseModel subclass
- Proper field definitions with types and defaults

### Testing Requirements

Comprehensive tests verify schema propagation:

```python
def test_args_schema_propagation():
    """Verify every wrapped tool has non-None args_schema."""
    tools = _AdaptHolder.get_all_tools()
    for tool in tools:
        assert hasattr(tool, 'args_schema')
        assert tool.args_schema is not None
        assert issubclass(tool.args_schema, BaseModel)

def test_parameter_round_trip():
    """Verify JSON string and kwargs produce identical calls."""
    # Test both parameter formats produce consistent results
```

### Best Practices

1. **Always use kwargs** when calling wrapped tools directly
2. **Check args_schema** before calling tools to understand required parameters
3. **Handle both success and error responses** consistently
4. **Use normalized names** when filtering tools by keywords
5. **Test with both parameter formats** to ensure compatibility

## API Reference

### MCP Manager

```python
from server_research_mcp.tools.mcp_manager import get_mcp_manager

# Get manager instance
manager = get_mcp_manager()

# Initialize specific servers
await manager.initialize(["memory", "zotero"])

# Call a tool
result = await manager.call_tool(
    server="memory",
    tool="search_nodes",
    arguments={"query": "transformers"}
)

# Shutdown
await manager.shutdown()
```

### Tool Usage

Each tool follows the CrewAI tool interface:

```python
from server_research_mcp.tools.mcp_tools import MemorySearchTool

# Create tool instance
tool = MemorySearchTool()

# Run tool (synchronous interface for CrewAI)
result = tool._run(query="machine learning")

# Parse result
import json
data = json.loads(result)
```

## Contributing

To add a new MCP server:

1. Add configuration to `MCPManager.SERVER_CONFIGS`
2. Update `MockMCPClient` for testing
3. Create tool wrappers in `mcp_tools.py`
4. Add tests in `test_mcp_integration.py`
5. Update documentation

## Support

- GitHub Issues: [Report bugs](https://github.com/your-repo/issues)
- Documentation: [Full docs](https://your-docs-site.com)
- Discord: [Community support](https://discord.gg/your-channel)