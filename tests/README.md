# Test Suite for server-research-mcp

This document describes the test suite structure, conventions, and execution patterns for the server-research-mcp project.

## Test Organization

The test suite is organized into three main categories:

### Unit Tests (`tests/unit/`)
- **Purpose**: Test individual components in isolation
- **Scope**: Single functions, classes, or modules
- **Dependencies**: Minimal external dependencies, extensive mocking
- **Execution**: Fast, reliable, no external services required

### Integration Tests (`tests/integration/`)
- **Purpose**: Test component interactions and workflows
- **Scope**: Multiple components working together
- **Dependencies**: Some external services (mocked by default)
- **Execution**: Moderate speed, may require specific configuration

### End-to-End Tests (`tests/e2e/`)
- **Purpose**: Test complete workflows from start to finish
- **Scope**: Full system behavior
- **Dependencies**: External services, real data flows
- **Execution**: Slower, requires full environment setup

## Test Markers

The test suite uses pytest markers to categorize tests by their requirements and characteristics:

### Core Markers
- `@pytest.mark.unit`: Unit tests for individual components
- `@pytest.mark.integration`: Integration tests for component interactions  
- `@pytest.mark.e2e`: End-to-end workflow tests
- `@pytest.mark.slow`: Tests that take significant time to complete
- `@pytest.mark.performance`: Performance benchmarking tests

### Dependency Markers
- `@pytest.mark.requires_llm`: Tests requiring LLM API access
- `@pytest.mark.requires_mcp`: Tests requiring MCP server connections
- `@pytest.mark.real_servers`: Tests requiring real MCP server connections
- `@pytest.mark.real_integration`: Tests requiring real external services

### MCP Server Testing Strategy

The test suite implements a sophisticated strategy for testing MCP (Model Context Protocol) tools:

#### Publisher Tools Dynamic/Fallback Logic

The `get_publisher_tools()` function implements a dynamic loading strategy:

1. **Primary Mode**: Attempts to connect to real MCP servers (e.g., `obsidian-mcp-tools`)
2. **Fallback Mode**: Uses Pydantic-compatible dummy implementations when servers are unavailable

**Publisher Tools Available**:
- `obsidian_create_note`: Create notes in Obsidian vault
- `obsidian_link_generator`: Generate links between notes
- `filesystem_write`: Write files to filesystem
- `obsidian_publish_note`: Publish notes to external platforms
- `obsidian_update_metadata`: Update note metadata

**Implementation Details**:
- All tools inherit from `BaseTool` with proper Pydantic schemas
- Fallback tools return structured JSON responses for testing
- Real tools connect to MCP servers when available
- Graceful degradation ensures tests pass in CI environments

#### Test Execution Modes

**CI/Basic Testing Mode** (Default):
```bash
# Runs with fallback tools, no real servers required
python -m pytest tests/unit/test_parsers.py::TestMCPTools
```

**Real Server Testing Mode**:
```bash
# Runs tests against real MCP servers (requires setup)
python -m pytest -m "real_servers" tests/
```

**Skip Real Server Tests**:
```bash
# Explicitly skip tests requiring real servers
python -m pytest -m "not real_servers" tests/
```

#### MCP Server Configuration

Real MCP servers require proper configuration:

**obsidian-mcp-tools** (Publisher Tools):
- Requires Node.js and npx
- Configured via MCP client configuration
- Provides tools for Obsidian vault interaction

**Other MCP Servers**:
- `memory`: Knowledge graph operations
- `context7`: Documentation retrieval
- `zotero`: Research paper management
- `sequential-thinking`: Multi-step reasoning

#### Tool Collection Architecture

The tool system uses a plug-and-play architecture:

```python
# Get tools by agent role
historian_tools = get_historian_tools()      # Memory + Context7 tools
researcher_tools = get_researcher_tools()    # Zotero + Thinking tools  
archivist_tools = get_archivist_tools()      # Schema + Summary + FileSystem tools
publisher_tools = get_publisher_tools()     # Obsidian + FileSystem tools

# Get all tools organized by category
all_tools = get_all_mcp_tools()
```

Each tool collection is designed for specific agent roles and use cases, ensuring proper separation of concerns and testability.

## Test Execution

### Running All Tests
```bash
python -m pytest tests/
```

### Running by Category
```bash
# Unit tests only
python -m pytest tests/unit/

# Integration tests only  
python -m pytest tests/integration/

# End-to-end tests only
python -m pytest tests/e2e/
```

### Running by Marker
```bash
# Fast tests only
python -m pytest -m "not slow"

# Tests requiring external services
python -m pytest -m "real_integration"

# Performance tests
python -m pytest -m "performance"
```

### Running Specific Test Patterns
```bash
# All MCP tool tests
python -m pytest -k "test_*_tools"

# Publisher-related tests
python -m pytest -k "publisher"

# Memory/historian tests
python -m pytest -k "historian or memory"
```

## Configuration

### Environment Variables
- `TESTING=true`: Enables test mode
- `CHROMADB_PATH`: Temporary directory for test ChromaDB
- `CHROMADB_ALLOW_RESET=true`: Allows ChromaDB reset in tests

### Test Fixtures
The test suite provides comprehensive fixtures for:
- Mock LLM instances
- Mock crew configurations  
- Mock MCP managers with tool-specific responses
- Temporary workspaces and file systems
- ChromaDB test isolation

### Coverage and Quality

The test suite emphasizes:
- **High Coverage**: Comprehensive test coverage across all components
- **Isolation**: Tests run independently without side effects
- **Reliability**: Consistent results across different environments
- **Performance**: Fast execution for rapid development feedback

For detailed information about specific test patterns or adding new tests, see the individual test files and their docstrings.