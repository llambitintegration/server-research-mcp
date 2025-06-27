# Server Research MCP - Package Structure

This package implements a CrewAI-based research system with Model Context Protocol (MCP) integration.

## Directory Structure

```
server_research_mcp/
├── __init__.py          # Main package exports
├── crew.py              # CrewAI crew definition
├── main.py              # Entry points (run, train, replay, test)
├── agents/              # Agent definitions (future expansion)
├── config/              # Configuration modules
│   ├── __init__.py
│   ├── llm_config.py    # Centralized LLM configuration
│   ├── agents.yaml      # Agent configurations
│   └── tasks.yaml       # Task configurations
├── models/              # Data models (future expansion)
├── schemas/             # Pydantic schemas (future expansion)
├── tools/               # Tool implementations
│   ├── __init__.py
│   └── mcp_tools.py     # MCP tool definitions
└── utils/               # Utility functions
    ├── __init__.py
    └── validators.py     # Input/output validators
```

## Key Components

### Configuration (`config/`)
- **llm_config.py**: Centralized LLM configuration supporting both Anthropic and OpenAI
- **agents.yaml**: YAML-based agent configurations
- **tasks.yaml**: YAML-based task configurations

### Tools (`tools/`)
- **mcp_tools.py**: MCP tool implementations for:
  - Memory search and management
  - Context7 library resolution and documentation
  - Sequential thinking for complex analysis

### Utilities (`utils/`)
- **validators.py**: Validation functions for:
  - Research topic validation
  - Context gathering output validation

### Core Files
- **crew.py**: Defines the ServerResearchMcp crew with Historian agent
- **main.py**: Entry points for running, training, and testing the crew

## Usage

```python
# Import and run the crew
from server_research_mcp import run
run()

# Or use the crew directly
from server_research_mcp import ServerResearchMcp
crew = ServerResearchMcp()
result = crew.crew().kickoff(inputs={"topic": "AI research"})
```

## Environment Variables

Required environment variables:
- `LLM_PROVIDER`: Either 'anthropic' or 'openai'
- `ANTHROPIC_API_KEY`: Required if using Anthropic
- `OPENAI_API_KEY`: Required if using OpenAI
- `LLM_MODEL`: Optional, defaults to provider-specific model