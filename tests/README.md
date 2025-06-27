# Server Research MCP - Test Suite

This directory contains the consolidated test suite for the server-research-mcp project.

## Test Structure

The tests are organized into three main categories:

```
tests/
├── conftest.py          # Shared fixtures and test configuration
├── pytest.ini           # Pytest configuration
├── unit/               # Unit tests for individual components
│   ├── test_agents.py   # Agent and crew functionality tests
│   ├── test_tools.py    # MCP tools tests
│   └── test_validation.py # Validation and user input tests
├── integration/        # Integration tests for component interactions
│   ├── test_mcp_servers.py # MCP server integration tests
│   ├── test_crew_workflow.py # Crew workflow integration tests
│   └── test_llm.py      # LLM connection and configuration tests
└── e2e/               # End-to-end workflow tests
    └── test_research_flow.py # Complete research workflow tests
```

## Running Tests

### Run All Tests
```bash
pytest
```

### Run Specific Test Categories
```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# End-to-end tests only
pytest tests/e2e/
```

### Run Tests by Marker
```bash
# Skip slow tests
pytest -m "not slow"

# Run only tests that don't require external services
pytest -m "not requires_llm and not requires_mcp"

# Run only unit tests
pytest -m unit
```

### Run Specific Test Files
```bash
pytest tests/unit/test_agents.py
pytest tests/integration/test_mcp_servers.py -v
```

## Test Markers

- `@pytest.mark.slow` - Long-running tests
- `@pytest.mark.requires_llm` - Tests requiring LLM API access
- `@pytest.mark.requires_mcp` - Tests requiring MCP server connections
- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.e2e` - End-to-end tests

## Environment Setup

### Required Environment Variables

For full test suite execution, set these environment variables:

```bash
# LLM Configuration (choose one provider)
export LLM_PROVIDER=anthropic  # or 'openai'
export ANTHROPIC_API_KEY=your_key_here
# OR
export OPENAI_API_KEY=your_key_here

# LLM Model (optional, defaults are provided)
export LLM_MODEL=claude-3-haiku-20240307  # for Anthropic
# OR
export LLM_MODEL=gpt-4o-mini  # for OpenAI
```

### Test-Only Environment

The test suite automatically sets up a test environment with:
- `TESTING=true`
- Temporary ChromaDB path
- ChromaDB reset enabled

## Common Test Patterns

### Mocking LLM Calls
```python
def test_with_mock_llm(mock_llm):
    """mock_llm fixture provides a mocked LLM instance."""
    mock_llm.call.return_value = "Mock response"
    # Your test code here
```

### Mocking MCP Servers
```python
def test_with_mock_mcp(mock_mcp_manager):
    """mock_mcp_manager fixture provides a mocked MCP manager."""
    # Your test code here
```

### Testing Validation
```python
def test_validation(valid_research_output, valid_report_output):
    """Use pre-defined valid outputs for testing."""
    # Your test code here
```

## Continuous Integration

For CI environments where LLM APIs and MCP servers are not available:

```bash
# Run only tests that don't require external services
pytest -m "not requires_llm and not requires_mcp"

# Run with coverage reporting
pytest --cov=src --cov-report=xml -m "not requires_llm and not requires_mcp"
```

## Debugging Tests

### Verbose Output
```bash
pytest -vv tests/unit/test_agents.py::TestAgents::test_researcher_agent
```

### Show Print Statements
```bash
pytest -s tests/integration/test_mcp_servers.py
```

### Debug Failed Tests
```bash
pytest --pdb --lf  # Drop into debugger on failure, run last failed
```

## Test Coverage

To run tests with coverage reporting:

```bash
# Install coverage plugin
pip install pytest-cov

# Run with coverage
pytest --cov=src --cov-report=html --cov-report=term-missing

# View HTML report
open htmlcov/index.html
```

## Writing New Tests

1. **Choose the appropriate directory:**
   - `unit/` for testing individual functions/classes
   - `integration/` for testing component interactions
   - `e2e/` for testing complete workflows

2. **Use appropriate fixtures from conftest.py:**
   - `sample_inputs` - Standard crew inputs
   - `mock_llm` - Mocked LLM instance
   - `mock_mcp_manager` - Mocked MCP manager
   - `valid_research_output` - Valid research output example
   - `valid_report_output` - Valid report output example

3. **Mark tests appropriately:**
   ```python
   @pytest.mark.slow
   @pytest.mark.requires_llm
   def test_real_llm_call():
       # Test code
   ```

4. **Follow naming conventions:**
   - Test files: `test_*.py`
   - Test classes: `Test*`
   - Test functions: `test_*`

## Troubleshooting

### ChromaDB Issues
If you see ChromaDB-related errors, ensure the test environment is properly set up:
```python
# This is handled automatically by conftest.py
os.environ['CHROMADB_ALLOW_RESET'] = 'true'
```

### MCP Server Tests Failing
MCP tests require Node.js and npx. To skip these tests:
```bash
pytest -m "not requires_mcp"
```

### LLM Tests Failing
Ensure you have valid API keys set in your environment or skip:
```bash
pytest -m "not requires_llm"
```