# Comprehensive Test Suite for Server Research MCP

This test suite provides thorough coverage for a fully functioning CrewAI research system with MCP integration.

## Test Structure

### Core Test Modules

1. **test_crew_core.py** - Core crew functionality
   - Crew initialization and configuration
   - Task execution and kickoff
   - Memory management
   - Output validation
   - Error handling and recovery

2. **test_agents.py** - Agent-specific tests
   - Historian agent with MCP tools
   - Researcher agent (future)
   - Synthesizer agent (future)
   - Validator agent (future)
   - Agent collaboration patterns

3. **test_mcp_integration.py** - MCP integration tests
   - Server lifecycle management
   - Tool execution and error handling
   - Async operations
   - Integration patterns
   - Performance optimization

4. **test_end_to_end.py** - End-to-end workflows
   - Complete research workflows
   - Quality assurance
   - Workflow orchestration
   - Data persistence
   - Error recovery

5. **test_knowledge_management.py** - Knowledge and RAG tests
   - Knowledge graph operations
   - RAG integration
   - Knowledge organization
   - Quality control

6. **test_performance.py** - Performance and scalability
   - Performance baselines
   - Scalability limits
   - Resource optimization
   - Load testing

### Support Files

- **conftest.py** - Comprehensive fixtures and test configuration
- **pytest.ini** - Pytest configuration
- **test_run_all.py** - Test runner with various options

## Running Tests

### Quick Start

```bash
# Run quick tests (excludes slow and integration tests)
python tests/test_run_all.py --type quick

# Run all tests
python tests/test_run_all.py --type all

# Run specific test category
python tests/test_run_all.py --type core
python tests/test_run_all.py --type agents
python tests/test_run_all.py --type mcp
```

### Using Pytest Directly

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=server_research_mcp --cov-report=html

# Run specific markers
pytest -m "not slow"
pytest -m integration
pytest -m performance

# Run specific test file
pytest tests/test_crew_core.py

# Run with verbose output
pytest -v

# Run until first failure
pytest -x
```

### Test Categories by Marker

- `@pytest.mark.integration` - Tests requiring external services
- `@pytest.mark.slow` - Long-running tests (>1 second)
- `@pytest.mark.mcp` - Tests requiring MCP servers
- `@pytest.mark.llm` - Tests requiring LLM API
- `@pytest.mark.async` - Asynchronous tests
- `@pytest.mark.performance` - Performance benchmarks
- `@pytest.mark.e2e` - End-to-end workflow tests

## Test Environment Setup

### Required Environment Variables

```bash
# LLM Configuration
export LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=your-key
# or
export LLM_PROVIDER=openai
export OPENAI_API_KEY=your-key

# Test Environment
export CHROMADB_PATH=/tmp/test_chromadb
export CHROMADB_ALLOW_RESET=true
```

### Installing Test Dependencies

```bash
# Basic test dependencies
pip install pytest pytest-asyncio pytest-mock

# Optional but recommended
pip install pytest-cov pytest-html pytest-json-report psutil numpy

# For MCP tests (requires Node.js)
npm install -g @modelcontextprotocol/server-memory
npm install -g @upstash/context7-mcp
npm install -g @modelcontextprotocol/server-sequential-thinking
```

## Key Testing Patterns

### 1. Mocking MCP Servers

The test suite includes comprehensive mocks for MCP servers to enable testing without external dependencies:

```python
def test_with_mcp_mock(mock_mcp_manager):
    result = mock_mcp_manager.call_tool("memory_search", query="test")
    assert "results" in result
```

### 2. Crew Testing with Mocks

Complete crew mocking for isolated testing:

```python
def test_crew_workflow(mock_crew, sample_inputs):
    crew = mock_crew.crew()
    result = crew.kickoff(inputs=sample_inputs)
    assert "research_paper" in result
```

### 3. Performance Testing

Built-in performance monitoring:

```python
def test_performance(performance_monitor):
    performance_monitor.start_timer("operation")
    # ... perform operation ...
    elapsed = performance_monitor.stop_timer("operation")
    assert elapsed < 1.0  # Under 1 second
```

### 4. Async Testing

Full async/await support:

```python
@pytest.mark.async
async def test_async_operation(async_mcp_client):
    await async_mcp_client.connect()
    result = await async_mcp_client.call_tool("test_tool")
    assert result["status"] == "success"
```

## Test Coverage Goals

- **Unit Tests**: 80%+ coverage of core functionality
- **Integration Tests**: Key workflows and external integrations
- **Performance Tests**: Baseline metrics and scalability limits
- **End-to-End Tests**: Complete research workflows

## Continuous Integration

The test suite is designed to work with CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run Tests
  run: |
    python tests/test_run_all.py --type unit
    python tests/test_run_all.py --type integration
    
- name: Upload Coverage
  uses: codecov/codecov-action@v3
```

## Troubleshooting

### Common Issues

1. **MCP Server Connection Errors**
   - Ensure Node.js and npx are installed
   - Check MCP server packages are installed globally
   - Use `@pytest.mark.skip` for tests requiring unavailable servers

2. **LLM API Errors**
   - Set appropriate environment variables
   - Use mocks for tests that don't need actual LLM calls
   - Check API key validity and rate limits

3. **Memory/Performance Issues**
   - Run performance tests separately: `pytest -m performance`
   - Adjust test parameters for your system
   - Use garbage collection in long-running tests

## Contributing

When adding new tests:

1. Follow the existing test structure
2. Use appropriate markers
3. Add comprehensive docstrings
4. Include both positive and negative test cases
5. Mock external dependencies
6. Consider performance implications

## Future Enhancements

- [ ] Add more agent types (researcher, synthesizer, validator)
- [ ] Expand integration test coverage
- [ ] Add visual test reporting
- [ ] Implement test data factories
- [ ] Add property-based testing for complex scenarios
- [ ] Create test performance benchmarks dashboard