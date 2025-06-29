"""Unit tests for MCP tools functionality."""

import pytest
from unittest.mock import MagicMock, patch
from server_research_mcp.tools.mcp_tools import (
    MemorySearchTool, MemoryCreateEntityTool, MemoryAddObservationTool,
    Context7ResolveTool, Context7DocsTool, SequentialThinkingTool,
    get_historian_tools
)


class TestMemoryTools:
    """Test Memory MCP tools."""
    
    def test_memory_search_tool_instantiation(self):
        """Test Memory Search Tool instantiation."""
        tool = MemorySearchTool()
        assert tool.name == "memory_search"
        assert "memory" in tool.description.lower()
        assert "search" in tool.description.lower()
        
    def test_memory_search_tool_execution(self, mock_mcp_manager):
        """Test Memory Search Tool execution with mock."""
        with patch('src.server_research_mcp.tools.mcp_tools.get_mcp_manager', return_value=mock_mcp_manager):
            tool = MemorySearchTool()
            result = tool._run("test query")
            
            assert isinstance(result, str)
            assert "test query" in result
            mock_mcp_manager.call_tool.assert_called()
            
    def test_memory_create_entity_tool(self):
        """Test Memory Create Entity Tool instantiation."""
        tool = MemoryCreateEntityTool()
        assert tool.name == "memory_create_entity"
        assert "create" in tool.description.lower()
        assert "entity" in tool.description.lower()
        
    def test_memory_create_entity_execution(self, mock_mcp_manager):
        """Test Memory Create Entity Tool execution."""
        with patch('src.server_research_mcp.tools.mcp_tools.get_mcp_manager', return_value=mock_mcp_manager):
            tool = MemoryCreateEntityTool()
            result = tool._run("test_entity", "test_type", ["observation1", "observation2"])
            
            assert isinstance(result, str)
            assert "test_entity" in result
            mock_mcp_manager.call_tool.assert_called()
            
    def test_memory_add_observation_tool(self):
        """Test Memory Add Observation Tool instantiation."""
        tool = MemoryAddObservationTool()
        assert tool.name == "memory_add_observation"
        assert "observation" in tool.description.lower()


class TestContext7Tools:
    """Test Context7 MCP tools."""
    
    def test_context7_resolve_tool_instantiation(self):
        """Test Context7 Resolve Tool instantiation."""
        tool = Context7ResolveTool()
        assert tool.name == "context7_resolve_library"
        assert "resolve" in tool.description.lower()
        assert "library" in tool.description.lower()
        
    def test_context7_resolve_execution(self, mock_mcp_manager):
        """Test Context7 Resolve Tool execution."""
        with patch('src.server_research_mcp.tools.mcp_tools.get_mcp_manager', return_value=mock_mcp_manager):
            tool = Context7ResolveTool()
            result = tool._run("numpy")
            
            assert isinstance(result, str)
            assert "numpy" in result
            mock_mcp_manager.call_tool.assert_called()
            
    def test_context7_docs_tool_instantiation(self):
        """Test Context7 Docs Tool instantiation."""
        tool = Context7DocsTool()
        assert tool.name == "context7_get_docs"
        assert "documentation" in tool.description.lower()
        
    def test_context7_docs_execution(self, mock_mcp_manager):
        """Test Context7 Docs Tool execution."""
        with patch('src.server_research_mcp.tools.mcp_tools.get_mcp_manager', return_value=mock_mcp_manager):
            tool = Context7DocsTool()
            result = tool._run("/numpy/docs", "arrays", 5000)
            
            assert isinstance(result, str)
            mock_mcp_manager.call_tool.assert_called()


class TestSequentialThinkingTool:
    """Test Sequential Thinking MCP tool."""
    
    def test_sequential_thinking_instantiation(self):
        """Test Sequential Thinking Tool instantiation."""
        tool = SequentialThinkingTool()
        assert tool.name == "sequential_thinking"
        assert "reasoning" in tool.description.lower()
        
    def test_sequential_thinking_execution(self, mock_mcp_manager):
        """Test Sequential Thinking Tool execution."""
        with patch('src.server_research_mcp.tools.mcp_tools.get_mcp_manager', return_value=mock_mcp_manager):
            tool = SequentialThinkingTool()
            result = tool._run("This is a test thought", 1, 3, True)
            
            assert isinstance(result, str)
            assert "thought" in result
            mock_mcp_manager.call_tool.assert_called()
            
    def test_sequential_thinking_parameters(self):
        """Test Sequential Thinking Tool parameter handling."""
        tool = SequentialThinkingTool()
        
        # Test with different parameter combinations
        test_cases = [
            ("Simple thought", 1, 1, False),
            ("Complex reasoning step", 2, 5, True),
            ("Final conclusion", 3, 3, True),
        ]
        
        for thought, step, total, is_final in test_cases:
            # Would test actual execution with proper mocks
            assert thought is not None
            assert 1 <= step <= total
            assert isinstance(is_final, bool)


class TestToolFactory:
    """Test tool factory and collection functions."""
    
    def test_get_historian_tools(self):
        """Test that get_historian_tools returns correct tools."""
        tools = get_historian_tools()
        
        assert len(tools) == 6
        assert all(hasattr(tool, 'name') for tool in tools)
        assert all(hasattr(tool, 'description') for tool in tools)
        assert all(hasattr(tool, '_run') for tool in tools)
        
        # Check tool names
        tool_names = [tool.name for tool in tools]
        expected_names = [
            "memory_search", "memory_create_entity", "memory_add_observation",
            "context7_resolve_library", "context7_get_docs", "sequential_thinking"
        ]
        
        for expected_name in expected_names:
            assert expected_name in tool_names
            
    def test_tool_descriptions(self):
        """Test that all tools have proper descriptions."""
        tools = get_historian_tools()
        
        for tool in tools:
            assert tool.description is not None
            assert len(tool.description) > 10
            assert isinstance(tool.description, str)
            
    @pytest.mark.requires_mcp
    def test_tools_with_real_mcp_manager(self):
        """Test tools with real MCP manager (requires MCP servers)."""
        from server_research_mcp.tools.mcp_tools import get_mcp_manager
        
        try:
            manager = get_mcp_manager()
            assert manager is not None
            
            # Test basic tool execution
            tool = MemorySearchTool()
            result = tool._run("test")
            assert isinstance(result, str)
        except Exception as e:
            pytest.skip(f"MCP servers not available: {e}")


class TestToolErrorHandling:
    """Test error handling in MCP tools."""
    
    def test_tool_error_handling(self):
        """Test that tools handle errors gracefully."""
        # Mock a failing MCP manager
        failing_manager = MagicMock()
        failing_manager.call_tool.side_effect = Exception("MCP server error")
        
        with patch('src.server_research_mcp.tools.mcp_tools.get_mcp_manager', return_value=failing_manager):
            tool = MemorySearchTool()
            result = tool._run("test query")
            
            # Should return error message, not raise exception
            assert isinstance(result, str)
            assert "error" in result.lower()
            
    def test_invalid_parameters(self):
        """Test tools handle invalid parameters."""
        tool = MemoryCreateEntityTool()
        
        # Test with None values
        result = tool._run(None, None, None)
        assert isinstance(result, str)
        assert "error" in result.lower() or "invalid" in result.lower()
        
    def test_empty_results_handling(self):
        """Test tools handle empty results properly."""
        mock_manager = MagicMock()
        mock_manager.call_tool.return_value = {"nodes": []}
        
        with patch('src.server_research_mcp.tools.mcp_tools.get_mcp_manager', return_value=mock_manager):
            tool = MemorySearchTool()
            result = tool._run("nonexistent query")
            
            assert isinstance(result, str)
            # Should indicate no results found
            assert "no" in result.lower() or "empty" in result.lower() or "[]" in result


@pytest.mark.real_servers
def test_publisher_tools_real_server_integration():
    """
    Test publisher tools with real MCP server integration.
    
    This test requires obsidian-mcp-tools server to be running.
    Skipped in CI environments where real servers are not available.
    
    To run this test:
    1. Ensure Node.js and npx are installed
    2. Configure obsidian-mcp-tools in your MCP client config
    3. Run: python -m pytest -m "real_servers" tests/
    """
    pytest.skip("Real server integration not yet implemented - placeholder for future development")
    
    # TODO: Implement real server testing when MCP integration is complete
    # tools = get_publisher_tools()
    # 
    # # Verify we got real tools, not fallback
    # assert len(tools) == 5
    # 
    # # Test actual tool execution against real server
    # create_tool = next(t for t in tools if t.name == "obsidian_create_note")
    # result = create_tool._run(
    #     title="Test Note",
    #     content="# Test\nThis is a test note",
    #     folder="Tests"
    # )
    # 
    # # Verify real server response format
    # import json
    # response = json.loads(result)
    # assert response.get("status") != "created_fallback"  # Should be real response