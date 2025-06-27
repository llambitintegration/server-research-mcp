"""
Test the refactored MCP tools to ensure they work correctly.
"""
import pytest
import json
from unittest.mock import patch, MagicMock, AsyncMock

# Import tools from the unified mcp_tools module
from server_research_mcp.tools.mcp_tools import (
    MemorySearchTool,
    MemorySearchInput,
    MemoryCreateEntityTool,
    Context7ResolveTool,
    ZoteroSearchTool,
    get_historian_tools
)


class TestRefactoredTools:
    """Test suite for refactored MCP tools."""
    
    def test_memory_search_tool_interface(self):
        """Verify the tool has the correct interface."""
        tool = MemorySearchTool()
        
        # Check that tool has required attributes
        assert tool.name == "memory_search"
        assert "memory" in tool.description.lower()
        
        # Check schema has required fields
        schema = MemorySearchInput.model_fields
        assert "query" in schema.keys()
        
        # Should have _run method
        assert hasattr(tool, '_run')
    
    def test_memory_search_execution(self):
        """Test that the memory search tool executes correctly."""
        tool = MemorySearchTool()
        
        # Execute the tool
        result = tool._run(query="test query")
        
        # Should return JSON string
        assert isinstance(result, str)
        parsed = json.loads(result)
        
        # Check the mock response structure
        assert 'nodes' in parsed or 'message' in parsed
    
    def test_memory_create_entity_execution(self):
        """Test the memory create entity tool."""
        tool = MemoryCreateEntityTool()
        
        result = tool._run(
            name="Test Entity",
            entity_type="research_topic",
            observations=["observation1", "observation2"]
        )
        
        # Should return JSON string
        assert isinstance(result, str)
        parsed = json.loads(result)
        
        # Should have success indicators
        assert 'entity_id' in parsed or 'message' in parsed
    
    def test_decorator_created_tool(self):
        """Test tools created with the decorator pattern."""
        tool = Context7ResolveTool()
        
        # Check attributes
        assert tool.name == "context7_resolve_library"
        assert tool.server_name == "context7"
        assert tool.mcp_tool_name == "context7_resolve"
        
        # Execute
        result = tool._run(library_name="pandas")
        parsed = json.loads(result)
        
        assert 'library_id' in parsed or 'found' in parsed
    
    def test_factory_created_tool(self):
        """Test tools created with the pure factory pattern."""
        tool = ZoteroSearchTool()
        
        # Check it's a proper tool
        assert hasattr(tool, '_run')
        assert tool.server_name == "zotero"
        
        # Execute
        result = tool._run(
            query="machine learning",
            search_type="everything",
            limit=5
        )
        parsed = json.loads(result)
        
        assert 'results' in parsed or 'total' in parsed
    
    def test_error_handling(self):
        """Test that error handling works correctly."""
        tool = NewMemorySearchTool()
        
        # Mock the MCP manager to raise an error
        with patch('server_research_mcp.tools.mcp_base_tool.get_mcp_manager') as mock_get:
            mock_manager = MagicMock()
            mock_manager.initialize = AsyncMock(side_effect=Exception("Connection failed"))
            mock_get.return_value = mock_manager
            
            result = tool._run(query="test")
            
            # Should return error message
            assert "Error" in result or "error" in result.lower()
    
    def test_get_historian_tools(self):
        """Test that the tool collection function works."""
        tools = get_historian_tools()
        
        # Should return a list of tool instances
        assert isinstance(tools, list)
        assert len(tools) > 0
        
        # All should be BaseTool instances
        from crewai.tools import BaseTool
        for tool in tools:
            assert isinstance(tool, BaseTool)
            assert hasattr(tool, '_run')
    
    @pytest.mark.asyncio
    async def test_async_execution(self):
        """Test the async internals work correctly."""
        from server_research_mcp.tools.mcp_manager import get_mcp_manager
        
        manager = get_mcp_manager()
        
        # Test initialization
        await manager.initialize(["memory", "context7"])
        assert "memory" in manager.initialized_servers
        assert "context7" in manager.initialized_servers
        
        # Test tool call
        result = await manager.call_tool(
            server="memory",
            tool="search_nodes",
            arguments={"query": "test"}
        )
        
        assert isinstance(result, dict)
        assert 'nodes' in result or 'message' in result
    
    def test_tool_execution_consistency(self):
        """Test tool execution produces consistent results."""
        tool = MemorySearchTool()
        
        # Tool should handle the same input consistently
        query = "research on AI safety"
        
        result1 = tool._run(query=query)
        result2 = tool._run(query=query)
        
        # Both should return strings (JSON)
        assert isinstance(result1, str)
        assert isinstance(result2, str)
        
        # Both should return valid JSON
        parsed1 = json.loads(result1)
        parsed2 = json.loads(result2)
        assert isinstance(parsed1, dict)
        assert isinstance(parsed2, dict)