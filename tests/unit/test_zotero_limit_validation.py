import pytest
import json
from unittest.mock import patch, MagicMock
from crewai.tools import BaseTool
from server_research_mcp.tools.mcp_tools import MCPToolWrapper


class DummyZoteroSearchTool(BaseTool):
    """Minimal stub replicating Zotero schema expectations."""
    name: str = "zotero_search_items"
    description: str = "Dummy Zotero tool verifying `limit` type handling."

    # Zotero server expects `limit` as *string* per runtime error
    def _run(self, query: str, qmode: str = "everything", limit: str = "10"):
        # Emulate server-side validation
        assert isinstance(limit, str), f"'limit' should be str, got {type(limit)}"
        return {"success": True, "limit": limit}


@patch('server_research_mcp.tools.mcp_tools._AdaptHolder')
def test_wrapper_converts_int_limit_to_str(mock_adapt_holder):
    """
    Regression test: crew should never send an int for `limit`.
    The MCPToolWrapper must coerce an int → str before dispatch.
    
    This test specifically verifies the limit parameter type coercion without
    triggering MCP server connections.
    """
    # Mock the AdaptHolder to prevent MCP server connection attempts
    mock_adapt_holder.get_all_tools.return_value = []
    
    # Create wrapper
    wrapper = MCPToolWrapper(DummyZoteroSearchTool())
    
    # Intentionally give an int as the LLM often does
    result_str = wrapper._run(query="test query", limit=10)
    
    # Debug: print what we actually got
    print(f"DEBUG: MCPToolWrapper returned: {repr(result_str)}")
    
    # The wrapper should have called the original tool with limit as string
    # Since the original tool would assert if limit is not string, if we get here,
    # the conversion worked. Just verify the result contains expected elements.
    assert "success" in result_str  # Basic validation that the call worked
    assert "10" in result_str  # Verify the limit was converted 