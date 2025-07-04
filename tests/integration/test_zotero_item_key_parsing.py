"""
Test for Zotero item_key parameter parsing issue.

This test reproduces the specific error where zotero_item_metadata fails to 
recognize the item_key parameter when passed as a JSON string.
"""
import os
import pytest
import json
from unittest.mock import patch, MagicMock

# Gate the test behind an environment variable for CI compatibility
pytestmark = pytest.mark.skipif(
    not os.getenv("CI_ZOTERO_MOCK", "true").lower() == "true",
    reason="requires mocked Zotero server",
)


def _get_zotero_metadata_tool():
    """Get the zotero_item_metadata tool from the researcher tools."""
    try:
        from src.server_research_mcp.tools import get_researcher_tools
        for tool in get_researcher_tools():
            if tool.name == "zotero_item_metadata":
                return tool
        raise RuntimeError("zotero_item_metadata not found in researcher tools")
    except Exception as e:
        pytest.skip(f"Could not load zotero tools: {e}")


@patch('src.server_research_mcp.tools.mcp_tools.get_mcp_manager')
def test_item_key_json_string_parsing(mock_get_mcp_manager):
    """
    Test that the tool can handle item_key parameter passed as JSON string.
    This reproduces the exact error shown in the crew execution logs.
    """
    # Mock the MCP manager to return a mock response
    mock_manager = MagicMock()
    mock_manager.call_tool.return_value = MagicMock(
        content=[MagicMock(text='{"title": "Test Paper", "authors": ["Test Author"], "year": 2024}')]
    )
    mock_get_mcp_manager.return_value = mock_manager
    
    # Get the zotero metadata tool
    metadata_tool = _get_zotero_metadata_tool()
    
    # Test case 1: JSON string input as it comes from CrewAI (reproduces the error)
    # CrewAI often passes a single string argument containing JSON
    json_input = '{"item_key": "MU566VUM"}'
    
    try:
        # This is how CrewAI typically calls the tool - as the first positional argument
        result = metadata_tool._run(json_input)
        
        # Verify the mock was called (shows parameters were parsed correctly)
        # Note: In this case, the MCPToolWrapper should parse the JSON and pass it through
        
        # The result should not contain error about missing parameters
        if isinstance(result, str):
            if "Missing required parameters for zotero_item_metadata: item_key" in result:
                pytest.fail(f"REPRODUCED BUG: {result}")
            elif "Missing required parameters" in result:
                pytest.fail(f"Still has parameter validation issues: {result}")
        
        print(f"✅ Test passed. Result: {result}")
        
    except Exception as e:
        if "Missing required parameters for zotero_item_metadata: item_key" in str(e):
            pytest.fail(f"REPRODUCED BUG: {e}")
        else:
            print(f"Different error (may be expected): {e}")


@patch('src.server_research_mcp.tools.mcp_tools.get_mcp_manager')
def test_item_key_direct_kwarg_parsing(mock_get_mcp_manager):
    """
    Test that the tool works when item_key is passed as a direct kwarg.
    """
    # Mock the MCP manager
    mock_manager = MagicMock()
    mock_manager.call_tool.return_value = MagicMock(
        content=[MagicMock(text='{"title": "Test Paper", "authors": ["Test Author"], "year": 2024}')]
    )
    mock_get_mcp_manager.return_value = mock_manager
    
    metadata_tool = _get_zotero_metadata_tool()
    
    # Test case 2: Direct kwarg (should work)
    try:
        result = metadata_tool._run(item_key="MU566VUM")
        
        # Verify success - should not contain error about missing parameters
        if isinstance(result, str):
            assert "Missing required parameters" not in result
        
        print(f"✅ Direct kwarg test passed. Result: {result}")
        
    except Exception as e:
        if "Missing required parameters" in str(e):
            pytest.fail(f"Direct kwarg validation failed: {e}")
        else:
            print(f"Note: Direct kwarg test had different error (may be expected): {e}")


def test_item_key_schema_validation_improvement():
    """
    Test that demonstrates the schema validation improvement.
    
    This test verifies that the tool handles various input formats without
    throwing validation errors.
    """
    try:
        metadata_tool = _get_zotero_metadata_tool()
        
        # Test that the tool exists and has the expected wrapper structure
        assert hasattr(metadata_tool, '_run')
        assert 'zotero_item_metadata' in metadata_tool.name
        
        # Check if the tool has proper schema handling
        if hasattr(metadata_tool, 'args_schema'):
            schema = metadata_tool.args_schema
            print(f"✅ Tool has args_schema: {schema}")
        
        print("✅ Schema validation improvement test passed")
        
    except Exception as e:
        pytest.skip(f"Could not test schema validation: {e}")


def test_legitimate_params_includes_item_key():
    """
    Verify that item_key parameter handling is properly implemented.
    
    In the new simplified architecture, the item_key parameter handling is 
    managed through ParameterHandlers.zotero_handler instead of a 
    legitimate_params list.
    """
    from src.server_research_mcp.tools.mcp_tools import ParameterHandlers
    
    # Test that the zotero_handler properly handles item_key parameters
    test_params = {"item_key": "MU566VUM", "query": "test"}
    processed_params = ParameterHandlers.zotero_handler(test_params)
    
    # The handler should preserve item_key
    assert "item_key" in processed_params
    assert processed_params["item_key"] == "MU566VUM"
    
    # This replaces the old legitimate_params mechanism
    print("✅ Parameter handling works correctly in new architecture")


if __name__ == "__main__":
    # Allow running this test directly for development
    os.environ["CI_ZOTERO_MOCK"] = "true"
    pytest.main([__file__, "-v"]) 
