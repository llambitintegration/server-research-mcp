import pytest
import json
from unittest.mock import patch, MagicMock
from crewai.tools import BaseTool
from server_research_mcp.tools.mcp_tools import ParameterHandlers


class DummyZoteroSearchTool(BaseTool):
    """Minimal stub replicating Zotero schema expectations."""
    name: str = "zotero_search_items"
    description: str = "Dummy Zotero tool verifying `limit` type handling."

    # Zotero server expects `limit` as *string* per runtime error
    def _run(self, query: str, qmode: str = "everything", limit: str = "10"):
        # Emulate server-side validation
        assert isinstance(limit, str), f"'limit' should be str, got {type(limit)}"
        return {"success": True, "limit": limit}


def test_zotero_handler_converts_int_limit_to_str():
    """
    Regression test: Zotero parameter handler should convert int limit to string.
    
    The ParameterHandlers.zotero_handler must coerce an int → str before dispatch.
    This test specifically verifies the limit parameter type coercion.
    """
    # Test parameters with int limit (as LLM often provides)
    test_params = {
        "query": "test query",
        "limit": 10,  # This should be converted to string
        "qmode": "everything"
    }
    
    # Apply the Zotero parameter handler
    processed_params = ParameterHandlers.zotero_handler(test_params)
    
    # Verify the limit was converted to string
    assert isinstance(processed_params['limit'], str), f"'limit' should be str, got {type(processed_params['limit'])}"
    assert processed_params['limit'] == "10", f"Expected '10', got {processed_params['limit']}"
    
    # Verify other parameters remain unchanged
    assert processed_params['query'] == "test query"
    assert processed_params['qmode'] == "everything"


def test_zotero_handler_preserves_string_limit():
    """
    Test that zotero_handler preserves limit when it's already a string.
    """
    test_params = {
        "query": "test query",
        "limit": "25",  # Already a string
        "qmode": "everything"
    }
    
    processed_params = ParameterHandlers.zotero_handler(test_params)
    
    # Verify limit remains as string
    assert isinstance(processed_params['limit'], str)
    assert processed_params['limit'] == "25"


def test_zotero_handler_works_without_limit():
    """
    Test that zotero_handler works correctly when no limit parameter is provided.
    """
    test_params = {
        "query": "test query",
        "qmode": "everything"
    }
    
    processed_params = ParameterHandlers.zotero_handler(test_params)
    
    # Verify parameters remain unchanged
    assert processed_params == test_params
    assert "limit" not in processed_params


def test_zotero_handler_handles_edge_cases():
    """
    Test that zotero_handler handles edge cases like zero, negative numbers, etc.
    """
    test_cases = [
        {"limit": 0, "expected": "0"},
        {"limit": -1, "expected": "-1"},
        {"limit": 100, "expected": "100"},
        {"limit": 1.5, "expected": "1.5"},  # Float case
    ]
    
    for case in test_cases:
        test_params = {
            "query": "test query",
            "limit": case["limit"]
        }
        
        processed_params = ParameterHandlers.zotero_handler(test_params)
        
        assert isinstance(processed_params['limit'], str)
        assert processed_params['limit'] == case["expected"] 
