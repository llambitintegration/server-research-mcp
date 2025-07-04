"""
Unit tests for MCP tool registry, server configuration, and tool mapping functionality.
"""
import pytest
import os
import sys
from unittest.mock import patch, MagicMock, call
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from server_research_mcp.tools.mcp_tools import (
    MCPToolAdapter,
    ServerConfig,
    ToolMapping,
    MCPToolRegistry,
    ParameterHandlers,
    get_registry,
    setup_registry,
    get_historian_tools,
    get_researcher_tools,
    get_archivist_tools,
    get_publisher_tools,
    get_all_mcp_tools,
    SchemaValidationTool,
    IntelligentSummaryTool,
    add_basic_tools,
    MCPToolWrapper,
    _AdaptHolder
)
from mcp import StdioServerParameters


class TestMCPToolAdapter:
    """Test MCPToolAdapter functionality."""
    
    def test_adapter_initialization(self):
        """Test MCPToolAdapter initialization."""
        server_params = StdioServerParameters(command="test", args=["arg1"], env={})
        parameter_handlers = {"test": lambda x: x}
        
        adapter = MCPToolAdapter(
            name="test_adapter",
            server_params=server_params,
            parameter_handlers=parameter_handlers
        )
        
        assert adapter.name == "test_adapter"
        assert adapter.server_params == server_params
        assert adapter.parameter_handlers == parameter_handlers
        assert adapter._tools is None
        assert adapter._ctx is None
    
    @patch('server_research_mcp.tools.mcp_tools.MCPAdapt')
    def test_adapter_initialization_async(self, mock_mcp_adapt):
        """Test adapter async initialization."""
        server_params = StdioServerParameters(command="test", args=[], env={})
        adapter = MCPToolAdapter("test", server_params)
        
        mock_ctx = MagicMock()
        mock_tools = [MagicMock(), MagicMock()]
        mock_ctx.__enter__.return_value = mock_tools
        mock_mcp_adapt.return_value = mock_ctx
        
        import asyncio
        asyncio.run(adapter.initialize())
        
        assert adapter._tools == mock_tools
        assert adapter._ctx == mock_ctx
        mock_mcp_adapt.assert_called_once()
    
    @patch('server_research_mcp.tools.mcp_tools.MCPAdapt')
    def test_adapter_initialization_exception(self, mock_mcp_adapt):
        """Test adapter initialization with exception."""
        server_params = StdioServerParameters(command="test", args=[], env={})
        adapter = MCPToolAdapter("test", server_params)
        
        mock_mcp_adapt.side_effect = Exception("Test error")
        
        import asyncio
        asyncio.run(adapter.initialize())
        
        assert adapter._tools == []
        assert adapter._ctx is None
    
    @patch('server_research_mcp.tools.mcp_tools.MCPAdapt')
    def test_get_tools_lazy_initialization(self, mock_mcp_adapt):
        """Test get_tools with lazy initialization."""
        server_params = StdioServerParameters(command="test", args=[], env={})
        adapter = MCPToolAdapter("test", server_params)
        
        mock_ctx = MagicMock()
        mock_tools = [MagicMock(), MagicMock()]
        mock_ctx.__enter__.return_value = mock_tools
        mock_mcp_adapt.return_value = mock_ctx
        
        # First call should initialize
        tools = adapter.get_tools()
        assert tools == mock_tools
        
        # Second call should return cached tools
        tools2 = adapter.get_tools()
        assert tools2 == mock_tools
        
        # Should only initialize once
        mock_mcp_adapt.assert_called_once()
    
    def test_adapter_shutdown(self):
        """Test adapter shutdown."""
        server_params = StdioServerParameters(command="test", args=[], env={})
        adapter = MCPToolAdapter("test", server_params)
        
        mock_ctx = MagicMock()
        adapter._ctx = mock_ctx
        
        adapter.shutdown()
        
        mock_ctx.__exit__.assert_called_once_with(None, None, None)
    
    def test_adapter_shutdown_exception(self):
        """Test adapter shutdown with exception."""
        server_params = StdioServerParameters(command="test", args=[], env={})
        adapter = MCPToolAdapter("test", server_params)
        
        mock_ctx = MagicMock()
        mock_ctx.__exit__.side_effect = Exception("Shutdown error")
        adapter._ctx = mock_ctx
        
        # Should not raise exception
        adapter.shutdown()
    
    def test_adapter_shutdown_no_context(self):
        """Test adapter shutdown with no context."""
        server_params = StdioServerParameters(command="test", args=[], env={})
        adapter = MCPToolAdapter("test", server_params)
        
        # Should not raise exception
        adapter.shutdown()


class TestServerConfig:
    """Test ServerConfig dataclass."""
    
    def test_server_config_defaults(self):
        """Test ServerConfig with default values."""
        config = ServerConfig(name="test", command="test_cmd")
        
        assert config.name == "test"
        assert config.command == "test_cmd"
        assert config.args == []
        assert config.env == {}
        assert config.parameter_handlers == {}
        assert config.enabled is True
    
    def test_server_config_full_initialization(self):
        """Test ServerConfig with all parameters."""
        parameter_handlers = {"test": lambda x: x}
        config = ServerConfig(
            name="test",
            command="test_cmd",
            args=["arg1", "arg2"],
            env={"VAR": "value"},
            parameter_handlers=parameter_handlers,
            enabled=False
        )
        
        assert config.name == "test"
        assert config.command == "test_cmd"
        assert config.args == ["arg1", "arg2"]
        assert config.env == {"VAR": "value"}
        assert config.parameter_handlers == parameter_handlers
        assert config.enabled is False


class TestToolMapping:
    """Test ToolMapping dataclass."""
    
    def test_tool_mapping_defaults(self):
        """Test ToolMapping with default values."""
        mapping = ToolMapping(
            agent_name="test_agent",
            tool_patterns=["pattern1", "pattern2"]
        )
        
        assert mapping.agent_name == "test_agent"
        assert mapping.tool_patterns == ["pattern1", "pattern2"]
        assert mapping.required_count == 0
        assert mapping.fallback_enabled is True
    
    def test_tool_mapping_full_initialization(self):
        """Test ToolMapping with all parameters."""
        mapping = ToolMapping(
            agent_name="test_agent",
            tool_patterns=["pattern1"],
            required_count=5,
            fallback_enabled=False
        )
        
        assert mapping.agent_name == "test_agent"
        assert mapping.tool_patterns == ["pattern1"]
        assert mapping.required_count == 5
        assert mapping.fallback_enabled is False


class TestMCPToolRegistry:
    """Test MCPToolRegistry functionality."""
    
    def test_registry_initialization(self):
        """Test registry initialization."""
        registry = MCPToolRegistry()
        
        assert registry.servers == {}
        assert registry.mappings == []
        assert registry.adapters == {}
        assert registry._initialized is False
        assert registry._all_tools is None
    
    def test_register_server(self):
        """Test server registration."""
        registry = MCPToolRegistry()
        
        registry.register_server(
            "test_server",
            command="test_cmd",
            args=["arg1"],
            env={"VAR": "value"}
        )
        
        assert "test_server" in registry.servers
        config = registry.servers["test_server"]
        assert config.name == "test_server"
        assert config.command == "test_cmd"
        assert config.args == ["arg1"]
        assert config.env == {"VAR": "value"}
    
    def test_map_tools(self):
        """Test tool mapping."""
        registry = MCPToolRegistry()
        
        registry.map_tools(
            agent="test_agent",
            patterns=["pattern1", "pattern2"],
            required_count=5,
            fallback_enabled=False
        )
        
        assert len(registry.mappings) == 1
        mapping = registry.mappings[0]
        assert mapping.agent_name == "test_agent"
        assert mapping.tool_patterns == ["pattern1", "pattern2"]
        assert mapping.required_count == 5
        assert mapping.fallback_enabled is False
    
    @patch('server_research_mcp.tools.mcp_tools.MCPToolAdapter')
    def test_initialize_all(self, mock_adapter_class):
        """Test initialize_all method."""
        registry = MCPToolRegistry()
        
        # Register test servers
        registry.register_server(
            "test_server1",
            command="cmd1",
            args=["arg1"],
            enabled=True
        )
        registry.register_server(
            "test_server2",
            command="cmd2",
            args=["arg2"],
            enabled=False
        )
        
        mock_adapter = MagicMock()
        mock_adapter_class.return_value = mock_adapter
        
        registry.initialize_all()
        
        assert registry._initialized is True
        assert len(registry.adapters) == 1  # Only enabled server
        assert "test_server1" in registry.adapters
        assert "test_server2" not in registry.adapters
        mock_adapter_class.assert_called_once()
    
    def test_initialize_all_exception(self):
        """Test initialize_all with exception."""
        registry = MCPToolRegistry()
        
        # Register server with invalid config
        registry.register_server("test_server", command="")
        
        with patch('server_research_mcp.tools.mcp_tools.StdioServerParameters', side_effect=Exception("Test error")):
            registry.initialize_all()
        
        assert registry._initialized is True
        assert len(registry.adapters) == 0
    
    def test_get_all_tools(self):
        """Test get_all_tools method."""
        registry = MCPToolRegistry()
        
        mock_adapter1 = MagicMock()
        mock_adapter1.get_tools.return_value = [MagicMock(), MagicMock()]
        mock_adapter2 = MagicMock()
        mock_adapter2.get_tools.return_value = [MagicMock()]
        
        registry.adapters = {
            "adapter1": mock_adapter1,
            "adapter2": mock_adapter2
        }
        registry._initialized = True
        
        tools = registry.get_all_tools()
        
        assert len(tools) == 3
        assert registry._all_tools is not None
        
        # Second call should return cached tools
        tools2 = registry.get_all_tools()
        assert tools2 == tools
        
        # Should only call get_tools once per adapter
        mock_adapter1.get_tools.assert_called_once()
        mock_adapter2.get_tools.assert_called_once()
    
    def test_get_agent_tools_with_mapping(self):
        """Test get_agent_tools with mapping."""
        registry = MCPToolRegistry()
        
        # Add mapping
        registry.map_tools(
            agent="test_agent",
            patterns=["pattern1", "pattern2"],
            required_count=3,
            fallback_enabled=True
        )
        
        # Mock tools
        mock_tool1 = MagicMock()
        mock_tool1.name = "pattern1_tool"
        mock_tool2 = MagicMock()
        mock_tool2.name = "pattern2_tool"
        mock_tool3 = MagicMock()
        mock_tool3.name = "other_tool"
        
        with patch.object(registry, 'get_all_tools', return_value=[mock_tool1, mock_tool2, mock_tool3]):
            tools = registry.get_agent_tools("test_agent")
        
        assert len(tools) >= 2  # At least the matched tools
        assert mock_tool1 in tools
        assert mock_tool2 in tools
        assert mock_tool3 not in tools
    
    def test_get_agent_tools_no_mapping(self):
        """Test get_agent_tools without mapping."""
        registry = MCPToolRegistry()
        
        tools = registry.get_agent_tools("nonexistent_agent")
        assert tools == []
    
    def test_get_agent_tools_with_fallback(self):
        """Test get_agent_tools with fallback tools."""
        registry = MCPToolRegistry()
        
        # Add mapping with required count
        registry.map_tools(
            agent="test_agent",
            patterns=["pattern1"],
            required_count=5,
            fallback_enabled=True
        )
        
        # Mock only one matching tool
        mock_tool = MagicMock()
        mock_tool.name = "pattern1_tool"
        
        with patch.object(registry, 'get_all_tools', return_value=[mock_tool]):
            tools = registry.get_agent_tools("test_agent")
        
        assert len(tools) == 5  # 1 matched + 4 fallback
        assert mock_tool in tools
    
    def test_get_agent_tools_no_fallback(self):
        """Test get_agent_tools without fallback."""
        registry = MCPToolRegistry()
        
        # Add mapping with required count but no fallback
        registry.map_tools(
            agent="test_agent",
            patterns=["pattern1"],
            required_count=5,
            fallback_enabled=False
        )
        
        # Mock only one matching tool
        mock_tool = MagicMock()
        mock_tool.name = "pattern1_tool"
        
        with patch.object(registry, 'get_all_tools', return_value=[mock_tool]):
            tools = registry.get_agent_tools("test_agent")
        
        assert len(tools) == 1  # Only matched tool, no fallback
        assert mock_tool in tools
    
    def test_create_fallback_tools(self):
        """Test _create_fallback_tools method."""
        registry = MCPToolRegistry()
        
        fallback_tools = registry._create_fallback_tools("test_agent", 3)
        
        assert len(fallback_tools) == 3
        for i, tool in enumerate(fallback_tools):
            assert tool.name == f"test_agent_fallback_tool_{i+1}"
            assert "fallback" in tool.description.lower()
    
    def test_shutdown_all(self):
        """Test shutdown_all method."""
        registry = MCPToolRegistry()
        
        mock_adapter1 = MagicMock()
        mock_adapter2 = MagicMock()
        
        registry.adapters = {
            "adapter1": mock_adapter1,
            "adapter2": mock_adapter2
        }
        registry._initialized = True
        registry._all_tools = [MagicMock()]
        
        registry.shutdown_all()
        
        mock_adapter1.shutdown.assert_called_once()
        mock_adapter2.shutdown.assert_called_once()
        assert registry._initialized is False
        assert registry._all_tools is None


class TestParameterHandlers:
    """Test ParameterHandlers static methods."""
    
    def test_zotero_handler(self):
        """Test zotero parameter handler."""
        params = {"limit": 10, "query": "test"}
        
        result = ParameterHandlers.zotero_handler(params)
        
        assert result["limit"] == "10"  # Should be converted to string
        assert result["query"] == "test"  # Should remain unchanged
    
    def test_zotero_handler_no_limit(self):
        """Test zotero handler without limit parameter."""
        params = {"query": "test"}
        
        result = ParameterHandlers.zotero_handler(params)
        
        assert result == {"query": "test"}  # Should remain unchanged
    
    def test_filesystem_handler(self):
        """Test filesystem parameter handler."""
        params = {"path": "/some/path//with//double//slashes"}
        
        result = ParameterHandlers.filesystem_handler(params)
        
        # Path should be normalized
        assert "//" not in result["path"]
    
    def test_filesystem_handler_no_path(self):
        """Test filesystem handler without path parameter."""
        params = {"other": "value"}
        
        result = ParameterHandlers.filesystem_handler(params)
        
        assert result == {"other": "value"}
    
    def test_search_handler_valid(self):
        """Test search parameter handler with valid query."""
        params = {"query": "test query"}
        
        result = ParameterHandlers.search_handler(params)
        
        assert result["query"] == "test query"
    
    def test_search_handler_empty_query(self):
        """Test search handler with empty query."""
        params = {"query": ""}
        
        with pytest.raises(ValueError, match="Search query cannot be empty"):
            ParameterHandlers.search_handler(params)
    
    def test_search_handler_whitespace_query(self):
        """Test search handler with whitespace-only query."""
        params = {"query": "   "}
        
        with pytest.raises(ValueError, match="Search query cannot be empty"):
            ParameterHandlers.search_handler(params)
    
    def test_search_handler_no_query(self):
        """Test search handler without query parameter."""
        params = {"other": "value"}
        
        result = ParameterHandlers.search_handler(params)
        
        assert result == {"other": "value"}


class TestGlobalRegistryFunctions:
    """Test global registry functions."""
    
    def test_get_registry(self):
        """Test get_registry function."""
        registry = get_registry()
        
        assert isinstance(registry, MCPToolRegistry)
        
        # Should return the same instance
        registry2 = get_registry()
        assert registry is registry2
    
    @patch.dict(os.environ, {}, clear=True)
    def test_setup_registry_minimal(self):
        """Test setup_registry with minimal environment."""
        registry = MCPToolRegistry()
        
        with patch('server_research_mcp.tools.mcp_tools.get_registry', return_value=registry):
            setup_registry()
        
        # Should have registered memory, filesystem, and sequential thinking
        assert "memory" in registry.servers
        assert "filesystem" in registry.servers
        assert "sequential_thinking" in registry.servers
        assert "zotero" not in registry.servers  # No credentials
    
    @patch.dict(os.environ, {
        'ZOTERO_API_KEY': 'test_key',
        'ZOTERO_LIBRARY_ID': 'test_id'
    })
    def test_setup_registry_with_zotero(self):
        """Test setup_registry with Zotero credentials."""
        registry = MCPToolRegistry()
        
        with patch('server_research_mcp.tools.mcp_tools.get_registry', return_value=registry):
            setup_registry()
        
        # Should have registered all servers including Zotero
        assert "memory" in registry.servers
        assert "zotero" in registry.servers
        assert "filesystem" in registry.servers
        assert "sequential_thinking" in registry.servers
        
        # Check Zotero configuration
        zotero_config = registry.servers["zotero"]
        assert zotero_config.env["ZOTERO_API_KEY"] == "test_key"
        assert zotero_config.env["ZOTERO_LIBRARY_ID"] == "test_id"
    
    @patch.dict(os.environ, {'OBSIDIAN_VAULT_PATH': '/custom/path'})
    def test_setup_registry_custom_obsidian_path(self):
        """Test setup_registry with custom Obsidian path."""
        registry = MCPToolRegistry()
        
        with patch('server_research_mcp.tools.mcp_tools.get_registry', return_value=registry):
            setup_registry()
        
        # Check filesystem configuration
        filesystem_config = registry.servers["filesystem"]
        assert "/custom/path" in filesystem_config.args
    
    def test_setup_registry_tool_mappings(self):
        """Test that setup_registry creates tool mappings."""
        registry = MCPToolRegistry()
        
        with patch('server_research_mcp.tools.mcp_tools.get_registry', return_value=registry):
            setup_registry()
        
        # Should have mappings for all agents
        agent_names = [mapping.agent_name for mapping in registry.mappings]
        assert "historian" in agent_names
        assert "researcher" in agent_names
        assert "archivist" in agent_names
        assert "publisher" in agent_names
    
    def test_agent_tool_functions(self):
        """Test individual agent tool functions."""
        mock_registry = MagicMock()
        mock_tools = [MagicMock(), MagicMock()]
        mock_registry.get_agent_tools.return_value = mock_tools
        
        with patch('server_research_mcp.tools.mcp_tools.get_registry', return_value=mock_registry):
            historian_tools = get_historian_tools()
            researcher_tools = get_researcher_tools()
            archivist_tools = get_archivist_tools()
            publisher_tools = get_publisher_tools()
        
        assert historian_tools == mock_tools
        assert researcher_tools == mock_tools
        assert archivist_tools == mock_tools
        assert publisher_tools == mock_tools
        
        # Verify correct agent names were used
        expected_calls = [
            call("historian"),
            call("researcher"),
            call("archivist"),
            call("publisher")
        ]
        mock_registry.get_agent_tools.assert_has_calls(expected_calls)
    
    def test_get_all_mcp_tools(self):
        """Test get_all_mcp_tools function."""
        mock_tools = [MagicMock(), MagicMock()]
        
        # Patch each individual function to avoid complex mocking
        with patch('server_research_mcp.tools.mcp_tools.get_historian_tools', return_value=mock_tools):
            with patch('server_research_mcp.tools.mcp_tools.get_researcher_tools', return_value=mock_tools):
                with patch('server_research_mcp.tools.mcp_tools.get_archivist_tools', return_value=mock_tools):
                    with patch('server_research_mcp.tools.mcp_tools.get_publisher_tools', return_value=mock_tools):
                        all_tools = get_all_mcp_tools()
        
        assert isinstance(all_tools, dict)
        assert "historian" in all_tools
        assert "researcher" in all_tools
        assert "archivist" in all_tools
        assert "publisher" in all_tools
        assert "context7" in all_tools  # Legacy compatibility
        
        # All should return same tools except context7
        assert all_tools["historian"] == mock_tools
        assert all_tools["researcher"] == mock_tools
        assert all_tools["archivist"] == mock_tools  
        assert all_tools["publisher"] == mock_tools
        assert all_tools["context7"] == []  # Deprecated


class TestBasicTools:
    """Test basic tool implementations."""
    
    def test_schema_validation_tool(self):
        """Test SchemaValidationTool."""
        tool = SchemaValidationTool()
        
        assert tool.name == "schema_validation"
        assert "validate" in tool.description.lower()
        assert hasattr(tool, 'args_schema')
        
        # Test valid JSON
        valid_json = '{"title": "Test", "authors": ["Author1"]}'
        result = tool._run(valid_json)
        assert "validation passed" in result.lower()
        
        # Test invalid JSON
        invalid_json = '{"title": "Test"}'  # Missing required field
        result = tool._run(invalid_json)
        assert "validation failed" in result.lower()
    
    def test_schema_validation_tool_dict_input(self):
        """Test SchemaValidationTool with dict input."""
        tool = SchemaValidationTool()
        
        # Test valid dict
        valid_dict = {"title": "Test", "authors": ["Author1"]}
        result = tool._run(valid_dict)
        assert "validation passed" in result.lower()
        
        # Test invalid dict
        invalid_dict = {"title": "Test"}
        result = tool._run(invalid_dict)
        assert "validation failed" in result.lower()
    
    def test_schema_validation_tool_error(self):
        """Test SchemaValidationTool with error."""
        tool = SchemaValidationTool()
        
        # Test invalid JSON string
        invalid_json = "not json"
        result = tool._run(invalid_json)
        assert "validation error" in result.lower()
    
    def test_intelligent_summary_tool(self):
        """Test IntelligentSummaryTool."""
        tool = IntelligentSummaryTool()
        
        assert tool.name == "intelligent_summary"
        assert "summarize" in tool.description.lower()
        assert hasattr(tool, 'args_schema')
        
        # Test short content
        short_content = "This is a short text."
        result = tool._run(short_content)
        assert result == short_content
        
        # Test long content
        long_content = "a" * 1000
        result = tool._run(long_content, max_length=100)
        assert len(result) <= 100
        assert result.endswith("...")
    
    def test_intelligent_summary_tool_default_length(self):
        """Test IntelligentSummaryTool with default max_length."""
        tool = IntelligentSummaryTool()
        
        # Test with default max_length
        long_content = "a" * 1000
        result = tool._run(long_content)
        assert len(result) <= 500  # Default max_length
    
    def test_add_basic_tools(self):
        """Test add_basic_tools function."""
        existing_tools = [MagicMock(), MagicMock()]
        
        enhanced_tools = add_basic_tools(existing_tools)
        
        assert len(enhanced_tools) == len(existing_tools) + 2
        assert all(tool in enhanced_tools for tool in existing_tools)
        
        # Check that basic tools were added
        tool_names = [tool.name for tool in enhanced_tools if hasattr(tool, 'name')]
        assert "schema_validation" in tool_names
        assert "intelligent_summary" in tool_names


class TestLegacyCompatibility:
    """Test legacy compatibility classes and functions."""
    
    def test_mcp_tool_wrapper(self):
        """Test MCPToolWrapper backward compatibility."""
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool._run.return_value = "result"
        
        wrapper = MCPToolWrapper(mock_tool)
        
        result = wrapper._run(param1="value1", param2="value2")
        
        mock_tool._run.assert_called_once_with(param1="value1", param2="value2")
        assert result == "result"
    
    def test_mcp_tool_wrapper_zotero_tool(self):
        """Test MCPToolWrapper with Zotero tool."""
        mock_tool = MagicMock()
        mock_tool.name = "zotero_search"
        mock_tool._run.return_value = "result"
        
        wrapper = MCPToolWrapper(mock_tool)
        
        result = wrapper._run(limit=10, query="test")
        
        # Should call with limit converted to string
        mock_tool._run.assert_called_once_with(limit="10", query="test")
        assert result == "result"
    
    def test_mcp_tool_wrapper_callable_tool(self):
        """Test MCPToolWrapper with callable tool."""
        mock_tool = MagicMock()
        mock_tool.return_value = "result"
        # Remove _run method to force fallback to callable
        del mock_tool._run
        
        wrapper = MCPToolWrapper(mock_tool)
        
        result = wrapper._run(param="value")
        
        mock_tool.assert_called_once_with(param="value")
        assert result == "result"
    
    def test_mcp_tool_wrapper_fallback(self):
        """Test MCPToolWrapper fallback behavior."""
        # Create a non-callable object
        mock_tool = MagicMock()
        # Remove _run method to force fallback
        if hasattr(mock_tool, '_run'):
            del mock_tool._run
        # Make it raise TypeError when called (simulating non-callable)
        mock_tool.side_effect = TypeError("not callable")
        
        wrapper = MCPToolWrapper(mock_tool)
        
        result = wrapper._run(param="value")
        
        # Should return string representation of kwargs
        assert "param" in str(result)
        assert "value" in str(result)
    
    def test_adapt_holder(self):
        """Test _AdaptHolder backward compatibility."""
        tools = _AdaptHolder.get_all_tools()
        
        assert isinstance(tools, list)
        assert len(tools) == 0
    
    def test_get_mcp_manager(self):
        """Test get_mcp_manager function."""
        from server_research_mcp.tools.mcp_tools import get_mcp_manager
        
        manager = get_mcp_manager()
        
        assert hasattr(manager, 'initialized_servers')
        assert hasattr(manager, 'call_tool')
        assert callable(manager.call_tool)
        
        # Test call_tool
        result = manager.call_tool("test_tool", {"param": "value"})
        assert isinstance(result, str)
        assert "test_tool" in result
    
    def test_basic_tools_constant(self):
        """Test BASIC_TOOLS constant."""
        from server_research_mcp.tools.mcp_tools import BASIC_TOOLS
        
        assert isinstance(BASIC_TOOLS, list)
        assert len(BASIC_TOOLS) == 2
        
        tool_names = [tool.name for tool in BASIC_TOOLS]
        assert "schema_validation" in tool_names
        assert "intelligent_summary" in tool_names


class TestIntegrationScenarios:
    """Test integration scenarios and edge cases."""
    
    def test_complete_registry_workflow(self):
        """Test complete registry workflow from setup to tool retrieval."""
        # Create fresh registry
        registry = MCPToolRegistry()
        
        # Register servers
        registry.register_server(
            "test_server",
            command="test_cmd",
            args=["arg1"],
            parameter_handlers={"test": ParameterHandlers.zotero_handler}
        )
        
        # Add tool mapping
        registry.map_tools(
            agent="test_agent",
            patterns=["test"],
            required_count=2,
            fallback_enabled=True
        )
        
        # Mock adapter
        mock_adapter = MagicMock()
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_adapter.get_tools.return_value = [mock_tool]
        
        with patch('server_research_mcp.tools.mcp_tools.MCPToolAdapter', return_value=mock_adapter):
            # Initialize registry
            registry.initialize_all()
            
            # Get tools for agent
            tools = registry.get_agent_tools("test_agent")
            
            # Should have 2 tools (1 matched + 1 fallback)
            assert len(tools) == 2
            assert mock_tool in tools
            
            # Shutdown
            registry.shutdown_all()
            
            mock_adapter.shutdown.assert_called_once()
    
    def test_error_resilience(self):
        """Test error resilience in registry operations."""
        registry = MCPToolRegistry()
        
        # Register server that will fail
        registry.register_server("failing_server", command="")
        
        # This should not raise exception
        registry.initialize_all()
        
        # Should still work for other operations
        tools = registry.get_agent_tools("nonexistent_agent")
        assert tools == []
        
        # Shutdown should not raise
        registry.shutdown_all()
    
    def test_concurrent_access_simulation(self):
        """Test simulation of concurrent access patterns."""
        registry = MCPToolRegistry()
        
        # Multiple registrations
        for i in range(5):
            registry.register_server(f"server_{i}", command=f"cmd_{i}")
            registry.map_tools(f"agent_{i}", [f"pattern_{i}"])
        
        # Should handle multiple initializations gracefully
        registry.initialize_all()
        registry.initialize_all()  # Second call should be idempotent
        
        assert registry._initialized is True
        assert len(registry.servers) == 5
        assert len(registry.mappings) == 5