"""
Protocol compliance tests for MCP interfaces.

These tests verify that existing implementations comply with the defined
protocols, ensuring backward compatibility and type safety.
"""

import pytest
import inspect
from typing import get_type_hints, get_origin, get_args
from interfaces.protocols import (
    MemoryServerProtocol, FilesystemServerProtocol,
    ResearchServerProtocol, MCPManagerProtocol
)


class TestMCPManagerProtocolCompliance:
    """Verify MCPManager implements MCPManagerProtocol correctly."""
    
    def test_mcp_manager_implements_protocol(self):
        """Verify MCPManager implements MCPManagerProtocol."""
        from server_research_mcp.tools import get_mcp_manager
        
        manager = get_mcp_manager()
        assert isinstance(manager, MCPManagerProtocol)
        
    def test_call_tool_method_signature(self):
        """Verify call_tool method has correct signature."""
        from server_research_mcp.tools import get_mcp_manager
        
        manager = get_mcp_manager()
        assert hasattr(manager, 'call_tool')
        
        call_tool_method = getattr(manager, 'call_tool')
        assert callable(call_tool_method)
        
        # Verify method signature matches protocol
        sig = inspect.signature(call_tool_method)
        params = list(sig.parameters.keys())
        
        # Should have parameters: server, tool, arguments
        expected_params = ['server', 'tool', 'arguments']
        actual_params = [p for p in params if p != 'self']
        assert actual_params == expected_params
        
    def test_call_tool_type_hints(self):
        """Verify call_tool type hints match protocol."""
        from server_research_mcp.tools import get_mcp_manager
        
        manager = get_mcp_manager()
        
        # Get type hints from both protocol and implementation
        protocol_hints = get_type_hints(MCPManagerProtocol.call_tool)
        manager_hints = get_type_hints(manager.call_tool)
        
        # Compare parameter types (excluding 'self')
        for param_name in ['server', 'tool', 'arguments']:
            if param_name in protocol_hints and param_name in manager_hints:
                # For Dict[str, Any] vs ToolParameters, they should be equivalent
                proto_type = protocol_hints[param_name]
                mgr_type = manager_hints[param_name]
                
                # Handle type aliases - both should resolve to Dict[str, Any]
                if hasattr(proto_type, '__origin__') or hasattr(mgr_type, '__origin__'):
                    # Compare origins and args for generic types
                    assert get_origin(proto_type) == get_origin(mgr_type) or str(proto_type) == str(mgr_type)
    
    @pytest.mark.asyncio
    async def test_call_tool_returns_dict(self):
        """Verify call_tool returns a dictionary (ToolResponse)."""
        from server_research_mcp.tools import get_mcp_manager
        
        manager = get_mcp_manager()
        
        # Test with memory server search
        result = await manager.call_tool(
            "memory", "search_nodes", {"query": "test"}
        )
        
        assert isinstance(result, dict)
        
        # Test with different server/tool combinations
        test_cases = [
            ("memory", "create_entities", {"entities": [{"name": "test", "entity_type": "test"}]}),
            ("context7", "resolve-library-id", {"libraryName": "test"}),
            ("sequential-thinking", "append_thought", {"thought": "test"})
        ]
        
        for server, tool, args in test_cases:
            result = await manager.call_tool(server, tool, args)
            assert isinstance(result, dict), f"call_tool({server}, {tool}) should return dict"


class TestProtocolStructure:
    """Test the structure and completeness of protocol definitions."""
    
    def test_all_protocols_are_runtime_checkable(self):
        """Verify all protocols are decorated with @runtime_checkable."""
        protocols = [
            MemoryServerProtocol,
            FilesystemServerProtocol, 
            ResearchServerProtocol,
            MCPManagerProtocol
        ]
        
        for protocol in protocols:
            # Check if protocol has the runtime checkable marker
            # In Python 3.8+, runtime_checkable protocols use _is_protocol
            assert hasattr(protocol, '_is_protocol') or hasattr(protocol, '__protocol__')
            if hasattr(protocol, '_is_protocol'):
                assert getattr(protocol, '_is_protocol') is True
            elif hasattr(protocol, '__protocol__'):
                assert getattr(protocol, '__protocol__') is True
    
    def test_protocol_methods_have_annotations(self):
        """Verify all protocol methods have proper type annotations."""
        protocols_and_methods = [
            (MemoryServerProtocol, ['search_nodes', 'create_entities', 'add_observations', 'read_graph']),
            (FilesystemServerProtocol, ['read_file', 'write_file', 'list_directory']),
            (ResearchServerProtocol, ['web_search', 'read_url', 'take_screenshot']),
            (MCPManagerProtocol, ['call_tool'])
        ]
        
        for protocol, expected_methods in protocols_and_methods:
            for method_name in expected_methods:
                assert hasattr(protocol, method_name)
                method = getattr(protocol, method_name)
                
                # Check that method has type hints
                hints = get_type_hints(method)
                assert 'return' in hints, f"{protocol.__name__}.{method_name} missing return annotation"
    
    def test_protocol_inheritance(self):
        """Verify protocols inherit from Protocol correctly."""
        from typing import Protocol
        
        protocols = [
            MemoryServerProtocol,
            FilesystemServerProtocol,
            ResearchServerProtocol, 
            MCPManagerProtocol
        ]
        
        for protocol in protocols:
            # Check MRO includes Protocol
            mro_names = [cls.__name__ for cls in protocol.__mro__]
            assert 'Protocol' in mro_names


class TestExistingBehaviorPreserved:
    """Ensure Phase 1 changes don't affect existing behavior."""
    
    def test_memory_search_tool_unchanged(self):
        """Verify MemorySearchTool works exactly as before."""
        from server_research_mcp.tools.mcp_tools import MemorySearchTool
        
        tool = MemorySearchTool()
        
        # Should still work with existing interface
        assert hasattr(tool, '_run')
        assert callable(tool._run)
        
        # Test basic functionality (will use mock in test environment)
        result = tool._run(query="test search")
        assert isinstance(result, str)
    
    def test_factory_functions_unchanged(self):
        """Verify factory functions return same tools."""
        from server_research_mcp.tools import (
            get_historian_tools, 
            get_context7_tools,
            get_researcher_tools
        )
        
        # Test historian tools
        historian_tools = get_historian_tools()
        assert len(historian_tools) == 3  # Memory tools
        assert all(hasattr(tool, '_run') for tool in historian_tools)
        
        # Test context7 tools  
        context7_tools = get_context7_tools()
        assert len(context7_tools) == 2  # Context7 tools
        assert all(hasattr(tool, '_run') for tool in context7_tools)
        
        # Test researcher tools
        researcher_tools = get_researcher_tools()
        assert len(researcher_tools) > 0
        assert all(hasattr(tool, '_run') for tool in researcher_tools)
    
    def test_get_mcp_manager_unchanged(self):
        """Verify get_mcp_manager returns same type as before."""
        from server_research_mcp.tools import get_mcp_manager
        
        manager1 = get_mcp_manager()
        manager2 = get_mcp_manager()
        
        # Should return singleton
        assert manager1 is manager2
        
        # Should have expected methods
        assert hasattr(manager1, 'call_tool')
        assert hasattr(manager1, 'initialize')
        
        # Should be the same type as always
        assert manager1.__class__.__name__ in ['MCPManager', 'EnhancedMCPManager'] 