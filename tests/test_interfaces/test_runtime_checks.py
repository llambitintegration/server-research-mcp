"""
Runtime protocol checking tests.

These tests verify that @runtime_checkable protocols work correctly at runtime,
allowing isinstance() and issubclass() checks to function properly.
"""

import pytest
from interfaces.protocols import (
    MemoryServerProtocol,
    FilesystemServerProtocol, 
    ResearchServerProtocol,
    MCPManagerProtocol
)


class TestRuntimeProtocolChecking:
    """Test runtime protocol checking with isinstance and issubclass."""
    
    def test_mcp_manager_isinstance_check(self):
        """Verify MCPManager passes isinstance check for MCPManagerProtocol."""
        from server_research_mcp.tools import get_mcp_manager
        
        manager = get_mcp_manager()
        
        # Should pass isinstance check
        assert isinstance(manager, MCPManagerProtocol)
        
        # Should also work with the class
        manager_class = manager.__class__
        assert issubclass(manager_class, MCPManagerProtocol)
    
    def test_protocol_isinstance_with_mock(self):
        """Test protocol isinstance checks work with mock objects."""
        from unittest.mock import MagicMock, AsyncMock
        
        # Create a mock that has the required methods
        mock_manager = MagicMock()
        mock_manager.call_tool = AsyncMock()
        
        # Should pass isinstance check if it has the right methods
        assert isinstance(mock_manager, MCPManagerProtocol)
    
    def test_protocol_isinstance_fails_correctly(self):
        """Test isinstance returns False for objects missing required methods."""
        class IncompleteManager:
            """Manager missing required methods."""
            pass
        
        incomplete = IncompleteManager()
        
        # Should fail isinstance check
        assert not isinstance(incomplete, MCPManagerProtocol)
    
    def test_protocol_isinstance_with_wrong_signature(self):
        """Test isinstance behavior with methods that have wrong signatures."""
        class WrongSignatureManager:
            """Manager with wrong method signature."""
            async def call_tool(self, wrong_param):
                pass
        
        wrong_manager = WrongSignatureManager()
        
        # isinstance should still pass (it only checks method existence)
        # Signature checking is done separately in compliance tests
        assert isinstance(wrong_manager, MCPManagerProtocol)


class TestProtocolAttributeChecking:
    """Test that protocols correctly identify required attributes."""
    
    def test_memory_protocol_methods(self):
        """Test MemoryServerProtocol method requirements."""
        required_methods = ['search_nodes', 'create_entities', 'add_observations', 'read_graph']
        
        class MockMemoryServer:
            async def search_nodes(self, query): pass
            async def create_entities(self, entities): pass
            async def add_observations(self, observations): pass
            async def read_graph(self): pass
        
        mock_server = MockMemoryServer()
        assert isinstance(mock_server, MemoryServerProtocol)
        
        # Test with missing method
        class IncompleteMemoryServer:
            async def search_nodes(self, query): pass
            # Missing other methods
        
        incomplete_server = IncompleteMemoryServer()
        assert not isinstance(incomplete_server, MemoryServerProtocol)
    
    def test_filesystem_protocol_methods(self):
        """Test FilesystemServerProtocol method requirements."""
        class MockFilesystemServer:
            async def read_file(self, path): pass
            async def write_file(self, path, content): pass
            async def list_directory(self, path): pass
        
        mock_server = MockFilesystemServer()
        assert isinstance(mock_server, FilesystemServerProtocol)
    
    def test_research_protocol_methods(self):
        """Test ResearchServerProtocol method requirements."""
        class MockResearchServer:
            async def web_search(self, query): pass
            async def read_url(self, url): pass
            async def take_screenshot(self, url): pass
        
        mock_server = MockResearchServer()
        assert isinstance(mock_server, ResearchServerProtocol)


class TestProtocolComposition:
    """Test protocol composition and multiple inheritance scenarios."""
    
    def test_class_implementing_multiple_protocols(self):
        """Test a class that implements multiple protocols."""
        class MultiProtocolServer:
            async def call_tool(self, server, tool, arguments): pass
            async def search_nodes(self, query): pass
            async def create_entities(self, entities): pass
            async def add_observations(self, observations): pass
            async def read_graph(self): pass
        
        multi_server = MultiProtocolServer()
        
        # Should implement both protocols
        assert isinstance(multi_server, MCPManagerProtocol)
        assert isinstance(multi_server, MemoryServerProtocol)
    
    def test_protocol_with_additional_methods(self):
        """Test that classes with extra methods still implement protocols."""
        class ExtendedManager:
            async def call_tool(self, server, tool, arguments): pass
            async def extra_method(self): pass
            def sync_method(self): pass
        
        extended = ExtendedManager()
        
        # Should still implement the protocol despite extra methods
        assert isinstance(extended, MCPManagerProtocol)


class TestProtocolErrorHandling:
    """Test error handling and edge cases in protocol checking."""
    
    def test_protocol_with_none_values(self):
        """Test protocol checking with None values."""
        assert not isinstance(None, MCPManagerProtocol)
        assert not isinstance(None, MemoryServerProtocol)
    
    def test_protocol_with_primitives(self):
        """Test protocol checking with primitive types."""
        primitives = [1, "string", [], {}, set()]
        
        for primitive in primitives:
            assert not isinstance(primitive, MCPManagerProtocol)
            assert not isinstance(primitive, MemoryServerProtocol)
    
    def test_protocol_with_builtin_types(self):
        """Test protocol checking with built-in types."""
        builtins = [dict, list, str, int]
        
        for builtin_type in builtins:
            assert not isinstance(builtin_type, MCPManagerProtocol)
            assert not issubclass(builtin_type, MCPManagerProtocol) 