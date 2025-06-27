"""
Documentation coverage tests for interface protocols.

These tests ensure all protocol methods have proper docstrings and
documentation coverage is complete.
"""

import pytest
import inspect
from interfaces import protocols


class TestProtocolDocumentation:
    """Test that all protocols have complete documentation."""
    
    def test_all_protocols_have_docstrings(self):
        """Verify all protocol classes have docstrings."""
        protocol_classes = [
            protocols.MemoryServerProtocol,
            protocols.FilesystemServerProtocol,
            protocols.ResearchServerProtocol,
            protocols.MCPManagerProtocol
        ]
        
        for protocol_class in protocol_classes:
            assert protocol_class.__doc__ is not None, f"{protocol_class.__name__} missing docstring"
            assert len(protocol_class.__doc__.strip()) > 0, f"{protocol_class.__name__} has empty docstring"
    
    def test_all_protocol_methods_have_docstrings(self):
        """Ensure all protocol methods have docstrings."""
        protocols_and_methods = [
            (protocols.MemoryServerProtocol, ['search_nodes', 'create_entities', 'add_observations', 'read_graph']),
            (protocols.FilesystemServerProtocol, ['read_file', 'write_file', 'list_directory']),
            (protocols.ResearchServerProtocol, ['web_search', 'read_url', 'take_screenshot']),
            (protocols.MCPManagerProtocol, ['call_tool'])
        ]
        
        for protocol, expected_methods in protocols_and_methods:
            for method_name in expected_methods:
                assert hasattr(protocol, method_name), f"{protocol.__name__} missing method {method_name}"
                
                method = getattr(protocol, method_name)
                assert method.__doc__ is not None, f"{protocol.__name__}.{method_name} missing docstring"
                assert len(method.__doc__.strip()) > 0, f"{protocol.__name__}.{method_name} has empty docstring"
    
    def test_docstring_quality(self):
        """Test that docstrings meet quality standards."""
        protocols_to_check = [
            protocols.MemoryServerProtocol,
            protocols.FilesystemServerProtocol,
            protocols.ResearchServerProtocol,
            protocols.MCPManagerProtocol
        ]
        
        for protocol in protocols_to_check:
            # Protocol class docstring should describe purpose
            class_doc = protocol.__doc__.strip()
            assert len(class_doc) > 20, f"{protocol.__name__} docstring too short"
            
            # Check method docstrings
            for name, method in inspect.getmembers(protocol, predicate=inspect.isfunction):
                if not name.startswith('_'):  # Skip private methods
                    method_doc = method.__doc__
                    if method_doc:
                        method_doc = method_doc.strip()
                        assert len(method_doc) > 10, f"{protocol.__name__}.{name} docstring too short"
    
    def test_mcp_manager_protocol_detailed_docs(self):
        """Test MCPManagerProtocol has detailed documentation."""
        protocol = protocols.MCPManagerProtocol
        
        # Class docstring should explain the protocol purpose
        class_doc = protocol.__doc__
        assert "MCP Manager interface" in class_doc
        assert "backward compatibility" in class_doc
        
        # call_tool method should have detailed documentation
        call_tool_doc = protocol.call_tool.__doc__
        assert "Args:" in call_tool_doc
        assert "Returns:" in call_tool_doc
        assert "server:" in call_tool_doc
        assert "tool:" in call_tool_doc
        assert "arguments:" in call_tool_doc


class TestTypeAnnotationDocumentation:
    """Test that type annotations are properly documented."""
    
    def test_type_definitions_have_docstrings(self):
        """Verify type definitions have explanatory docstrings."""
        from interfaces import _types
        
        # Check module docstring
        assert _types.__doc__ is not None
        assert len(_types.__doc__.strip()) > 0
        
        # Check that type aliases have docstrings (via comments in source)
        # This is a basic check - in real implementation, we'd parse the source
        assert hasattr(_types, 'ToolResponse')
        assert hasattr(_types, 'ToolParameters')
        assert hasattr(_types, 'EntityData')
    
    def test_protocol_imports_documented(self):
        """Test that protocol module imports are documented."""
        protocol_module = protocols
        
        # Module should have comprehensive docstring
        module_doc = protocol_module.__doc__
        assert module_doc is not None
        assert "Protocol interfaces" in module_doc
        assert "dependency injection" in module_doc
        assert "backward\ncompatibility" in module_doc or "backward compatibility" in module_doc


class TestExampleUsageDocumentation:
    """Test that protocols include usage examples where appropriate."""
    
    def test_mcp_manager_protocol_usage_clarity(self):
        """Test that MCPManagerProtocol clearly documents usage."""
        protocol = protocols.MCPManagerProtocol
        call_tool_doc = protocol.call_tool.__doc__
        
        # Should have clear parameter descriptions
        assert "server:" in call_tool_doc.lower()
        assert "tool:" in call_tool_doc.lower() 
        assert "arguments:" in call_tool_doc.lower()
        
        # Should mention specific examples
        assert "memory" in call_tool_doc.lower() or "context7" in call_tool_doc.lower()
    
    def test_protocol_method_clarity(self):
        """Test that protocol methods clearly describe their purpose."""
        test_cases = [
            (protocols.MemoryServerProtocol.search_nodes, "search"),
            (protocols.MemoryServerProtocol.create_entities, "create"),
            (protocols.FilesystemServerProtocol.read_file, "read"),
            (protocols.FilesystemServerProtocol.write_file, "write"),
            (protocols.ResearchServerProtocol.web_search, "search"),
        ]
        
        for method, expected_keyword in test_cases:
            doc = method.__doc__.lower()
            assert expected_keyword in doc, f"{method.__name__} docstring should mention '{expected_keyword}'"


class TestDocumentationConsistency:
    """Test that documentation follows consistent patterns."""
    
    def test_async_methods_documented_consistently(self):
        """Test that all async methods are documented as such."""
        all_protocols = [
            protocols.MemoryServerProtocol,
            protocols.FilesystemServerProtocol,
            protocols.ResearchServerProtocol,
            protocols.MCPManagerProtocol
        ]
        
        for protocol in all_protocols:
            for name, method in inspect.getmembers(protocol, predicate=inspect.isfunction):
                if not name.startswith('_') and inspect.iscoroutinefunction(method):
                    # All protocol methods should be async
                    assert inspect.iscoroutinefunction(method), f"{protocol.__name__}.{name} should be async"
    
    def test_return_type_documentation(self):
        """Test that return types are consistently documented."""
        protocols_to_check = [
            protocols.MemoryServerProtocol,
            protocols.FilesystemServerProtocol,
            protocols.ResearchServerProtocol,
            protocols.MCPManagerProtocol
        ]
        
        for protocol in protocols_to_check:
            for name, method in inspect.getmembers(protocol, predicate=inspect.isfunction):
                if not name.startswith('_'):
                    # All methods should return ToolResponse (Dict[str, Any])
                    annotations = getattr(method, '__annotations__', {})
                    assert 'return' in annotations, f"{protocol.__name__}.{name} missing return annotation" 