"""MCP-specific test fixtures extracted from conftest.py."""

import pytest
from unittest.mock import MagicMock, patch
import asyncio


@pytest.fixture
def mock_mcp_manager():
    """Enhanced mock MCP manager with comprehensive tool support."""
    mock_manager = MagicMock()
    
    # Counter for unique entity IDs
    entity_counter = {"count": 0}
    
    def mock_get_historian_tools():
        """Mock historian tools - memory management with MCPAdapt interface"""
        tools = []
        tool_names = [
            'search_nodes', 'create_entities', 'create_relations', 'add_observations',
            'delete_entities', 'delete_observations', 'delete_relations', 'read_graph', 'open_nodes'
        ]
        for name in tool_names:
            mock_tool = MagicMock()
            mock_tool.name = name
            mock_tool.run = MagicMock(return_value={"results": [{"entity": "test"}], "success": True})
            tools.append(mock_tool)
        return tools
    
    def mock_get_researcher_tools():
        """Mock researcher tools - Zotero/research with MCPAdapt interface"""
        tools = []
        tool_names = [
            'zotero_search_items', 'zotero_item_metadata', 'zotero_item_fulltext',
            'resolve-library-id', 'get-library-docs', 'search_nodes'
        ]
        for name in tool_names:
            mock_tool = MagicMock()
            mock_tool.name = name  
            mock_tool.run = MagicMock(return_value={"results": [{"item": "test"}], "success": True})
            tools.append(mock_tool)
        return tools
    
    def mock_get_archivist_tools():
        """Mock archivist tools - analysis with MCPAdapt interface"""
        tools = []
        tool_names = ['sequentialthinking']
        for name in tool_names:
            mock_tool = MagicMock()
            mock_tool.name = name
            mock_tool.run = MagicMock(return_value={"status": "recorded", "success": True})
            tools.append(mock_tool)
        return tools
    
    def mock_get_publisher_tools():
        """Mock publisher tools - publishing with MCPAdapt interface"""  
        tools = []
        tool_names = [
            'create_entities', 'create_relations', 'add_observations', 'delete_entities',
            'search_nodes', 'read_graph', 'delete_observations', 'delete_relations',
            'open_nodes', 'create_directory', 'write_file'
        ]
        for name in tool_names:
            mock_tool = MagicMock()
            mock_tool.name = name
            mock_tool.run = MagicMock(return_value={"success": True, "entity_id": "test123"})
            tools.append(mock_tool)
        return tools
    
    # Assign tool getter methods
    mock_manager.get_historian_tools = mock_get_historian_tools
    mock_manager.get_researcher_tools = mock_get_researcher_tools
    mock_manager.get_archivist_tools = mock_get_archivist_tools
    mock_manager.get_publisher_tools = mock_get_publisher_tools
    
    def mock_list_tools():
        """Return comprehensive list of available tool names."""
        return [
            # Memory tools (historian)
            'search_nodes', 'create_entities', 'create_relations', 'add_observations',
            'delete_entities', 'delete_observations', 'delete_relations', 'read_graph', 'open_nodes',
            # Research tools  
            'zotero_search_items', 'zotero_item_metadata', 'zotero_item_fulltext',
            'resolve-library-id', 'get-library-docs',
            # Analysis tools
            'sequentialthinking',
            # Filesystem tools
            'create_directory', 'write_file', 'read_file', 'list_directory',
            # Legacy compatibility mappings
            'memory_search', 'memory_create_entity', 'memory_add_observation',
            'context7_resolve_library', 'context7_get_docs',
            'sequential_thinking_append_thought', 'sequential_thinking_get_thoughts'
        ]
    
    mock_manager.list_tools = mock_list_tools
    
    # Connection management
    mock_manager.is_connected = MagicMock(return_value=True)
    mock_manager.restart = MagicMock(return_value=True)
    mock_manager.shutdown = MagicMock(return_value=True)
    
    def mock_call_tool(tool_name, arguments=None, **kwargs):
        """Enhanced mock tool calling with comprehensive responses."""
        if arguments is None:
            arguments = kwargs
            
        # Handle tool name mappings for backwards compatibility
        tool_mapping = {
            'memory_search': 'search_nodes',
            'memory_create_entity': 'create_entities', 
            'memory_add_observation': 'add_observations',
            'context7_resolve_library': 'resolve-library-id',
            'context7_get_docs': 'get-library-docs',
            'sequential_thinking_append_thought': 'sequentialthinking',
            'sequential_thinking_get_thoughts': 'sequentialthinking'
        }
        
        actual_tool_name = tool_mapping.get(tool_name, tool_name)
        
        # Generate appropriate responses based on tool type
        if 'search' in actual_tool_name:
            query = arguments.get('query', '') if arguments else ''
            return {
                "results": [
                    {"entity": "artificial intelligence", "relevance": 0.95, "type": "concept"},
                    {"entity": "machine learning", "relevance": 0.87, "type": "technique"},
                    {"entity": "neural networks", "relevance": 0.82, "type": "method"}
                ],
                "query": query,
                "success": True
            }
        elif 'create_entities' in actual_tool_name:
            entities = arguments.get('entities', []) if arguments else []
            created_entities = []
            for i, entity in enumerate(entities):
                entity_counter["count"] += 1
                entity_id = f"entity_{entity_counter['count']}"
                created_entities.append({
                    "id": entity_id,
                    "entity_id": entity_id,
                    "name": entity.get('name', f'Entity {i+1}'),
                    "type": entity.get('entityType', 'concept'),
                    "observations": entity.get('observations', [])
                })
            return {
                "success": True,
                "entities": created_entities,
                "entity_id": created_entities[0]["entity_id"] if created_entities else "entity_1",
                "message": f"Created {len(created_entities)} entities successfully"
            }
        elif 'add_observations' in actual_tool_name:
            observations = arguments.get('observations', []) if arguments else []
            return {
                "success": True,
                "observations_added": len(observations),
                "message": f"Added {len(observations)} observations"
            }
        elif 'resolve' in actual_tool_name:
            return {
                "library_id": "/tensorflow/v2.x",
                "confidence": 0.92,
                "success": True
            }
        elif 'docs' in actual_tool_name or 'get-library-docs' in actual_tool_name:
            return {
                "content": "TensorFlow documentation content",
                "tokens_used": 8500,
                "sections": ["Introduction", "API Reference", "Examples"],
                "success": True
            }
        elif 'sequential' in actual_tool_name:
            if 'append' in tool_name or 'sequentialthinking' == actual_tool_name:
                return {
                    "status": "recorded",
                    "thought_id": f"thought_{entity_counter['count']}",
                    "success": True
                }
            elif 'get' in tool_name:
                return {
                    "thoughts": [
                        {"id": 1, "content": "Analysis step 1"},
                        {"id": 2, "content": "Analysis step 2"},
                        {"id": 3, "content": "Analysis step 3"}
                    ],
                    "success": True
                }
        elif 'zotero' in actual_tool_name:
            if 'search' in actual_tool_name:
                return {
                    "items": [
                        {"id": "test_item_1", "title": "AI Testing Methods", "year": 2024},
                        {"id": "test_item_2", "title": "Machine Learning Validation", "year": 2023}
                    ],
                    "success": True
                }
            elif 'metadata' in actual_tool_name or 'get_item' in tool_name:
                return {
                    "title": "Test Research Paper",
                    "authors": ["Author One", "Author Two"],
                    "year": 2024,
                    "sections": ["Abstract", "Introduction", "Methods", "Results", "Conclusion"],
                    "success": True
                }
            elif 'fulltext' in actual_tool_name:
                return {
                    "content": "This is the full text of the research paper...",
                    "success": True
                }
        elif 'write_file' in actual_tool_name or 'create_directory' in actual_tool_name:
            return {
                "success": True,
                "path": "/test/path",
                "message": "Operation completed successfully"
            }
        
        # Default response
        return {"success": True, "data": "mock_response", "status": "success"}
    
    mock_manager.call_tool = mock_call_tool
    
    # Async methods for live testing
    async def mock_initialize(servers):
        """Mock async initialization."""
        mock_manager.initialized_servers = servers
        return True
    
    async def mock_async_call_tool(server, tool, arguments):
        """Mock async tool calling."""
        return mock_call_tool(tool, arguments)
    
    mock_manager.initialize = mock_initialize
    mock_manager.async_call_tool = mock_async_call_tool
    mock_manager.initialized_servers = []
    mock_manager.adapters = {}
    
    return mock_manager


@pytest.fixture
def enable_enhanced_mcp():
    """Enable enhanced MCP mode for testing."""
    with patch.dict('os.environ', {'ENABLE_ENHANCED_MCP': 'true'}):
        yield


@pytest.fixture
def real_server_environment():
    """Real server environment for live testing (when available)."""
    import os
    
    # Check if real MCP servers are configured
    mcp_configured = (
        os.getenv('ZOTERO_USER_ID') and 
        os.getenv('ZOTERO_API_KEY') and
        os.getenv('ANTHROPIC_API_KEY')
    )
    
    if not mcp_configured:
        pytest.skip("Real MCP server environment not configured")
    
    yield {
        'zotero_user_id': os.getenv('ZOTERO_USER_ID'),
        'zotero_api_key': os.getenv('ZOTERO_API_KEY'),
        'anthropic_api_key': os.getenv('ANTHROPIC_API_KEY')
    }


@pytest.fixture(scope="function")
async def live_mcp_manager():
    """Live MCP manager for integration testing."""
    try:
        from server_research_mcp.tools.mcp_tools import get_mcp_manager
        manager = get_mcp_manager()
        
        # Initialize with available servers
        available_servers = ["memory", "sequential-thinking"]
        await manager.initialize(available_servers)
        
        yield manager
        
        # Cleanup
        await manager.shutdown()
    except ImportError:
        pytest.skip("MCP tools not available")
    except Exception as e:
        pytest.skip(f"MCP manager initialization failed: {e}")


@pytest.fixture
def mock_mcp_connection_issues():
    """Mock MCP connection issues for testing error handling."""
    def failing_call_tool(tool_name, arguments=None, **kwargs):
        if "timeout" in tool_name:
            raise asyncio.TimeoutError("Connection timeout")
        elif "connection" in tool_name:
            raise ConnectionError("Connection refused")
        else:
            return {"success": True, "data": "mock_response"}
    
    mock_manager = MagicMock()
    mock_manager.call_tool = failing_call_tool
    mock_manager.is_connected = MagicMock(return_value=False)
    
    return mock_manager


@pytest.fixture
def mcp_performance_monitor():
    """Monitor MCP performance during testing."""
    import time
    
    class MCPPerformanceMonitor:
        def __init__(self):
            self.call_times = []
            self.errors = []
        
        def record_call(self, tool_name, duration, success=True, error=None):
            self.call_times.append({
                'tool': tool_name,
                'duration': duration,
                'success': success,
                'timestamp': time.time()
            })
            if error:
                self.errors.append(error)
        
        def get_stats(self):
            if not self.call_times:
                return {"avg_duration": 0, "total_calls": 0, "error_rate": 0}
            
            avg_duration = sum(call['duration'] for call in self.call_times) / len(self.call_times)
            error_rate = len(self.errors) / len(self.call_times)
            
            return {
                "avg_duration": avg_duration,
                "total_calls": len(self.call_times),
                "error_rate": error_rate,
                "errors": self.errors
            }
    
    return MCPPerformanceMonitor() 