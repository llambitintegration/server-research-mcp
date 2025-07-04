"""
Test agent tool assignments to ensure proper tool isolation.

This test ensures that each agent gets the correct tools and prevents
cross-contamination of tools between agents.
"""
import pytest
from unittest.mock import patch
from server_research_mcp.tools.mcp_tools import get_registry, setup_registry


class TestAgentToolAssignments:
    """Test proper tool assignment isolation for each agent."""
    
    def test_historian_gets_memory_tools_only(self):
        """Test historian gets memory tools and no Zotero/filesystem tools."""
        registry = get_registry()
        tools = registry.get_agent_tools("historian")
        
        # Should have memory tools
        memory_tool_names = [tool.name for tool in tools]
        expected_memory_tools = [
            'create_entities', 'create_relations', 'add_observations',
            'delete_entities', 'delete_relations', 'delete_observations',
            'read_graph', 'search_nodes', 'open_nodes'
        ]
        
        # Verify all expected memory tools are present
        for expected_tool in expected_memory_tools:
            assert any(expected_tool in name for name in memory_tool_names), \
                f"Historian missing expected memory tool: {expected_tool}"
        
        # Should NOT have Zotero tools
        zotero_tools = [name for name in memory_tool_names if 'zotero' in name.lower()]
        assert len(zotero_tools) == 0, f"Historian incorrectly assigned Zotero tools: {zotero_tools}"
        
        # Should NOT have general filesystem tools (except via memory server)
        bad_filesystem_tools = [name for name in memory_tool_names 
                               if any(pattern in name.lower() for pattern in ['write_file', 'edit_file', 'list_directory'])]
        assert len(bad_filesystem_tools) == 0, f"Historian incorrectly assigned filesystem tools: {bad_filesystem_tools}"

    def test_researcher_gets_zotero_tools_only(self):
        """Test researcher gets Zotero tools and no memory/filesystem tools."""
        registry = get_registry()
        tools = registry.get_agent_tools("researcher")
        
        # Should have exactly 3 Zotero tools
        assert len(tools) == 3, f"Researcher should have 3 tools, got {len(tools)}"
        
        tool_names = [tool.name for tool in tools]
        expected_zotero_tools = ['zotero_item_metadata', 'zotero_item_fulltext', 'zotero_search_items']
        
        # Verify all Zotero tools are present
        for expected_tool in expected_zotero_tools:
            assert expected_tool in tool_names, f"Researcher missing Zotero tool: {expected_tool}"
        
        # Should ONLY have Zotero tools
        non_zotero_tools = [name for name in tool_names if 'zotero' not in name.lower()]
        assert len(non_zotero_tools) == 0, f"Researcher incorrectly assigned non-Zotero tools: {non_zotero_tools}"

    def test_archivist_gets_sequential_thinking_only(self):
        """Test archivist gets sequential thinking tools only."""
        registry = get_registry()
        tools = registry.get_agent_tools("archivist")
        
        # Should have exactly 1 sequential thinking tool
        assert len(tools) == 1, f"Archivist should have 1 tool, got {len(tools)}"
        
        tool_names = [tool.name for tool in tools]
        assert 'sequentialthinking' in tool_names, "Archivist missing sequential thinking tool"
        
        # Should NOT have other tools
        non_sequential_tools = [name for name in tool_names if 'sequential' not in name.lower()]
        assert len(non_sequential_tools) == 0, f"Archivist incorrectly assigned non-sequential tools: {non_sequential_tools}"

    def test_publisher_gets_filesystem_tools_only(self):
        """Test publisher gets filesystem tools and no memory/Zotero tools."""
        registry = get_registry()
        tools = registry.get_agent_tools("publisher")
        
        # Should have 11 filesystem tools
        assert len(tools) == 11, f"Publisher should have 11 tools, got {len(tools)}"
        
        tool_names = [tool.name for tool in tools]
        expected_filesystem_tools = [
            'read_file', 'read_multiple_files', 'write_file', 'edit_file',
            'move_file', 'search_files', 'get_file_info', 'create_directory',
            'list_directory', 'directory_tree', 'list_allowed_directories'
        ]
        
        # Verify all filesystem tools are present
        for expected_tool in expected_filesystem_tools:
            assert expected_tool in tool_names, f"Publisher missing filesystem tool: {expected_tool}"
        
        # Should NOT have memory tools
        memory_tools = [name for name in tool_names 
                       if any(pattern in name.lower() for pattern in ['entities', 'relations', 'observations'])]
        assert len(memory_tools) == 0, f"Publisher incorrectly assigned memory tools: {memory_tools}"
        
        # Should NOT have Zotero tools
        zotero_tools = [name for name in tool_names if 'zotero' in name.lower()]
        assert len(zotero_tools) == 0, f"Publisher incorrectly assigned Zotero tools: {zotero_tools}"

    def test_no_tool_cross_contamination(self):
        """Test that tools are not duplicated across agents inappropriately."""
        registry = get_registry()
        
        # Get all agent tools
        historian_tools = [tool.name for tool in registry.get_agent_tools("historian")]
        researcher_tools = [tool.name for tool in registry.get_agent_tools("researcher")]
        archivist_tools = [tool.name for tool in registry.get_agent_tools("archivist")]
        publisher_tools = [tool.name for tool in registry.get_agent_tools("publisher")]
        
        # Zotero tools should ONLY be in researcher
        zotero_tools = [name for name in researcher_tools if 'zotero' in name.lower()]
        assert len(zotero_tools) == 3, "Researcher should have exactly 3 Zotero tools"
        
        # No other agent should have Zotero tools
        other_agents_zotero = []
        for agent_name, tools in [("historian", historian_tools), ("archivist", archivist_tools), ("publisher", publisher_tools)]:
            agent_zotero = [name for name in tools if 'zotero' in name.lower()]
            if agent_zotero:
                other_agents_zotero.extend([(agent_name, tool) for tool in agent_zotero])
        
        assert len(other_agents_zotero) == 0, f"Non-researcher agents have Zotero tools: {other_agents_zotero}"
        
        # Memory-specific tools should ONLY be in historian
        memory_patterns = ['entities', 'relations', 'observations', 'graph', 'nodes']
        memory_tools = [name for name in historian_tools 
                       if any(pattern in name.lower() for pattern in memory_patterns)]
        assert len(memory_tools) >= 5, "Historian should have at least 5 memory-specific tools"
        
        # Sequential thinking should ONLY be in archivist
        sequential_tools = [name for name in archivist_tools if 'sequential' in name.lower()]
        assert len(sequential_tools) == 1, "Archivist should have exactly 1 sequential tool"

    def test_tool_pattern_mapping_correctness(self):
        """Test that tool pattern mappings work as expected."""
        registry = get_registry()
        
        # Test that patterns are correctly configured
        mappings = {mapping.agent_name: mapping.tool_patterns for mapping in registry.mappings}
        
        # Verify correct patterns
        assert "historian" in mappings
        assert "researcher" in mappings
        assert "archivist" in mappings
        assert "publisher" in mappings
        
        # Historian should use memory-specific patterns
        historian_patterns = mappings["historian"]
        expected_patterns = ["entities", "relations", "observations", "graph", "nodes"]
        assert historian_patterns == expected_patterns, f"Historian patterns incorrect: {historian_patterns}"
        
        # Researcher should use Zotero pattern
        researcher_patterns = mappings["researcher"]
        assert researcher_patterns == ["zotero"], f"Researcher patterns incorrect: {researcher_patterns}"
        
        # Archivist should use sequential pattern
        archivist_patterns = mappings["archivist"]
        assert archivist_patterns == ["sequential"], f"Archivist patterns incorrect: {archivist_patterns}"
        
        # Publisher should use filesystem patterns
        publisher_patterns = mappings["publisher"]
        expected_patterns = ["file", "directory", "write", "edit", "move", "list_"]
        assert publisher_patterns == expected_patterns, f"Publisher patterns incorrect: {publisher_patterns}"

    def test_all_mcp_servers_registered(self):
        """Test that all required MCP servers are registered."""
        registry = get_registry()
        
        expected_servers = ["memory", "zotero", "filesystem", "sequential_thinking"]
        registered_servers = list(registry.servers.keys())
        
        for server in expected_servers:
            assert server in registered_servers, f"Required MCP server not registered: {server}"
        
        # All servers should be enabled
        for server_name, config in registry.servers.items():
            assert config.enabled, f"MCP server {server_name} is disabled"

    def test_agent_tool_count_requirements(self):
        """Test that agents meet their minimum tool count requirements."""
        registry = get_registry()
        
        # Test minimum requirements
        requirements = {
            "historian": 6,
            "researcher": 3,
            "archivist": 1,
            "publisher": 11
        }
        
        for agent_name, min_count in requirements.items():
            tools = registry.get_agent_tools(agent_name)
            actual_count = len(tools)
            assert actual_count >= min_count, \
                f"{agent_name} has {actual_count} tools, needs at least {min_count}" 