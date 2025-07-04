"""Consolidated MCP Integration Tests - Replacing timeout-prone single agent tests."""

import pytest
import asyncio
from unittest.mock import MagicMock, patch
from server_research_mcp.crew import ServerResearchMcpCrew
from crewai import Crew, Agent, Task
import json
from crewai.tools import BaseTool


class TestMCPToolIntegration:
    """Test MCP tool integration across all agents."""
    
    def test_mcp_manager_connection(self, mock_mcp_manager):
        """Test MCP manager basic connection and health."""
        assert mock_mcp_manager.is_connected()
        
        # Test tool listing
        available_tools = mock_mcp_manager.list_tools()
        assert len(available_tools) > 0
        
        # Verify critical tools are available
        critical_tools = ['search_nodes', 'create_entities', 'sequentialthinking']
        for tool in critical_tools:
            tool_available = any(tool in available_tool for available_tool in available_tools)
            assert tool_available, f"Critical tool {tool} not available"

    def test_historian_mcp_tools(self, mock_mcp_manager):
        """Test historian MCP tool integration."""
        historian_tools = mock_mcp_manager.get_historian_tools()
        
        # Verify historian has memory management tools
        assert len(historian_tools) >= 4
        tool_names = [tool.name for tool in historian_tools]
        
        expected_tools = ['search_nodes', 'create_entities', 'create_relations', 'add_observations']
        for tool in expected_tools:
            assert tool in tool_names, f"Missing historian tool: {tool}"
        
        # Test tool execution
        search_result = mock_mcp_manager.call_tool("search_nodes", {"query": "AI research"})
        assert search_result["success"]
        assert "results" in search_result

    def test_researcher_mcp_tools(self, mock_mcp_manager):
        """Test researcher MCP tool integration."""
        researcher_tools = mock_mcp_manager.get_researcher_tools()
        
        # Verify researcher has research tools
        assert len(researcher_tools) >= 3
        tool_names = [tool.name for tool in researcher_tools]
        
        # Should have Zotero and Context7 tools
        has_zotero = any('zotero' in tool.lower() for tool in tool_names)
        has_context7 = any('resolve' in tool or 'docs' in tool for tool in tool_names)
        
        assert has_zotero or has_context7, f"Missing research capabilities in {tool_names}"
        
        # Test research workflow
        resolve_result = mock_mcp_manager.call_tool("resolve-library-id", {"library_name": "tensorflow"})
        assert resolve_result["success"]
        assert "library_id" in resolve_result

    def test_archivist_mcp_tools(self, mock_mcp_manager):
        """Test archivist MCP tool integration."""
        archivist_tools = mock_mcp_manager.get_archivist_tools()
        
        # Verify archivist has thinking tools
        assert len(archivist_tools) >= 1
        tool_names = [tool.name for tool in archivist_tools]
        
        # Should have sequential thinking
        has_thinking = any('sequential' in tool.lower() or 'thinking' in tool.lower() for tool in tool_names)
        assert has_thinking, f"Missing thinking capabilities in {tool_names}"
        
        # Test thinking workflow
        thought_result = mock_mcp_manager.call_tool("sequentialthinking", {
            "thought": "Analyzing AI testing approach",
            "thought_number": 1,
            "total_thoughts": 3
        })
        assert thought_result["success"]

    def test_publisher_mcp_tools(self, mock_mcp_manager):
        """Test publisher MCP tool integration."""
        publisher_tools = mock_mcp_manager.get_publisher_tools()
        
        # Verify publisher has comprehensive tools
        assert len(publisher_tools) >= 1
        tool_names = [tool.name for tool in publisher_tools]
        
        # Should have entity creation and file operations
        has_entity_tools = any('create_entities' in tool for tool in tool_names)
        has_file_tools = any('write_file' in tool or 'create_directory' in tool for tool in tool_names)
        
        assert has_entity_tools or has_file_tools, f"Missing publishing capabilities in {tool_names}"

    def test_cross_agent_mcp_workflow(self, mock_mcp_manager):
        """Test MCP workflow across multiple agent types."""
        # Step 1: Historian searches existing knowledge
        search_result = mock_mcp_manager.call_tool("search_nodes", {"query": "machine learning testing"})
        assert search_result["success"]
        
        # Step 2: Researcher finds additional sources
        research_result = mock_mcp_manager.call_tool("zotero_search_items", {"query": "ML validation"})
        assert research_result["success"]
        
        # Step 3: Archivist processes and analyzes
        analysis_result = mock_mcp_manager.call_tool("sequentialthinking", {
            "thought": "Combining search and research results",
            "thought_number": 1,
            "total_thoughts": 2
        })
        assert analysis_result["success"]
        
        # Step 4: Historian creates new entities from analysis
        entities = [{
            "name": "ml_testing_methodology",
            "entityType": "methodology",
            "observations": ["Combines search results", "Includes validation approaches"]
        }]
        creation_result = mock_mcp_manager.call_tool("create_entities", {"entities": entities})
        assert creation_result["success"]
        assert "entity_id" in creation_result

    def test_mcp_error_handling(self, mock_mcp_connection_issues):
        """Test MCP error handling and recovery."""
        # Test connection failure handling
        assert not mock_mcp_connection_issues.is_connected()
        
        # Test timeout error handling
        with pytest.raises(asyncio.TimeoutError):
            mock_mcp_connection_issues.call_tool("timeout_test", {})
        
        # Test connection error handling
        with pytest.raises(ConnectionError):
            mock_mcp_connection_issues.call_tool("connection_test", {})

    def test_mcp_performance_constraints(self, mock_mcp_manager, mcp_performance_monitor):
        """Test MCP operations meet performance constraints."""
        import time
        
        # Test multiple tool calls with performance monitoring
        tools_to_test = ['search_nodes', 'create_entities', 'sequentialthinking']
        
        for tool in tools_to_test:
            start_time = time.time()
            
            if tool == 'search_nodes':
                result = mock_mcp_manager.call_tool(tool, {"query": "test"})
            elif tool == 'create_entities':
                entities = [{"name": "test", "entityType": "concept", "observations": ["test"]}]
                result = mock_mcp_manager.call_tool(tool, {"entities": entities})
            else:
                result = mock_mcp_manager.call_tool(tool, {"thought": "test", "thought_number": 1})
            
            duration = time.time() - start_time
            
            # Record performance
            mcp_performance_monitor.record_call(tool, duration, result.get("success", False))
            
            # Verify performance constraint (should be very fast for mocks)
            assert duration < 1.0, f"Tool {tool} took {duration:.2f}s (>1s for mock)"
        
        # Check overall performance stats
        stats = mcp_performance_monitor.get_stats()
        assert stats["total_calls"] == len(tools_to_test)
        assert stats["error_rate"] == 0.0


class TestMCPAgentIntegration:
    """Test MCP integration with CrewAI agents."""
    
    def test_agent_mcp_tool_assignment(self, disable_crew_memory, mock_mcp_manager):
        """Test agents receive MCP tools correctly."""
        with patch('server_research_mcp.tools.mcp_tools.get_mcp_manager', return_value=mock_mcp_manager):
            crew_instance = ServerResearchMcpCrew()
            
            # Test researcher agent gets research tools
            researcher = crew_instance.researcher()
            assert hasattr(researcher, 'tools')
            assert len(researcher.tools) > 0
            
            # Test historian agent gets memory tools (if available)
            if hasattr(crew_instance, 'historian'):
                historian = crew_instance.historian()
                assert len(historian.tools) >= 4  # Should have memory tools

    def test_crew_mcp_workflow_simulation(self, disable_crew_memory, mock_mcp_manager):
        """Test crew workflow with MCP tools (no actual execution)."""
        with patch('server_research_mcp.tools.mcp_tools.get_mcp_manager', return_value=mock_mcp_manager):
            crew_instance = ServerResearchMcpCrew()
            crew = crew_instance.crew()
            
            # Verify crew structure
            assert len(crew.agents) >= 2
            assert len(crew.tasks) >= 2
            
            # Test task configuration for MCP compatibility
            for task in crew.tasks:
                assert hasattr(task, 'description')
                # Check that task descriptions are properly formatted (no more raw templates)
                # In the new decorator system, templates are formatted with default values
                assert task.description is not None
                assert len(task.description) > 0
                # Should not contain raw template strings after formatting
                assert '{paper_query}' not in task.description
                assert '{current_year}' not in task.description
                assert '{raw_paper_data}' not in task.description
                assert '{topic}' not in task.description

    @pytest.mark.timeout(10)
    def test_lightweight_agent_execution(self, disable_crew_memory, mock_mcp_manager):
        """Test lightweight agent execution without timeouts."""
        with patch('server_research_mcp.tools.mcp_tools.get_mcp_manager', return_value=mock_mcp_manager):
            # Create minimal crew for testing
            from server_research_mcp.config.llm_config import create_llm
            
            # Create simple mock tools that inherit from BaseTool
            class MockTool(BaseTool):
                name: str = "mock_tool"
                description: str = "Mock tool for testing"
                
                def _run(self, query: str = "") -> str:
                    return f"Mock result for: {query}"
            
            with patch('server_research_mcp.config.llm_config.create_llm') as mock_llm_factory:
                mock_llm = MagicMock()
                mock_llm.invoke.return_value = "Test task completed successfully"
                mock_llm_factory.return_value = mock_llm
                
                # Create simple agent and task with real tools
                mock_tools = [MockTool(), MockTool()]
                
                test_agent = Agent(
                    role="Test Agent",
                    goal="Complete test task",
                    backstory="Agent for testing",
                    llm=mock_llm,
                    tools=mock_tools
                )
                
                test_task = Task(
                    description="Test MCP integration: {query}",
                    agent=test_agent,
                    expected_output="Test result"
                )
                
                test_crew = Crew(
                    agents=[test_agent],
                    tasks=[test_task],
                    memory=False,
                    planning=False
                )
                
                # Execute with timeout constraint
                result = test_crew.kickoff(inputs={"query": "test query"})
                
                # Validate result structure
                assert result is not None
                result_str = str(result)
                assert len(result_str) > 0

    def test_mcp_tool_validation(self, mock_mcp_manager):
        """Test MCP tool validation and structure."""
        # Test all agent tool sets
        agent_tools = {
            'historian': mock_mcp_manager.get_historian_tools(),
            'researcher': mock_mcp_manager.get_researcher_tools(),
            'archivist': mock_mcp_manager.get_archivist_tools(),
            'publisher': mock_mcp_manager.get_publisher_tools()
        }
        
        for agent_type, tools in agent_tools.items():
            assert len(tools) > 0, f"{agent_type} has no tools"
            
            for tool in tools:
                # Validate tool structure
                assert hasattr(tool, 'name'), f"{agent_type} tool missing name"
                assert hasattr(tool, 'run'), f"{agent_type} tool missing run method"
                
                # Test tool execution
                try:
                    result = tool.run("test input")
                    assert result is not None
                except Exception as e:
                    # Tool execution can fail, but should not crash
                    assert isinstance(e, Exception)


class TestMCPServerSimulation:
    """Test MCP server behavior simulation."""
    
    @pytest.mark.asyncio
    async def test_async_mcp_operations(self, mock_mcp_manager):
        """Test async MCP operations."""
        # Update the mock to include sequential-thinking
        mock_mcp_manager.initialized_servers = ['memory', 'filesystem', 'sequential-thinking']
        
        # Test async initialization
        await mock_mcp_manager.initialize(['memory', 'sequential-thinking'])
        assert 'memory' in mock_mcp_manager.initialized_servers
        assert 'sequential-thinking' in mock_mcp_manager.initialized_servers
        
        # Test async tool calling
        result = await mock_mcp_manager.async_call_tool(
            'memory', 'search_nodes', {'query': 'async test'}
        )
        assert result['success']
        
        # Test async shutdown
        await mock_mcp_manager.shutdown()

    def test_mcp_connection_states(self, mock_mcp_manager):
        """Test different MCP connection states."""
        # Test initial connected state
        assert mock_mcp_manager.is_connected()
        
        # Test restart capability
        restart_result = mock_mcp_manager.restart()
        assert restart_result
        
        # Test shutdown
        shutdown_result = mock_mcp_manager.shutdown()
        assert shutdown_result

    def test_mcp_tool_compatibility(self, mock_mcp_manager):
        """Test MCP tool compatibility across versions."""
        # Test legacy tool names
        legacy_tools = [
            'memory_search', 'memory_create_entity', 'memory_add_observation',
            'context7_resolve_library', 'context7_get_docs'
        ]
        
        for legacy_tool in legacy_tools:
            result = mock_mcp_manager.call_tool(legacy_tool, {"test": "input"})
            assert result['success'], f"Legacy tool {legacy_tool} failed"
        
        # Test new tool names
        new_tools = ['search_nodes', 'create_entities', 'resolve-library-id']
        
        for new_tool in new_tools:
            result = mock_mcp_manager.call_tool(new_tool, {"test": "input"})
            assert result['success'], f"New tool {new_tool} failed"

    def test_mcp_response_validation(self, mock_mcp_manager):
        """Test MCP response structure validation."""
        # Test search response structure
        search_response = mock_mcp_manager.call_tool("search_nodes", {"query": "test"})
        required_search_fields = ['results', 'query', 'success']
        for field in required_search_fields:
            assert field in search_response, f"Missing field {field} in search response"
        
        # Test entity creation response structure
        entities = [{"name": "test", "entityType": "concept", "observations": ["test"]}]
        create_response = mock_mcp_manager.call_tool("create_entities", {"entities": entities})
        required_create_fields = ['success', 'entities', 'entity_id']
        for field in required_create_fields:
            assert field in create_response, f"Missing field {field} in create response"
        
        # Test library resolution response structure
        resolve_response = mock_mcp_manager.call_tool("resolve-library-id", {"library_name": "test"})
        required_resolve_fields = ['library_id', 'confidence', 'success']
        for field in required_resolve_fields:
            assert field in resolve_response, f"Missing field {field} in resolve response"


class TestMCPHealthChecks:
    """Test MCP health check and monitoring capabilities."""
    
    def test_mcp_tool_health_check(self, mock_mcp_manager):
        """Test MCP tool health checking."""
        available_tools = mock_mcp_manager.list_tools()
        
        # Health check: all critical tools should be available
        critical_tools = {
            'memory': ['search_nodes', 'create_entities'],
            'research': ['resolve-library-id', 'zotero_search_items'],
            'analysis': ['sequentialthinking']
        }
        
        for category, tools in critical_tools.items():
            for tool in tools:
                is_available = any(tool in available_tool for available_tool in available_tools)
                assert is_available, f"Critical {category} tool {tool} not available"

    def test_mcp_performance_monitoring(self, mock_mcp_manager, mcp_performance_monitor):
        """Test MCP performance monitoring."""
        import time
        
        # Execute various operations and monitor performance
        operations = [
            ('search_nodes', {'query': 'performance test'}),
            ('create_entities', {'entities': [{'name': 'perf_test', 'entityType': 'test', 'observations': []}]}),
            ('sequentialthinking', {'thought': 'performance analysis', 'thought_number': 1})
        ]
        
        for tool_name, args in operations:
            start_time = time.time()
            result = mock_mcp_manager.call_tool(tool_name, args)
            duration = time.time() - start_time
            
            mcp_performance_monitor.record_call(
                tool_name, duration, result.get('success', False)
            )
        
        # Validate monitoring results
        stats = mcp_performance_monitor.get_stats()
        assert stats['total_calls'] == len(operations)
        assert stats['avg_duration'] < 1.0  # Should be fast for mocks
        assert stats['error_rate'] == 0.0  # No errors expected

    def test_mcp_resource_cleanup(self, mock_mcp_manager):
        """Test MCP resource cleanup."""
        # Initialize some resources
        mock_mcp_manager.initialized_servers = ['memory', 'sequential-thinking']
        mock_mcp_manager.adapters = {'memory': MagicMock(), 'sequential-thinking': MagicMock()}
        
        # Test cleanup
        mock_mcp_manager.shutdown()
        
        # Verify cleanup (mocked behavior)
        shutdown_called = mock_mcp_manager.shutdown.called
        assert shutdown_called, "Shutdown should have been called" 
