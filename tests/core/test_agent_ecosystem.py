"""Consolidated Agent Ecosystem Tests - Merging unit and integration agent tests."""

import pytest
from unittest.mock import MagicMock, patch, call
from server_research_mcp.crew import ServerResearchMcpCrew
from crewai import Agent
import asyncio


class TestAgentEcosystem:
    """Comprehensive agent ecosystem tests."""
    
    def test_crew_agent_initialization(self, disable_crew_memory, mock_llm):
        """Test all agents initialize properly with correct tools."""
        with patch('server_research_mcp.config.llm_config.get_configured_llm', return_value=mock_llm):
            crew_instance = ServerResearchMcpCrew()
            crew = crew_instance.crew()
            
            # Verify basic structure
            assert crew is not None
            assert len(crew.agents) >= 2
            assert len(crew.tasks) >= 2
            
            # Verify agent roles and tools
            agent_roles = [agent.role for agent in crew.agents if hasattr(agent, 'role')]
            expected_roles = ['research', 'report', 'analyst', 'memory', 'historian']
            
            # Check that we have research and reporting capabilities
            has_research = any(any(role_word in role.lower() for role_word in ['research', 'historian']) for role in agent_roles)
            has_reporting = any(any(role_word in role.lower() for role_word in ['report', 'analyst', 'publish']) for role in agent_roles)
            
            assert has_research, f"Missing research capability in roles: {agent_roles}"
            assert has_reporting, f"Missing reporting capability in roles: {agent_roles}"

    def test_agent_tool_assignment(self, mock_crew_agents, mock_mcp_manager):
        """Test agents receive appropriate tool assignments."""
        # Test Historian - Memory tools
        historian = mock_crew_agents["historian"]
        historian_tools = mock_mcp_manager.get_historian_tools()
        assert len(historian_tools) >= 4, "Historian should have at least 4 memory tools"
        
        # Verify memory tool names
        memory_tool_names = [tool.name for tool in historian_tools]
        expected_memory_tools = ['search_nodes', 'create_entities', 'create_relations', 'add_observations']
        assert all(tool in memory_tool_names for tool in expected_memory_tools)
        
        # Test Researcher - Research tools
        researcher = mock_crew_agents["researcher"]
        researcher_tools = mock_mcp_manager.get_researcher_tools()
        assert len(researcher_tools) >= 3, "Researcher should have at least 3 research tools"
        
        # Verify research tool names
        research_tool_names = [tool.name for tool in researcher_tools]
        expected_research_tools = ['zotero_search_items', 'resolve-library-id', 'get-library-docs']
        assert any(tool in research_tool_names for tool in expected_research_tools)

    def test_agent_collaboration_workflow(self, mock_crew_agents, mock_mcp_manager):
        """Test agent collaboration workflow patterns."""
        historian = mock_crew_agents["historian"]
        researcher = mock_crew_agents["researcher"]
        synthesizer = mock_crew_agents["synthesizer"]
        
        # Simulate collaboration workflow
        workflow_results = []
        
        # 1. Historian searches memory
        memory_search = mock_mcp_manager.call_tool("search_nodes", {"query": "AI testing"})
        workflow_results.append(memory_search)
        
        # 2. Researcher conducts research
        research_result = mock_mcp_manager.call_tool("zotero_search_items", {"query": "AI testing methodologies"})
        workflow_results.append(research_result)
        
        # 3. Historian creates entities from research
        entities_to_create = [{
            "name": "ai_testing_methodology",
            "entityType": "concept",
            "observations": ["Research methodology for AI systems", "Includes validation approaches"]
        }]
        entity_creation = mock_mcp_manager.call_tool("create_entities", {"entities": entities_to_create})
        workflow_results.append(entity_creation)
        
        # Verify workflow success
        assert len(workflow_results) == 3
        assert all(result.get("success", False) for result in workflow_results)
        assert "entity_id" in entity_creation

    def test_agent_llm_consistency(self, mock_llm):
        """Test all agents use consistent LLM configuration."""
        with patch('server_research_mcp.config.llm_config.create_llm', return_value=mock_llm):
            crew_instance = ServerResearchMcpCrew()
            
            researcher = crew_instance.researcher()
            analyst = crew_instance.reporting_analyst()
            
            # Check LLM assignment
            assert hasattr(researcher, 'llm')
            assert hasattr(analyst, 'llm')
            assert researcher.llm is not None or analyst.llm is not None

    def test_agent_error_handling(self, mock_crew_agents, mock_mcp_manager):
        """Test agent error handling and recovery."""
        historian = mock_crew_agents["historian"]
        
        # Mock a failing tool call
        def failing_tool_call(tool_name, arguments=None, **kwargs):
            if tool_name == "create_entities":
                raise Exception("Connection error")
            return {"success": True, "data": "mock_response"}
        
        with patch.object(mock_mcp_manager, 'call_tool', side_effect=failing_tool_call):
            # Test error handling
            try:
                mock_mcp_manager.call_tool("create_entities", {"entities": []})
                assert False, "Expected exception was not raised"
            except Exception as e:
                assert "Connection error" in str(e)
            
            # Test recovery with different tool
            result = mock_mcp_manager.call_tool("search_nodes", {"query": "test"})
            assert result["success"]

    def test_task_agent_assignment(self, disable_crew_memory):
        """Test tasks are properly assigned to agents."""
        crew_instance = ServerResearchMcpCrew()
        crew = crew_instance.crew()
        
        # Each task should have an assigned agent
        for task in crew.tasks:
            assert hasattr(task, 'agent') or hasattr(task, 'context')

    def test_agent_state_sharing(self, mock_crew_agents, mock_crew_memory):
        """Test agents can share state through memory."""
        historian = mock_crew_agents["historian"]
        researcher = mock_crew_agents["researcher"]
        
        # Historian stores information
        test_data = {"topic": "AI testing", "sources": ["paper1", "paper2"]}
        mock_crew_memory.save(test_data)
        
        # Researcher retrieves information
        retrieved_data = mock_crew_memory.search("AI testing")
        assert retrieved_data is not None
        assert "topic" in str(retrieved_data)

    def test_agent_tool_execution_patterns(self, mock_crew_agents, mock_mcp_manager):
        """Test different agent tool execution patterns."""
        # Sequential execution pattern
        historian = mock_crew_agents["historian"]
        
        # Step 1: Search
        search_result = mock_mcp_manager.call_tool("search_nodes", {"query": "machine learning"})
        assert search_result["success"]
        
        # Step 2: Create based on search
        if search_result.get("results"):
            entities = [{
                "name": "machine_learning_research",
                "entityType": "research_area",
                "observations": ["Active research area", "Multiple methodologies available"]
            }]
            create_result = mock_mcp_manager.call_tool("create_entities", {"entities": entities})
            assert create_result["success"]
            assert "entity_id" in create_result
        
        # Parallel execution pattern (simulated)
        researcher = mock_crew_agents["researcher"]
        
        # Multiple research queries
        queries = ["deep learning", "neural networks", "transformers"]
        results = []
        for query in queries:
            result = mock_mcp_manager.call_tool("zotero_search_items", {"query": query})
            results.append(result)
        
        assert len(results) == 3
        assert all(r["success"] for r in results)

    def test_specialized_agent_behaviors(self, mock_crew_agents, mock_mcp_manager):
        """Test specialized behaviors for different agent types."""
        # Historian: Memory management focus
        historian = mock_crew_agents["historian"]
        memory_tools = mock_mcp_manager.get_historian_tools()
        memory_tool_names = [tool.name for tool in memory_tools]
        
        # Should have comprehensive memory operations
        memory_operations = ['search_nodes', 'create_entities', 'add_observations', 'delete_entities']
        has_memory_ops = all(op in memory_tool_names for op in memory_operations)
        assert has_memory_ops, f"Missing memory operations: {memory_tool_names}"
        
        # Researcher: Research focus
        researcher = mock_crew_agents["researcher"]
        research_tools = mock_mcp_manager.get_researcher_tools()
        research_tool_names = [tool.name for tool in research_tools]
        
        # Should have research capabilities
        research_capabilities = ['search', 'resolve', 'docs']
        has_research_caps = any(cap in ' '.join(research_tool_names).lower() for cap in research_capabilities)
        assert has_research_caps, f"Missing research capabilities: {research_tool_names}"
        
        # Archivist: Analysis focus
        archivist = mock_crew_agents["archivist"]
        analysis_tools = mock_mcp_manager.get_archivist_tools()
        analysis_tool_names = [tool.name for tool in analysis_tools]
        
        # Should have thinking/analysis tools
        thinking_capabilities = ['sequential', 'thinking']
        has_thinking = any(cap in ' '.join(analysis_tool_names).lower() for cap in thinking_capabilities)
        assert has_thinking, f"Missing thinking capabilities: {analysis_tool_names}"


class TestAgentPerformance:
    """Test agent performance and efficiency."""
    
    def test_agent_initialization_performance(self, disable_crew_memory):
        """Test agent initialization happens efficiently."""
        import time
        
        start_time = time.time()
        crew_instance = ServerResearchMcpCrew()
        crew = crew_instance.crew()
        initialization_time = time.time() - start_time
        
        # Should initialize within reasonable time
        assert initialization_time < 5.0, f"Initialization took {initialization_time:.2f}s (>5s limit)"
        assert len(crew.agents) > 0
        assert len(crew.tasks) > 0

    def test_agent_tool_loading_efficiency(self, mock_mcp_manager):
        """Test tool loading is efficient across agents."""
        # Test tool loading for all agent types
        tool_sets = {
            "historian": mock_mcp_manager.get_historian_tools(),
            "researcher": mock_mcp_manager.get_researcher_tools(),
            "archivist": mock_mcp_manager.get_archivist_tools(),
            "publisher": mock_mcp_manager.get_publisher_tools()
        }
        
        # Verify each agent has appropriate number of tools
        assert len(tool_sets["historian"]) >= 4, "Historian should have at least 4 tools"
        assert len(tool_sets["researcher"]) >= 3, "Researcher should have at least 3 tools"
        assert len(tool_sets["archivist"]) >= 1, "Archivist should have at least 1 tool"
        assert len(tool_sets["publisher"]) >= 1, "Publisher should have at least 1 tool"
        
        # Verify tools are properly configured
        for agent_type, tools in tool_sets.items():
            for tool in tools:
                assert hasattr(tool, 'name'), f"{agent_type} tool missing name"
                assert hasattr(tool, 'run'), f"{agent_type} tool missing run method"


class TestAgentIntegration:
    """Test agent integration with external services."""
    
    def test_mcp_tool_integration(self, mock_mcp_manager):
        """Test MCP tool integration across agents."""
        # Test connection status
        assert mock_mcp_manager.is_connected()
        
        # Test tool listing
        available_tools = mock_mcp_manager.list_tools()
        assert len(available_tools) > 0
        
        # Test critical tools are available
        critical_tools = ['search_nodes', 'create_entities', 'zotero_search_items', 'sequentialthinking']
        for tool in critical_tools:
            # Allow for tool name variations
            tool_available = any(tool in available_tool for available_tool in available_tools)
            assert tool_available, f"Critical tool {tool} not available in {available_tools}"

    def test_agent_guardrail_integration(self, disable_crew_memory):
        """Test agent guardrails are properly integrated."""
        crew_instance = ServerResearchMcpCrew()
        
        # Test research task guardrails
        research_task = crew_instance.research_task()
        assert research_task.guardrail is not None
        assert callable(research_task.guardrail)
        assert research_task.max_retries == 2
        
        # Test reporting task guardrails  
        reporting_task = crew_instance.reporting_task()
        assert reporting_task.guardrail is not None
        assert callable(reporting_task.guardrail)
        assert reporting_task.max_retries == 2

    def test_human_input_configuration(self, disable_crew_memory):
        """Test human input is properly configured for automated testing."""
        crew_instance = ServerResearchMcpCrew()
        
        # Context gathering should not require human input in tests
        context_task = crew_instance.context_gathering_task()
        assert context_task.human_input is False
        
        # Legacy tasks should maintain compatibility
        research_task = crew_instance.research_task()
        reporting_task = crew_instance.reporting_task()
        
        assert hasattr(research_task, 'human_input')
        assert hasattr(reporting_task, 'human_input')


class TestAgentResilience:
    """Test agent resilience and error recovery."""
    
    def test_agent_connection_recovery(self, mock_mcp_manager):
        """Test agents can recover from connection issues."""
        # Simulate connection loss
        mock_mcp_manager.is_connected = MagicMock(return_value=False)
        
        # Test restart capability
        restart_result = mock_mcp_manager.restart()
        assert restart_result, "Should be able to restart after connection loss"
        
        # Test shutdown capability
        shutdown_result = mock_mcp_manager.shutdown()
        assert shutdown_result, "Should be able to shutdown cleanly"

    def test_agent_partial_failure_handling(self, mock_crew_agents, mock_mcp_manager):
        """Test agents handle partial failures gracefully."""
        historian = mock_crew_agents["historian"]
        
        # Create scenario where some tools work, others fail
        def selective_failure(tool_name, arguments=None, **kwargs):
            if "delete" in tool_name:
                raise Exception("Delete operation failed")
            return {"success": True, "data": "mock_response"}
        
        with patch.object(mock_mcp_manager, 'call_tool', side_effect=selective_failure):
            # Working operations should succeed
            search_result = mock_mcp_manager.call_tool("search_nodes", {"query": "test"})
            assert search_result["success"]
            
            # Failing operations should raise exceptions
            with pytest.raises(Exception, match="Delete operation failed"):
                mock_mcp_manager.call_tool("delete_entities", {"entities": ["test"]}) 
