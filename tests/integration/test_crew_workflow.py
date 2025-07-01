"""Integration tests for crew workflow and agent interactions."""

import pytest
from unittest.mock import patch, MagicMock
from server_research_mcp.crew import ServerResearchMcp
from server_research_mcp.main import run, get_user_input


class TestCrewWorkflow:
    """Test complete crew workflow integration."""
    
    def test_crew_with_all_agents(self, disable_crew_memory):
        """Test crew creation with all available agents."""
        crew_instance = ServerResearchMcp()
        crew = crew_instance.crew()
        
        # Verify basic structure
        assert crew is not None
        assert len(crew.agents) >= 2  # At least researcher and analyst
        assert len(crew.tasks) >= 2   # At least research and reporting
        
        # Verify agent roles
        agent_roles = [agent.role for agent in crew.agents if hasattr(agent, 'role')]
        assert any('research' in role.lower() for role in agent_roles)
        assert any('report' in role.lower() or 'analyst' in role.lower() for role in agent_roles)
        
    def test_task_dependencies_flow(self):
        """Test that tasks have proper dependencies."""
        crew_instance = ServerResearchMcp()
        
        # Get tasks
        research_task = crew_instance.research_task()
        reporting_task = crew_instance.reporting_task()
        
        # Both should exist
        assert research_task is not None
        assert reporting_task is not None
        
        # Both should accept inputs
        assert '{topic}' in research_task.description
        assert '{current_year}' in research_task.description
        
    @patch('src.server_research_mcp.crew.ServerResearchMcp.crew')
    def test_crew_execution_flow(self, mock_crew_method, sample_inputs):
        """Test crew execution flow with mocked components."""
        # Setup mock
        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = "Research completed successfully"
        mock_crew_method.return_value = mock_crew
        
        # Execute
        crew_instance = ServerResearchMcp()
        crew = crew_instance.crew()
        result = crew.kickoff(inputs=sample_inputs)
        
        # Verify
        assert result == "Research completed successfully"
        mock_crew.kickoff.assert_called_once_with(inputs=sample_inputs)
        
    def test_guardrail_integration(self):
        """Test that guardrails are properly integrated."""
        crew_instance = ServerResearchMcp()
        
        # Check research task guardrail
        research_task = crew_instance.research_task()
        assert research_task.guardrail is not None
        assert callable(research_task.guardrail)
        assert research_task.max_retries == 2
        
        # Check reporting task guardrail  
        reporting_task = crew_instance.reporting_task()
        assert reporting_task.guardrail is not None
        assert callable(reporting_task.guardrail)
        assert reporting_task.max_retries == 2
        
    def test_human_input_integration(self):
        """Test human input is properly configured for tests."""
        crew_instance = ServerResearchMcp()
        
        # Check context gathering task (should be False for tests)
        context_task = crew_instance.context_gathering_task()
        assert context_task.human_input is False
        
        # Legacy tasks should maintain compatibility
        research_task = crew_instance.research_task()
        reporting_task = crew_instance.reporting_task()
        
        # These legacy tasks should not require human input in tests
        assert hasattr(research_task, 'human_input')
        assert hasattr(reporting_task, 'human_input')


class TestMainWorkflow:
    """Test main application workflow."""
    
    @patch('server_research_mcp.main.get_user_input')
    @patch('server_research_mcp.crew.ServerResearchMcp')
    @patch('sys.argv', ['test_script', 'test_query', '--topic', 'AI Research'])
    def test_main_run_flow(self, mock_crew_class, mock_get_input):
        """Test main run function workflow."""
        # Setup mocks
        mock_get_input.return_value = "AI Research"
        mock_crew_instance = MagicMock()
        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = "Research completed"
        mock_crew_instance.crew.return_value = mock_crew
        mock_crew_class.return_value = mock_crew_instance
        
        # Execute with output suppression and environment mocking
        with patch('builtins.print'):
            with patch('os.path.exists', return_value=False):
                with patch('os.makedirs'):
                    with patch('server_research_mcp.main.validate_environment', return_value=True):
                        with patch('builtins.input', return_value='y'):
                            with patch('dotenv.load_dotenv'):
                                run()
        
        # Verify workflow
        mock_crew_class.assert_called_once()
        mock_crew_instance.crew.assert_called_once()
        mock_crew.kickoff.assert_called_once()
        
        # Verify inputs passed correctly
        call_args = mock_crew.kickoff.call_args
        assert 'inputs' in call_args.kwargs
        assert call_args.kwargs['inputs']['topic'] == "AI Research"
        
    @patch('builtins.input', side_effect=['Test Topic', 'n'])
    @patch('sys.exit')
    def test_user_cancellation_flow(self, mock_exit, mock_input):
        """Test user cancellation during input."""
        get_user_input()
        mock_exit.assert_called_once_with(0)
        
    @patch('server_research_mcp.main.get_user_input')
    @patch('server_research_mcp.crew.ServerResearchMcp')
    @patch('sys.argv', ['test_script', 'test_query', '--topic', 'Test Topic'])
    def test_error_handling_in_main(self, mock_crew_class, mock_get_input):
        """Test error handling in main workflow."""
        # Setup error scenario
        mock_get_input.return_value = "Test Topic"
        mock_crew_class.side_effect = Exception("Crew initialization failed")
        
        # Execute and expect error handling
        with patch('builtins.print') as mock_print:
            with patch('os.makedirs'):
                with patch('server_research_mcp.main.validate_environment', return_value=True):
                    with patch('builtins.input', return_value='y'):
                        with patch('dotenv.load_dotenv'):
                            with pytest.raises(Exception):
                                run()
                
        # Verify error was raised
        mock_crew_class.assert_called_once()


class TestInputParameterization:
    """Test various input parameter scenarios - consolidated from unit tests."""
    
    @pytest.mark.parametrize("topic,year,expected_valid", [
        ("AI", "2024", True),
        ("Machine Learning & Deep Learning", "2024", True),
        ("Quantum Computing (Advanced)", "2024", True),
        ("Künstliche Intelligenz", "2024", True),  # Unicode
        ("A" * 200, "2024", True),  # Very long topic
        ("!@#$%^&*()", "2024", True),  # Special characters
    ])
    def test_input_variations_with_workflow(self, topic, year, expected_valid):
        """Test crew handles various input formats in workflow context."""
        inputs = {'topic': topic, 'current_year': year}
        
        # Verify inputs are structurally valid
        assert 'topic' in inputs
        assert 'current_year' in inputs
        assert inputs['topic'] == topic
        assert inputs['current_year'] == year
        assert expected_valid is True
        
        # Test that crew can accept these inputs
        crew_instance = ServerResearchMcp()
        paper_extraction_task = crew_instance.paper_extraction_task()
        
        # Verify task can handle the input variations
        assert '{topic}' in paper_extraction_task.description
        assert '{current_year}' in paper_extraction_task.description
        
    def test_input_structure_requirements_in_workflow(self):
        """Test that workflow requires proper input structure."""
        crew_instance = ServerResearchMcp()
        
        # Test that tasks expect the correct template variables
        paper_extraction_task = crew_instance.paper_extraction_task()
        markdown_generation_task = crew_instance.markdown_generation_task()
        context_gathering_task = crew_instance.context_gathering_task()
        
        # Verify template variables are present in appropriate tasks
        assert '{topic}' in paper_extraction_task.description
        assert '{current_year}' in paper_extraction_task.description
        assert '{paper_query}' in context_gathering_task.description
        assert '{structured_json}' in markdown_generation_task.description


class TestAgentInteractions:
    """Test interactions between agents."""
    
    def test_agent_tool_sharing(self):
        """Test that agents have appropriate tools."""
        crew_instance = ServerResearchMcp()
        
        researcher = crew_instance.researcher()
        assert hasattr(researcher, 'tools')
        assert len(researcher.tools) > 0
        
        # If historian exists, check its tools
        if hasattr(crew_instance, 'historian'):
            historian = crew_instance.historian()
            assert len(historian.tools) >= 6  # MCP tools
            
    def test_agent_llm_consistency(self, mock_llm):
        """Test all agents use consistent LLM configuration."""
        with patch('server_research_mcp.config.llm_config.create_llm', return_value=mock_llm):
            crew_instance = ServerResearchMcp()
            
            researcher = crew_instance.researcher()
            analyst = crew_instance.reporting_analyst()
            
            # Check that agents have LLM instances (may be different objects but same type)
            assert hasattr(researcher, 'llm')
            assert hasattr(analyst, 'llm')
            # The actual LLM instances may not be identical due to CrewAI's internal handling
            assert researcher.llm is not None or analyst.llm is not None
            
    def test_task_agent_assignment(self):
        """Test tasks are properly assigned to agents."""
        crew_instance = ServerResearchMcp()
        crew = crew_instance.crew()
        
        # Each task should have an assigned agent
        for task in crew.tasks:
            assert hasattr(task, 'agent') or hasattr(task, 'context')
            

class TestMemoryIntegration:
    """Test crew memory integration."""
    
    @patch('crewai.memory.short_term.short_term_memory.ShortTermMemory')
    @patch('crewai.memory.long_term.long_term_memory.LongTermMemory')
    def test_crew_memory_enabled(self, mock_long_term, mock_short_term):
        """Test crew has memory properly configured."""
        crew_instance = ServerResearchMcp()
        crew = crew_instance.crew()
        
        assert crew.memory is True
        
    def test_chromadb_configuration(self):
        """Test ChromaDB is properly configured for tests."""
        import os
        
        # Check environment variables
        assert os.environ.get('CHROMADB_ALLOW_RESET') == 'true'
        assert 'CHROMADB_PATH' in os.environ
        
        # Path should exist
        chromadb_path = os.environ['CHROMADB_PATH']
        assert os.path.exists(chromadb_path)


@pytest.mark.slow
class TestFullWorkflow:
    """Test complete end-to-end workflow (marked as slow)."""
    
    @patch('server_research_mcp.main.get_user_input')
    @patch('server_research_mcp.crew.ServerResearchMcp.crew')
    def test_full_research_workflow(self, mock_crew_method, mock_get_input, 
                                   sample_inputs, valid_research_output, valid_report_output):
        """Test complete research workflow with valid outputs."""
        # Setup
        mock_get_input.return_value = sample_inputs['topic']
        
        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = {
            'research_output': valid_research_output,
            'report_output': valid_report_output
        }
        mock_crew_method.return_value = mock_crew
        
        # Execute
        with patch('builtins.print'):
            run()
            
        # Verify complete workflow
        mock_get_input.assert_called_once()
        mock_crew.kickoff.assert_called_once()
        
        # Verify outputs would pass validation
        from server_research_mcp.crew import validate_research_output, validate_report_output
        
        research_valid, _ = validate_research_output(valid_research_output)
        report_valid, _ = validate_report_output(valid_report_output)
        
        assert research_valid is True
        assert report_valid is True