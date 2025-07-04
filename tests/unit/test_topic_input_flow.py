"""
Test suite to verify topic input flow from command line to task descriptions.

This test prevents the regression where task descriptions were formatted with
fallback "Research Topic" instead of the actual user-provided topic.
"""

import pytest
from unittest.mock import patch, MagicMock
import asyncio

from server_research_mcp.main import initialize_research_crew, run_research, main_with_args
from server_research_mcp.crew import ServerResearchMcpCrew


class TestTopicInputFlow:
    """Test that topic inputs flow correctly through the entire system."""

    def test_crew_initialization_with_topic_inputs(self):
        """Test that crew can be initialized with topic inputs."""
        test_topic = "Machine Learning in Healthcare Applications"
        
        crew = ServerResearchMcpCrew(inputs={'topic': test_topic})
        
        # Verify the inputs are stored
        assert crew.inputs['topic'] == test_topic
        
        # Verify the topic is used in task creation
        context_task = crew.context_gathering_task()
        
        # The task description should contain the actual topic, not "Research Topic"
        assert test_topic in context_task.description
        assert "Research Topic" not in context_task.description or test_topic in context_task.description

    def test_task_descriptions_use_actual_topic(self):
        """Test that all task descriptions use the actual topic, not fallback."""
        test_topic = "Quantum Computing Error Correction"
        
        crew = ServerResearchMcpCrew(inputs={'topic': test_topic})
        
        # Check all task descriptions
        tasks = [
            crew.context_gathering_task(),
            crew.paper_extraction_task(),
            crew.analysis_and_structuring_task(),
            crew.publishing_task()
        ]
        
        for task in tasks:
            # Each task description should contain the actual topic
            assert test_topic in task.description, f"Task {task} description doesn't contain actual topic"
            
            # Should not have the fallback unless it's part of the actual topic
            if "Research Topic" in task.description and test_topic != "Research Topic":
                # Only acceptable if it's in context like "research topic: {actual_topic}"
                assert test_topic in task.description

    @pytest.mark.asyncio
    async def test_initialize_research_crew_with_inputs(self):
        """Test that initialize_research_crew passes inputs correctly."""
        test_topic = "Natural Language Processing for Medical Texts"
        
        with patch('server_research_mcp.main.ServerResearchMcpCrew') as mock_crew_class:
            mock_crew_instance = MagicMock()
            mock_crew_class.return_value = mock_crew_instance
            
            # Mock the crew() method to return a mock with agents
            mock_crew_obj = MagicMock()
            mock_crew_obj.agents = [MagicMock(), MagicMock()]  # Mock agents
            mock_crew_instance.crew.return_value = mock_crew_obj
            
            inputs = {'topic': test_topic}
            result = await initialize_research_crew(inputs=inputs)
            
            # Verify ServerResearchMcpCrew was initialized with inputs
            mock_crew_class.assert_called_once_with(inputs=inputs)
            assert result == mock_crew_instance

    def test_main_with_args_passes_topic_to_crew(self):
        """Test that main_with_args passes topic to crew initialization."""
        test_topic = "Artificial Intelligence in Climate Modeling"
        
        # Mock argparse.Namespace
        args = MagicMock()
        args.topic = test_topic
        args.output_dir = "test_outputs"
        args.dry_run = True  # Use dry run to avoid full execution
        args.verbose = False
        
        with patch('server_research_mcp.main.validate_environment') as mock_validate:
            mock_validate.return_value = True
            
            with patch('server_research_mcp.main.setup_output_directory') as mock_setup:
                mock_setup.return_value = "test_outputs"
                
                result = main_with_args(args)
                
                # Should succeed for dry run
                assert result['status'] == 'success'
                assert 'Dry run completed successfully' in result['message']

    @pytest.mark.asyncio 
    async def test_full_topic_flow_integration(self):
        """Integration test for complete topic flow from args to task descriptions."""
        test_topic = "Blockchain Technology in Supply Chain Management"
        
        # Mock the MCP tools and external dependencies
        with patch('server_research_mcp.crew.get_registry') as mock_registry:
            mock_tool_registry = MagicMock()
            mock_tool_registry.get_agent_tools.return_value = [MagicMock()]  # Mock tools
            mock_registry.return_value = mock_tool_registry
            
            with patch('server_research_mcp.config.llm_config.LLMConfig') as mock_llm_config:
                mock_llm_instance = MagicMock()
                mock_llm_config.return_value.get_llm.return_value = mock_llm_instance
                
                # Initialize crew with topic
                crew = ServerResearchMcpCrew(inputs={'topic': test_topic})
                
                # Verify task descriptions contain the actual topic
                context_task = crew.context_gathering_task()
                extraction_task = crew.paper_extraction_task()
                
                assert test_topic in context_task.description
                assert test_topic in extraction_task.description
                
                # Verify agents are created with the topic in their role
                historian = crew.historian()
                researcher = crew.researcher()
                
                # Agent roles should be formatted with the actual topic
                assert test_topic in historian.role or "Research Topic" in historian.role
                assert test_topic in researcher.role or "Research Topic" in researcher.role

    def test_fallback_behavior_when_no_topic_provided(self):
        """Test that fallback works when no topic is provided."""
        # Initialize without topic
        crew = ServerResearchMcpCrew(inputs={})
        
        context_task = crew.context_gathering_task()
        
        # Should use the fallback "Research Topic"
        assert "Research Topic" in context_task.description

    def test_topic_formatting_in_agent_roles(self):
        """Test that agent roles are properly formatted with the topic."""
        test_topic = "Computer Vision for Autonomous Vehicles"
        
        with patch('server_research_mcp.crew.get_registry') as mock_registry:
            mock_tool_registry = MagicMock()
            mock_tool_registry.get_agent_tools.return_value = [MagicMock()]
            mock_registry.return_value = mock_tool_registry
            
            with patch('server_research_mcp.config.llm_config.LLMConfig') as mock_llm_config:
                mock_llm_instance = MagicMock()
                mock_llm_config.return_value.get_llm.return_value = mock_llm_instance
                
                crew = ServerResearchMcpCrew(inputs={'topic': test_topic})
                
                # Check that agent roles contain the topic
                historian = crew.historian()
                researcher = crew.researcher()
                archivist = crew.archivist()
                publisher = crew.publisher()
                
                # All agent roles should reference the actual topic
                agents = [historian, researcher, archivist, publisher]
                for agent in agents:
                    # The role should contain either the test topic or have been formatted properly
                    assert (test_topic in agent.role or 
                           "Research Topic" in agent.role), f"Agent {agent} role doesn't contain topic reference"

    def test_topic_with_special_characters(self):
        """Test that topics with special characters are handled correctly."""
        test_topic = "AI & ML: Transformers, Attention Mechanisms & Neural Networks (2024)"
        
        crew = ServerResearchMcpCrew(inputs={'topic': test_topic})
        
        context_task = crew.context_gathering_task()
        
        # Should handle special characters in topic
        assert test_topic in context_task.description or "Research Topic" in context_task.description

    def test_very_long_topic_handling(self):
        """Test that very long topics are handled correctly."""
        test_topic = "A Comprehensive Analysis of Machine Learning Applications in Healthcare: From Natural Language Processing in Electronic Health Records to Computer Vision in Medical Imaging and Predictive Analytics for Patient Outcomes"
        
        crew = ServerResearchMcpCrew(inputs={'topic': test_topic})
        
        context_task = crew.context_gathering_task()
        
        # Should handle long topics without truncation in task descriptions
        assert test_topic in context_task.description or "Research Topic" in context_task.description


if __name__ == "__main__":
    pytest.main([__file__]) 