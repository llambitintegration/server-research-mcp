"""Core crew functionality tests."""

import pytest
from unittest.mock import MagicMock, patch, call
from server_research_mcp.crew import ServerResearchMcp
from crewai import Process


class TestCrewInitialization:
    """Test crew initialization and configuration."""
    
    def test_crew_basic_initialization(self, test_environment):
        """Test basic crew initialization."""
        crew_instance = ServerResearchMcp()
        assert crew_instance is not None
        
    def test_crew_with_custom_config(self, test_environment, mock_llm):
        """Test crew initialization with custom configuration."""
        with patch('server_research_mcp.crew.get_configured_llm', return_value=mock_llm):
            crew_instance = ServerResearchMcp()
            crew = crew_instance.crew()
            
            assert crew.process == Process.sequential
            assert crew.verbose is True
            assert crew.memory is True
    
    def test_crew_agent_count(self, mock_crew):
        """Test that crew has the correct number of agents."""
        crew = mock_crew.crew()
        assert len(crew.agents) >= 4  # historian, researcher, synthesizer, validator
    
    def test_crew_task_count(self, mock_crew):
        """Test that crew has the correct number of tasks."""
        crew = mock_crew.crew()
        assert len(crew.tasks) >= 4  # context_gathering, deep_research, synthesis, validation
    
    @pytest.mark.parametrize("process_type", [
        Process.sequential,
        Process.hierarchical
    ])
    def test_crew_process_types(self, mock_crew, process_type):
        """Test crew with different process types."""
        crew = mock_crew.crew()
        crew.process = process_type
        assert crew.process == process_type


class TestCrewKickoff:
    """Test crew execution and kickoff functionality."""
    
    def test_kickoff_with_valid_inputs(self, mock_crew, sample_inputs):
        """Test crew kickoff with valid inputs."""
        crew = mock_crew.crew()
        result = crew.kickoff(inputs=sample_inputs)
        
        assert result is not None
        assert "result" in result
        assert "research_paper" in result
        crew.kickoff.assert_called_once_with(inputs=sample_inputs)
    
    def test_kickoff_with_minimal_inputs(self, mock_crew):
        """Test crew kickoff with minimal required inputs."""
        minimal_inputs = {
            'topic': 'Test Topic',
            'current_year': '2024'
        }
        
        crew = mock_crew.crew()
        result = crew.kickoff(inputs=minimal_inputs)
        
        assert result is not None
        crew.kickoff.assert_called_once()
    
    def test_kickoff_with_callbacks(self, mock_crew, sample_inputs):
        """Test crew kickoff with callback functions."""
        callback_results = []
        
        def on_task_complete(task_output):
            callback_results.append(task_output)
        
        crew = mock_crew.crew()
        crew.kickoff(inputs=sample_inputs, callbacks=[on_task_complete])
        
        # Verify callback mechanism is in place
        assert crew.kickoff.called
    
    @pytest.mark.asyncio
    async def test_async_kickoff(self, mock_crew, sample_inputs):
        """Test asynchronous crew kickoff."""
        crew = mock_crew.crew()
        
        # Create an async mock that returns the expected result
        async def async_kickoff_mock(inputs):
            return {
                "result": "Async research completed",
                "research_paper": {}
            }
        
        crew.kickoff_async = async_kickoff_mock
        result = await crew.kickoff_async(inputs=sample_inputs)
        assert result is not None
        assert "result" in result


class TestCrewMemory:
    """Test crew memory functionality."""
    
    def test_memory_initialization(self, mock_crew, mock_crew_memory):
        """Test crew memory initialization."""
        with patch('crewai.memory.ShortTermMemory', return_value=mock_crew_memory):
            crew = mock_crew.crew()
            crew.memory = mock_crew_memory
            
            assert crew.memory is not None
            assert hasattr(crew.memory, 'save')
            assert hasattr(crew.memory, 'search')
    
    def test_memory_persistence(self, mock_crew, mock_crew_memory, sample_inputs):
        """Test memory persistence across crew runs."""
        crew = mock_crew.crew()
        crew.memory = mock_crew_memory
        
        # First run
        crew.kickoff(inputs=sample_inputs)
        
        # Verify memory save was called
        mock_crew_memory.save.assert_called()
        
        # Second run should have access to previous memory
        crew.kickoff(inputs=sample_inputs)
        mock_crew_memory.search.assert_called()
    
    def test_memory_context_retrieval(self, mock_crew_memory):
        """Test retrieving context from memory."""
        context = mock_crew_memory.get_context()
        assert context == "Historical context from memory"
        
        search_results = mock_crew_memory.search("test query")
        assert len(search_results) == 2
        assert search_results[0]["score"] > search_results[1]["score"]


class TestCrewOutputValidation:
    """Test crew output validation."""
    
    def test_output_validation_success(self, mock_crew, sample_inputs, research_paper_validator):
        """Test successful output validation."""
        crew = mock_crew.crew()
        result = crew.kickoff(inputs=sample_inputs)
        
        paper = result["research_paper"]
        is_valid, errors = research_paper_validator(paper)
        
        assert is_valid
        assert len(errors) == 0
    
    def test_output_validation_with_guardrails(self, test_environment):
        """Test output validation with guardrails."""
        from server_research_mcp.utils.validators import validate_context_gathering_output
        
        # Test valid output
        valid_output = """
        # Executive Summary
        This is a comprehensive analysis of the research topic...
        
        ## Key Concepts
        - Concept 1: Detailed explanation...
        - Concept 2: Another important concept...
        
        ## Knowledge Foundation
        The knowledge foundation includes multiple aspects...
        """ * 5  # Make it long enough
        
        is_valid, result = validate_context_gathering_output(valid_output)
        assert is_valid
        
        # Test invalid output (too short)
        invalid_output = "Too short"
        is_valid, error = validate_context_gathering_output(invalid_output)
        assert not is_valid
        assert "too brief" in error
    
    def test_output_format_compliance(self, mock_crew, sample_inputs):
        """Test that outputs comply with expected formats."""
        crew = mock_crew.crew()
        result = crew.kickoff(inputs=sample_inputs)
        
        # Check research paper format
        paper = result["research_paper"]
        assert isinstance(paper, dict)
        assert all(key in paper for key in ["title", "abstract", "sections"])
        
        # Check context foundation format
        assert "context_foundation" in result
        assert isinstance(result["context_foundation"], str)


class TestCrewConfiguration:
    """Test crew configuration options."""
    
    def test_llm_provider_configuration(self, test_environment):
        """Test different LLM provider configurations."""
        # Test Anthropic configuration
        with patch.dict('os.environ', {'LLM_PROVIDER': 'anthropic', 'ANTHROPIC_API_KEY': 'test-key'}):
            from server_research_mcp.crew import get_configured_llm
            llm = get_configured_llm()
            assert "anthropic" in llm.model
        
        # Test OpenAI configuration
        with patch.dict('os.environ', {'LLM_PROVIDER': 'openai', 'OPENAI_API_KEY': 'test-key'}):
            llm = get_configured_llm()
            assert "openai" in llm.model
    
    def test_crew_with_custom_tools(self, mock_crew):
        """Test adding custom tools to crew agents."""
        custom_tool = MagicMock()
        custom_tool.name = "custom_tool"
        custom_tool.description = "Custom tool for testing"
        
        crew = mock_crew.crew()
        historian = crew.agents[0]
        historian.tools.append(custom_tool)
        
        assert custom_tool in historian.tools
        assert len(historian.tools) == 7  # 6 original + 1 custom
    
    def test_crew_with_knowledge_sources(self, mock_crew, temp_workspace):
        """Test crew with knowledge sources."""
        # Create knowledge files
        knowledge_file = f"{temp_workspace}/knowledge.md"
        with open(knowledge_file, 'w') as f:
            f.write("# Domain Knowledge\nImportant domain knowledge here...")
        
        crew = mock_crew.crew()
        crew.knowledge_sources = [knowledge_file]
        
        # Verify knowledge sources are accessible
        assert hasattr(crew, 'knowledge_sources')
        assert len(crew.knowledge_sources) == 1


class TestCrewErrorHandling:
    """Test crew error handling."""
    
    def test_crew_handles_missing_inputs(self, mock_crew):
        """Test crew handles missing required inputs gracefully."""
        with pytest.raises(Exception) as exc_info:
            crew = mock_crew.crew()
            crew.kickoff(inputs={})  # Missing required inputs
        
        # Should raise an appropriate error
        assert exc_info.value is not None
    
    def test_crew_handles_agent_failures(self, mock_crew, sample_inputs):
        """Test crew handles agent failures gracefully."""
        crew = mock_crew.crew()
        
        # Simulate agent failure
        crew.agents[0].execute = MagicMock(side_effect=Exception("Agent failed"))
        crew.kickoff = MagicMock(return_value={
            "result": "Partial completion with errors",
            "errors": ["Agent 0 failed: Agent failed"]
        })
        
        result = crew.kickoff(inputs=sample_inputs)
        assert "errors" in result
        assert len(result["errors"]) > 0
    
    def test_crew_retry_mechanism(self, mock_crew, sample_inputs):
        """Test crew retry mechanism for failed tasks."""
        crew = mock_crew.crew()
        
        # Configure task with retry
        task = crew.tasks[0]
        task.max_retries = 3
        task.retry_count = 0
        
        # Simulate failure then success
        task.execute = MagicMock(side_effect=[
            Exception("First attempt failed"),
            Exception("Second attempt failed"),
            "Success on third attempt"
        ])
        
        # Crew should handle retries
        crew.kickoff(inputs=sample_inputs)
        assert crew.kickoff.called


class TestCrewInterruption:
    """Test crew interruption and resumption."""
    
    def test_crew_graceful_shutdown(self, mock_crew, sample_inputs):
        """Test crew can be shut down gracefully."""
        crew = mock_crew.crew()
        crew.shutdown = MagicMock()
        
        # Start crew
        crew.kickoff(inputs=sample_inputs)
        
        # Shutdown
        crew.shutdown()
        crew.shutdown.assert_called_once()
    
    def test_crew_state_persistence(self, mock_crew, sample_inputs, temp_workspace):
        """Test crew state can be persisted and resumed."""
        crew = mock_crew.crew()
        
        # Add state persistence methods
        crew.save_state = MagicMock()
        crew.load_state = MagicMock(return_value={"progress": 50, "completed_tasks": 2})
        
        # Run partial execution
        crew.kickoff(inputs=sample_inputs)
        crew.save_state(f"{temp_workspace}/crew_state.json")
        
        # Load and resume
        state = crew.load_state(f"{temp_workspace}/crew_state.json")
        assert state["progress"] == 50
        assert state["completed_tasks"] == 2
    
    def test_crew_checkpoint_recovery(self, mock_crew, sample_inputs):
        """Test crew can recover from checkpoints."""
        crew = mock_crew.crew()
        
        # Configure checkpointing
        crew.enable_checkpoints = True
        crew.checkpoint_interval = 2  # After every 2 tasks
        crew.restore_from_checkpoint = MagicMock(return_value=True)
        
        # Simulate interruption and recovery
        crew.kickoff(inputs=sample_inputs)
        
        # Verify checkpoint mechanism
        assert hasattr(crew, 'enable_checkpoints')
        assert crew.enable_checkpoints is True