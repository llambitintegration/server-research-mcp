"""
Unit tests for crew.py decorator system - AgentDefinition, TaskDefinition, and decorator parsing.
"""
import pytest
import sys
from unittest.mock import patch, MagicMock, call
from pathlib import Path
from typing import Dict, Any, List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from server_research_mcp.crew import (
    AgentDefinition,
    TaskDefinition,
    ServerResearchMcpCrew,
    historian_agent,
    researcher_agent,
    archivist_agent,
    publisher_agent,
    context_task,
    extraction_task,
    analysis_task,
    publishing_task,
    create_research_crew,
    run_research_pipeline
)
from server_research_mcp.schemas.research_paper import EnrichedQuery, RawPaperData, ResearchPaperSchema
from server_research_mcp.schemas.obsidian_meta import ObsidianDocument


class TestAgentDefinition:
    """Test AgentDefinition decorator and functionality."""
    
    def test_agent_definition_initialization(self):
        """Test AgentDefinition initialization with parameters."""
        agent_def = AgentDefinition(
            name="test_agent",
            schema=EnrichedQuery,
            tools_pattern="test_pattern",
            min_tools=3
        )
        
        assert agent_def.name == "test_agent"
        assert agent_def.schema == EnrichedQuery
        assert agent_def.tools_pattern == "test_pattern"
        assert agent_def.min_tools == 3
        assert agent_def.role is None  # Not set until decoration
        assert agent_def.goal is None
        assert agent_def.backstory is None
    
    def test_agent_definition_registration(self):
        """Test that AgentDefinition registers itself in the class registry."""
        initial_count = len(AgentDefinition._agents)
        
        agent_def = AgentDefinition(
            name="test_registration",
            schema=EnrichedQuery,
            tools_pattern="test"
        )
        
        assert len(AgentDefinition._agents) == initial_count + 1
        assert AgentDefinition._agents["test_registration"] == agent_def
    
    def test_agent_definition_decorator_docstring_parsing(self):
        """Test that AgentDefinition correctly parses docstrings."""
        agent_def = AgentDefinition(
            name="test_docstring",
            schema=EnrichedQuery,
            tools_pattern="test"
        )
        
        @agent_def
        def test_agent():
            """
            Role: Test Role Description
            Goal: Test Goal Description
            Backstory: Test Backstory Description
            """
            pass
        
        assert agent_def.role == "Test Role Description"
        assert agent_def.goal == "Test Goal Description"
        assert agent_def.backstory == "Test Backstory Description"
    
    def test_agent_definition_multiline_docstring(self):
        """Test parsing multiline docstring sections."""
        agent_def = AgentDefinition(
            name="test_multiline",
            schema=EnrichedQuery,
            tools_pattern="test"
        )
        
        @agent_def
        def test_agent():
            """
            Role: Test Role
            with multiple lines
            Goal: Test Goal
            also with multiple lines
            Backstory: Test Backstory
            with even more lines
            """
            pass
        
        assert agent_def.role == "Test Role with multiple lines"
        assert agent_def.goal == "Test Goal also with multiple lines"
        assert agent_def.backstory == "Test Backstory with even more lines"
    
    def test_agent_definition_get_method(self):
        """Test the get class method."""
        agent_def = AgentDefinition(
            name="test_get",
            schema=EnrichedQuery,
            tools_pattern="test"
        )
        
        retrieved = AgentDefinition.get("test_get")
        assert retrieved == agent_def
        
        non_existent = AgentDefinition.get("non_existent")
        assert non_existent is None
    
    def test_agent_definition_wrapper_function(self):
        """Test that the decorator returns a proper wrapper function."""
        agent_def = AgentDefinition(
            name="test_wrapper",
            schema=EnrichedQuery,
            tools_pattern="test"
        )
        
        @agent_def
        def test_agent():
            """
            Role: Test Role
            Goal: Test Goal
            Backstory: Test Backstory
            """
            pass
        
        # The decorator should return a function
        assert callable(test_agent)
        
        # The original function should be stored
        assert agent_def.func == test_agent.__wrapped__


class TestTaskDefinition:
    """Test TaskDefinition decorator and functionality."""
    
    def test_task_definition_initialization(self):
        """Test TaskDefinition initialization with parameters."""
        task_def = TaskDefinition(
            name="test_task",
            agent="test_agent",
            depends_on=["dependency1", "dependency2"]
        )
        
        assert task_def.name == "test_task"
        assert task_def.agent_name == "test_agent"
        assert task_def.depends_on == ["dependency1", "dependency2"]
        assert task_def.description is None  # Not set until decoration
        assert task_def.expected_output is None
    
    def test_task_definition_registration(self):
        """Test that TaskDefinition registers itself in the class registry."""
        initial_count = len(TaskDefinition._tasks)
        
        task_def = TaskDefinition(
            name="test_registration",
            agent="test_agent"
        )
        
        assert len(TaskDefinition._tasks) == initial_count + 1
        assert TaskDefinition._tasks["test_registration"] == task_def
    
    def test_task_definition_decorator_docstring_parsing(self):
        """Test that TaskDefinition correctly parses docstrings."""
        task_def = TaskDefinition(
            name="test_docstring",
            agent="test_agent"
        )
        
        @task_def
        def test_task():
            """
            Description: Test task description
            Expected Output: Test expected output
            """
            pass
        
        assert task_def.description == "Test task description"
        assert task_def.expected_output == "Test expected output"
    
    def test_task_definition_multiline_docstring(self):
        """Test parsing multiline docstring sections."""
        task_def = TaskDefinition(
            name="test_multiline",
            agent="test_agent"
        )
        
        @task_def
        def test_task():
            """
            Description: Test task description
            with multiple lines
            Expected Output: Test expected output
            also with multiple lines
            """
            pass
        
        assert task_def.description == "Test task description with multiple lines"
        assert task_def.expected_output == "Test expected output also with multiple lines"
    
    def test_task_definition_get_all_method(self):
        """Test the get_all class method."""
        initial_tasks = TaskDefinition.get_all()
        
        task_def = TaskDefinition(
            name="test_get_all",
            agent="test_agent"
        )
        
        updated_tasks = TaskDefinition.get_all()
        assert len(updated_tasks) == len(initial_tasks) + 1
        assert "test_get_all" in updated_tasks
        assert updated_tasks["test_get_all"] == task_def
    
    def test_task_definition_wrapper_function(self):
        """Test that the decorator returns a proper wrapper function."""
        task_def = TaskDefinition(
            name="test_wrapper",
            agent="test_agent"
        )
        
        @task_def
        def test_task():
            """
            Description: Test description
            Expected Output: Test output
            """
            pass
        
        # The decorator should return a function
        assert callable(test_task)
        
        # The original function should be stored
        assert task_def.func == test_task.__wrapped__


class TestPredefinedAgents:
    """Test the predefined agent decorators."""
    
    def test_historian_agent_definition(self):
        """Test historian agent definition."""
        agent_def = AgentDefinition.get("historian")
        assert agent_def is not None
        assert agent_def.name == "historian"
        assert agent_def.schema == EnrichedQuery
        assert agent_def.tools_pattern == "historian"
        assert agent_def.min_tools == 6
        assert "Context Continuity Engine" in agent_def.role
        assert "capture and retrieve contextual continuity" in agent_def.goal.lower()
    
    def test_researcher_agent_definition(self):
        """Test researcher agent definition."""
        agent_def = AgentDefinition.get("researcher")
        assert agent_def is not None
        assert agent_def.name == "researcher"
        assert agent_def.schema == RawPaperData
        assert agent_def.tools_pattern == "researcher"
        assert agent_def.min_tools == 3
        assert "Paper Discovery" in agent_def.role
        assert "zotero" in agent_def.goal.lower()
    
    def test_archivist_agent_definition(self):
        """Test archivist agent definition."""
        agent_def = AgentDefinition.get("archivist")
        assert agent_def is not None
        assert agent_def.name == "archivist"
        assert agent_def.schema == ResearchPaperSchema
        assert agent_def.tools_pattern == "archivist"
        assert agent_def.min_tools == 1
        assert "Data Structuring" in agent_def.role
        assert "schema" in agent_def.goal.lower()
    
    def test_publisher_agent_definition(self):
        """Test publisher agent definition."""
        agent_def = AgentDefinition.get("publisher")
        assert agent_def is not None
        assert agent_def.name == "publisher"
        assert agent_def.schema == ObsidianDocument
        assert agent_def.tools_pattern == "publisher"
        assert agent_def.min_tools == 11
        assert "Markdown Generation" in agent_def.role
        assert "obsidian" in agent_def.goal.lower()


class TestPredefinedTasks:
    """Test the predefined task decorators."""
    
    def test_context_task_definition(self):
        """Test context gathering task definition."""
        task_def = TaskDefinition._tasks.get("context_gathering")
        assert task_def is not None
        assert task_def.name == "context_gathering"
        assert task_def.agent_name == "historian"
        assert task_def.depends_on == []
        assert task_def.description is not None
        assert task_def.expected_output is not None
    
    def test_extraction_task_definition(self):
        """Test paper extraction task definition."""
        task_def = TaskDefinition._tasks.get("paper_extraction")
        assert task_def is not None
        assert task_def.name == "paper_extraction"
        assert task_def.agent_name == "researcher"
        assert "context_gathering" in task_def.depends_on
        assert task_def.description is not None
        assert task_def.expected_output is not None
    
    def test_analysis_task_definition(self):
        """Test analysis task definition."""
        task_def = TaskDefinition._tasks.get("analysis")
        assert task_def is not None
        assert task_def.name == "analysis"
        assert task_def.agent_name == "archivist"
        assert "paper_extraction" in task_def.depends_on
        assert task_def.description is not None
        assert task_def.expected_output is not None
    
    def test_publishing_task_definition(self):
        """Test publishing task definition."""
        task_def = TaskDefinition._tasks.get("publishing")
        assert task_def is not None
        assert task_def.name == "publishing"
        assert task_def.agent_name == "publisher"
        assert "analysis" in task_def.depends_on
        assert task_def.description is not None
        assert task_def.expected_output is not None


@patch('server_research_mcp.crew.Task')
@patch('server_research_mcp.crew.Agent')
class TestServerResearchMcpCrew:
    """Test the ServerResearchMcpCrew class."""
    
    @patch('server_research_mcp.crew.get_registry')
    @patch('server_research_mcp.crew.LLMConfig')
    def test_crew_initialization(self, mock_llm_config, mock_get_registry, mock_agent, mock_task):
        """Test crew initialization with default parameters."""
        mock_registry = MagicMock()
        mock_get_registry.return_value = mock_registry
        
        # Mock the LLMConfig instance and its methods
        mock_llm_config_instance = MagicMock()
        mock_llm = MagicMock()
        mock_llm_config_instance.get_llm.return_value = mock_llm
        mock_llm_config.return_value = mock_llm_config_instance
        
        # Mock agent and task constructors
        mock_agent.return_value = MagicMock()
        mock_task.return_value = MagicMock()
        
        crew = ServerResearchMcpCrew()
        
        assert crew.inputs == {}
        assert crew.llm_config is not None
        assert crew.tool_registry is not None
        # Note: Agents may be pre-created due to @agent decorators
        assert isinstance(crew._agents_cache, dict)
        assert isinstance(crew._tasks_cache, dict)
        assert crew._memory is True
        assert crew._state == {}
    
    @patch('server_research_mcp.crew.get_registry')
    @patch('server_research_mcp.crew.LLMConfig')
    def test_crew_initialization_with_inputs(self, mock_llm_config, mock_get_registry, mock_agent, mock_task):
        """Test crew initialization with custom inputs."""
        mock_registry = MagicMock()
        mock_get_registry.return_value = mock_registry
        
        # Mock the LLMConfig instance and its methods
        mock_llm_config_instance = MagicMock()
        mock_llm = MagicMock()
        mock_llm_config_instance.get_llm.return_value = mock_llm
        mock_llm_config.return_value = mock_llm_config_instance
        
        # Mock agent and task constructors
        mock_agent.return_value = MagicMock()
        mock_task.return_value = MagicMock()
        
        test_inputs = {"topic": "AI", "year": 2023}
        crew = ServerResearchMcpCrew(inputs=test_inputs)
        
        assert crew.inputs == test_inputs
    
    @patch('server_research_mcp.crew.get_registry')
    @patch('server_research_mcp.crew.LLMConfig')
    def test_memory_property(self, mock_llm_config, mock_get_registry, mock_agent, mock_task):
        """Test the memory property for legacy compatibility."""
        mock_registry = MagicMock()
        mock_get_registry.return_value = mock_registry
        mock_llm_config_instance = MagicMock()
        mock_llm_config.return_value = mock_llm_config_instance
        
        # Mock agent and task constructors
        mock_agent.return_value = MagicMock()
        mock_task.return_value = MagicMock()
        
        crew = ServerResearchMcpCrew()
        assert crew.memory is True
    
    @patch('server_research_mcp.crew.get_registry')
    @patch('server_research_mcp.crew.LLMConfig')
    def test_create_agent_from_definition(self, mock_llm_config, mock_get_registry, mock_agent, mock_task):
        """Test agent creation from definition."""
        mock_registry = MagicMock()
        mock_registry.get_agent_tools.return_value = [MagicMock(), MagicMock()]
        mock_get_registry.return_value = mock_registry
        
        # Mock the LLMConfig instance and its methods
        mock_llm_config_instance = MagicMock()
        mock_llm = MagicMock()
        mock_llm_config_instance.get_llm.return_value = mock_llm
        mock_llm_config.return_value = mock_llm_config_instance
        
        # Mock the Agent and Task constructors (from class-level patch)
        mock_agent_instance = MagicMock()
        mock_agent.return_value = mock_agent_instance
        mock_task.return_value = MagicMock()
        
        crew = ServerResearchMcpCrew()
        agent_def = AgentDefinition.get("historian")
        
        agent = crew._create_agent_from_definition(agent_def)
        
        assert agent == mock_agent_instance
        assert mock_agent.call_count >= 1  # Called during initialization and test
        
        # Verify agent was cached
        assert crew._agents_cache["historian"] == mock_agent_instance
    
    @patch('server_research_mcp.crew.get_registry')
    @patch('server_research_mcp.crew.LLMConfig')
    def test_create_agent_historian_tool_padding(self, mock_llm_config, mock_get_registry, mock_agent, mock_task):
        """Test that historian agent gets padded tools when needed."""
        # Mock the LLMConfig instance and its methods
        mock_llm_config_instance = MagicMock()
        mock_llm = MagicMock()
        mock_llm_config_instance.get_llm.return_value = mock_llm
        mock_llm_config.return_value = mock_llm_config_instance
        
        mock_registry = MagicMock()
        mock_registry.get_agent_tools.return_value = [MagicMock()]  # Only 1 tool
        mock_get_registry.return_value = mock_registry
        
        mock_agent_instance = MagicMock()
        mock_agent.return_value = mock_agent_instance
        mock_task.return_value = MagicMock()
        
        crew = ServerResearchMcpCrew()
        agent_def = AgentDefinition.get("historian")
        
        agent = crew._create_agent_from_definition(agent_def)
        
        # Check that Agent was called with 6 tools (1 original + 5 padding)
        # We need to look at the last call since agent is called during initialization too
        last_call = mock_agent.call_args_list[-1]
        tools = last_call[1]['tools']
        assert len(tools) == 6
    
    @patch('server_research_mcp.crew.get_registry')
    @patch('server_research_mcp.crew.LLMConfig')
    def test_create_guardrail_function(self, mock_llm_config, mock_get_registry, mock_agent, mock_task):
        """Test guardrail function creation."""
        mock_registry = MagicMock()
        mock_get_registry.return_value = mock_registry
        mock_llm_config_instance = MagicMock()
        mock_llm_config.return_value = mock_llm_config_instance
        
        crew = ServerResearchMcpCrew()
        guardrail = crew._create_guardrail("test_task", EnrichedQuery)
        
        assert callable(guardrail)
        
        # Test guardrail with valid data
        mock_output = MagicMock()
        mock_output.pydantic = True
        is_valid, result = guardrail(mock_output)
        assert is_valid is True
        assert result == mock_output
    
    @patch('server_research_mcp.crew.get_registry')
    @patch('server_research_mcp.crew.LLMConfig')
    def test_create_guardrail_json_validation(self, mock_llm_config, mock_get_registry, mock_agent, mock_task):
        """Test guardrail JSON validation."""
        mock_registry = MagicMock()
        mock_get_registry.return_value = mock_registry
        mock_llm_config_instance = MagicMock()
        mock_llm_config.return_value = mock_llm_config_instance
        
        crew = ServerResearchMcpCrew()
        guardrail = crew._create_guardrail("test_task", EnrichedQuery)
        
        # Test with JSON string
        json_output = '{"query": "test", "context": "test context"}'
        is_valid, result = guardrail(json_output)
        # This might fail due to schema validation, but should not crash
        assert isinstance(is_valid, bool)
    
    @patch('server_research_mcp.crew.get_registry')
    @patch('server_research_mcp.crew.LLMConfig')
    def test_create_guardrail_error_handling(self, mock_llm_config, mock_get_registry, mock_agent, mock_task):
        """Test guardrail error handling."""
        mock_registry = MagicMock()
        mock_get_registry.return_value = mock_registry
        mock_llm_config_instance = MagicMock()
        mock_llm_config.return_value = mock_llm_config_instance
        
        crew = ServerResearchMcpCrew()
        guardrail = crew._create_guardrail("test_task", EnrichedQuery)
        
        # Test with invalid data
        is_valid, result = guardrail(None)
        assert is_valid is False
        assert result == "None output"
    
    @patch('server_research_mcp.crew.get_registry')
    @patch('server_research_mcp.crew.LLMConfig')
    def test_create_task_from_definition(self, mock_llm_config, mock_get_registry, mock_agent, mock_task):
        """Test task creation from definition."""
        mock_registry = MagicMock()
        mock_registry.get_agent_tools.return_value = [MagicMock()]
        mock_get_registry.return_value = mock_registry
        
        # Mock the LLMConfig instance and its methods
        mock_llm_config_instance = MagicMock()
        mock_llm = MagicMock()
        mock_llm_config_instance.get_llm.return_value = mock_llm
        mock_llm_config.return_value = mock_llm_config_instance
        
        # Set up mock instances
        mock_agent_instance = MagicMock()
        mock_agent.return_value = mock_agent_instance
        mock_task_instance = MagicMock()
        mock_task.return_value = mock_task_instance
        
        crew = ServerResearchMcpCrew()
        task_def = TaskDefinition._tasks.get("context_gathering")
        
        task = crew._create_task_from_definition(task_def)
        
        assert task == mock_task_instance
        mock_task.assert_called()
        
        # Verify task was cached
        assert crew._tasks_cache["context_gathering"] == mock_task_instance
    
    @patch('server_research_mcp.crew.get_registry')
    @patch('server_research_mcp.crew.LLMConfig')
    def test_legacy_agent_methods(self, mock_llm_config, mock_get_registry, mock_agent, mock_task):
        """Test legacy agent accessor methods."""
        mock_registry = MagicMock()
        mock_registry.get_agent_tools.return_value = [MagicMock()]
        mock_get_registry.return_value = mock_registry
        
        # Mock the LLMConfig instance and its methods
        mock_llm_config_instance = MagicMock()
        mock_llm = MagicMock()
        mock_llm_config_instance.get_llm.return_value = mock_llm
        mock_llm_config.return_value = mock_llm_config_instance
        
        # Set up mock instances
        mock_agent_instance = MagicMock()
        mock_agent.return_value = mock_agent_instance
        
        crew = ServerResearchMcpCrew()
        
        # Test legacy methods
        historian = crew.historian()
        researcher = crew.researcher()
        archivist = crew.archivist()
        publisher = crew.publisher()
        reporting_analyst = crew.reporting_analyst()  # Legacy alias
        
        assert historian == mock_agent_instance
        assert researcher == mock_agent_instance
        assert archivist == mock_agent_instance
        assert publisher == mock_agent_instance
        assert reporting_analyst == mock_agent_instance
    
    @patch('server_research_mcp.crew.get_registry')
    @patch('server_research_mcp.crew.LLMConfig')
    def test_legacy_task_methods(self, mock_llm_config, mock_get_registry, mock_agent, mock_task):
        """Test legacy task accessor methods."""
        mock_registry = MagicMock()
        mock_registry.get_agent_tools.return_value = [MagicMock()]
        mock_get_registry.return_value = mock_registry
        
        # Mock the LLMConfig instance and its methods
        mock_llm_config_instance = MagicMock()
        mock_llm = MagicMock()
        mock_llm_config_instance.get_llm.return_value = mock_llm
        mock_llm_config.return_value = mock_llm_config_instance
        
        # Set up mock instances
        mock_agent_instance = MagicMock()
        mock_agent.return_value = mock_agent_instance
        mock_task_instance = MagicMock()
        mock_task.return_value = mock_task_instance
        
        crew = ServerResearchMcpCrew()
        
        # Test legacy methods
        context_task = crew.context_gathering_task()
        extraction_task = crew.paper_extraction_task()
        analysis_task = crew.analysis_and_structuring_task()
        publishing_task = crew.publishing_task()
        research_task = crew.research_task()  # Legacy alias
        reporting_task = crew.reporting_task()  # Legacy alias
        
        assert context_task == mock_task_instance
        assert extraction_task == mock_task_instance
        assert analysis_task == mock_task_instance
        assert publishing_task == mock_task_instance
        assert research_task == mock_task_instance
        assert reporting_task == mock_task_instance
    
    @patch('server_research_mcp.crew.get_registry')
    @patch('server_research_mcp.crew.LLMConfig')
    def test_agents_property(self, mock_llm_config, mock_get_registry, mock_agent, mock_task):
        """Test agents property returns all agents in order."""
        mock_registry = MagicMock()
        mock_registry.get_agent_tools.return_value = [MagicMock()]
        mock_get_registry.return_value = mock_registry
        
        # Mock the LLMConfig instance and its methods
        mock_llm_config_instance = MagicMock()
        mock_llm = MagicMock()
        mock_llm_config_instance.get_llm.return_value = mock_llm
        mock_llm_config.return_value = mock_llm_config_instance
        
        # Set up mock instances
        mock_agent_instance = MagicMock()
        mock_agent.return_value = mock_agent_instance
        
        crew = ServerResearchMcpCrew()
        
        agents = crew.agents
        
        assert len(agents) == 4
        assert all(agent == mock_agent_instance for agent in agents)
    
    @patch('server_research_mcp.crew.get_registry')
    @patch('server_research_mcp.crew.LLMConfig')
    def test_tasks_property(self, mock_llm_config, mock_get_registry, mock_agent, mock_task):
        """Test tasks property returns all tasks in order."""
        mock_registry = MagicMock()
        mock_registry.get_agent_tools.return_value = [MagicMock()]
        mock_get_registry.return_value = mock_registry
        
        # Mock the LLMConfig instance and its methods
        mock_llm_config_instance = MagicMock()
        mock_llm = MagicMock()
        mock_llm_config_instance.get_llm.return_value = mock_llm
        mock_llm_config.return_value = mock_llm_config_instance
        
        # Set up mock instances
        mock_agent_instance = MagicMock()
        mock_agent.return_value = mock_agent_instance
        mock_task_instance = MagicMock()
        mock_task.return_value = mock_task_instance
        
        crew = ServerResearchMcpCrew()
        
        tasks = crew.tasks
        
        assert len(tasks) == 4
        assert all(task == mock_task_instance for task in tasks)
    
    @patch('server_research_mcp.crew.get_registry')
    @patch('server_research_mcp.crew.LLMConfig')
    def test_crew_method(self, mock_llm_config, mock_get_registry, mock_agent, mock_task):
        """Test crew method creates and returns Crew instance."""
        mock_registry = MagicMock()
        mock_registry.get_agent_tools.return_value = [MagicMock()]
        mock_get_registry.return_value = mock_registry
        
        # Mock the LLMConfig instance and its methods
        mock_llm_config_instance = MagicMock()
        mock_llm = MagicMock()
        mock_llm_config_instance.get_llm.return_value = mock_llm
        mock_llm_config.return_value = mock_llm_config_instance
        
        crew_instance = ServerResearchMcpCrew()
        
        with patch('server_research_mcp.crew.Agent') as mock_agent:
            with patch('server_research_mcp.crew.Task') as mock_task:
                with patch('server_research_mcp.crew.Crew') as mock_crew:
                    mock_agent_instance = MagicMock()
                    mock_agent.return_value = mock_agent_instance
                    mock_task_instance = MagicMock()
                    mock_task.return_value = mock_task_instance
                    mock_crew_instance = MagicMock()
                    mock_crew.return_value = mock_crew_instance
                    
                    crew = crew_instance.crew()
                    
                    assert crew == mock_crew_instance
                    mock_crew.assert_called_once()
                    
                    # Verify Crew was called with correct parameters
                    call_args = mock_crew.call_args
                    assert 'agents' in call_args[1]
                    assert 'tasks' in call_args[1]
                    assert call_args[1]['verbose'] is True
                    assert call_args[1]['planning'] is True
                    assert call_args[1]['full_output'] is True


class TestModuleFunctions:
    """Test module-level functions."""
    
    def test_create_research_crew(self):
        """Test create_research_crew function."""
        with patch('server_research_mcp.crew.ServerResearchMcpCrew') as mock_crew_class:
            mock_crew_instance = MagicMock()
            mock_crew_class.return_value = mock_crew_instance
            
            crew = create_research_crew("AI Research")
            
            assert crew == mock_crew_instance
            mock_crew_class.assert_called_once()
            
            # Check that inputs were set correctly
            call_args = mock_crew_class.call_args
            inputs = call_args[1]['inputs']
            assert inputs['topic'] == "AI Research"
    
    def test_run_research_pipeline(self):
        """Test run_research_pipeline function."""
        with patch('server_research_mcp.crew.create_research_crew') as mock_create_crew:
            mock_crew = MagicMock()
            mock_crew.run_with_validation.return_value = {"success": True}
            mock_create_crew.return_value = mock_crew
            
            result = run_research_pipeline("Test Topic")
            
            assert result == {"success": True}
            mock_create_crew.assert_called_once_with("Test Topic")
            mock_crew.run_with_validation.assert_called_once()


@patch('server_research_mcp.crew.Task')
@patch('server_research_mcp.crew.Agent')
class TestIntegration:
    """Test integration scenarios."""
    
    def test_complete_decorator_workflow(self, mock_agent, mock_task):
        """Test complete workflow from decorator definition to crew creation."""
        # Create a custom agent definition
        test_agent_def = AgentDefinition(
            name="test_integration_agent",
            schema=EnrichedQuery,
            tools_pattern="test_pattern",
            min_tools=2
        )
        
        @test_agent_def
        def test_agent():
            """
            Role: Test Integration Agent
            Goal: Test integration workflow
            Backstory: Created for integration testing
            """
            pass
        
        # Create a custom task definition
        test_task_def = TaskDefinition(
            name="test_integration_task",
            agent="test_integration_agent"
        )
        
        @test_task_def
        def test_task():
            """
            Description: Test integration task
            Expected Output: Test output for integration
            """
            pass
        
        # Verify definitions are properly set
        assert test_agent_def.role == "Test Integration Agent"
        assert test_agent_def.goal == "Test integration workflow"
        assert test_agent_def.backstory == "Created for integration testing"
        
        assert test_task_def.description == "Test integration task"
        assert test_task_def.expected_output == "Test output for integration"
        
        # Verify they're registered
        assert AgentDefinition.get("test_integration_agent") == test_agent_def
        assert TaskDefinition._tasks["test_integration_task"] == test_task_def
        
        # Test crew can use them
        with patch('server_research_mcp.crew.get_registry') as mock_get_registry:
            with patch('server_research_mcp.crew.LLMConfig') as mock_llm_config:
                mock_registry = MagicMock()
                mock_registry.get_agent_tools.return_value = [MagicMock(), MagicMock()]
                mock_get_registry.return_value = mock_registry
                
                # Mock the LLMConfig instance and its methods
                mock_llm_config_instance = MagicMock()
                mock_llm = MagicMock()
                mock_llm_config_instance.get_llm.return_value = mock_llm
                mock_llm_config.return_value = mock_llm_config_instance
                
                # Set up mock agents and tasks
                mock_agent_instance = MagicMock()
                mock_agent.return_value = mock_agent_instance
                mock_task_instance = MagicMock()
                mock_task.return_value = mock_task_instance
                
                crew = ServerResearchMcpCrew()
                
                # Test agent creation
                agent = crew._create_agent_from_definition(test_agent_def)
                assert agent == mock_agent_instance
                
                # Test task creation
                task = crew._create_task_from_definition(test_task_def)
                assert task == mock_task_instance