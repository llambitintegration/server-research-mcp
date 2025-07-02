"""Enhanced pytest configuration with infrastructure fixes."""

import pytest
import os
import sys
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from pathlib import Path
import asyncio

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Import fixtures from dedicated fixture files
from .fixtures.mcp_fixtures import *


def enhanced_chromadb_config_patch():
    """Enhanced ChromaDB configuration patch that handles all _type field issues."""
    try:
        import chromadb
        from chromadb.api.configuration import CollectionConfigurationInternal
        
        # Patch the from_json method to handle missing _type
        original_from_json = CollectionConfigurationInternal.from_json
        
        def patched_from_json(cls, json_map):
            if isinstance(json_map, dict) and '_type' not in json_map:
                json_map['_type'] = 'hnsw'
            return original_from_json(json_map)
        
        CollectionConfigurationInternal.from_json = classmethod(patched_from_json)
        
        # Also patch from_json_str
        original_from_json_str = CollectionConfigurationInternal.from_json_str
        
        def patched_from_json_str(cls, json_str):
            import json
            try:
                config_json = json.loads(json_str)
                if isinstance(config_json, dict) and '_type' not in config_json:
                    config_json['_type'] = 'hnsw'
                    json_str = json.dumps(config_json)
            except:
                pass
            return original_from_json_str(json_str)
        
        CollectionConfigurationInternal.from_json_str = classmethod(patched_from_json_str)
        
        # Additional patches for CrewAI Settings validation
        try:
            from crewai.utilities.config import Settings
            original_settings_init = Settings.__init__
            
            def patched_settings_init(self, **data):
                # Add default _type for any configuration that needs it
                if '_type' not in data:
                    data['_type'] = 'hnsw'
                return original_settings_init(self, **data)
            
            Settings.__init__ = patched_settings_init
        except ImportError:
            pass  # CrewAI Settings not available
        
        print("✅ Enhanced ChromaDB and CrewAI configuration patched for tests")
        
    except ImportError:
        print("⚠️  ChromaDB not available for patching")
    except Exception as e:
        print(f"⚠️  ChromaDB patching failed: {e}")


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Enhanced test environment setup with infrastructure fixes."""
    # Create a temporary directory for test data
    temp_dir = tempfile.mkdtemp(prefix="test_chromadb_")
    
    # Set environment variables to isolate ChromaDB for tests
    os.environ["CHROMADB_PATH"] = temp_dir
    os.environ["CHROMADB_ALLOW_RESET"] = "true"
    
    # Only disable memory if not already set (allow test-specific overrides)
    if "DISABLE_CREW_MEMORY" not in os.environ:
        os.environ["DISABLE_CREW_MEMORY"] = "true"
    
    # Apply enhanced patches to prevent infrastructure issues
    enhanced_chromadb_config_patch()
    
    yield temp_dir
    
    # Cleanup
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass


@pytest.fixture
def enhanced_event_loop():
    """Enhanced event loop fixture that prevents reuse issues."""
    try:
        # Try to get existing loop
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop, create new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    yield loop
    
    # Enhanced cleanup
    try:
        # Cancel all running tasks
        pending = asyncio.all_tasks(loop)
        for task in pending:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete cancellation with timeout
        if pending:
            try:
                loop.run_until_complete(
                    asyncio.wait_for(
                        asyncio.gather(*pending, return_exceptions=True),
                        timeout=1.0
                    )
                )
            except asyncio.TimeoutError:
                pass  # Tasks didn't cancel in time, force close
        
    except Exception:
        pass
    finally:
        try:
            if not loop.is_closed():
                loop.close()
        except Exception:
            pass


@pytest.fixture
def enhanced_llm_config():
    """Enhanced LLM configuration that matches actual deployment."""
    return {
        'provider': 'anthropic',
        'model': 'anthropic/claude-sonnet-4-20250514',  # Match actual configuration
        'api_key': 'test-anthropic-key',
        'temperature': 0.7,
        'max_tokens': 4000
    }


@pytest.fixture
def enhanced_llm_instance(enhanced_llm_config):
    """Enhanced mock LLM instance with consistent responses."""
    from unittest.mock import MagicMock
    mock_llm = MagicMock()
    
    def smart_mock_call(messages):
        """Smart mock that provides contextually appropriate responses."""
        if isinstance(messages, str):
            content = messages.lower()
            if "42" in content:
                return "The answer is 42"
            elif "hello" in content:
                return "Hello, world"
            elif "successful" in content:
                return "LLM test successful"
            elif "2+2" in content:
                return "4"
            elif any(word in content for word in ["yes or no", "confirm", "continue"]):
                return "yes"
            else:
                return "Mock LLM response"
        elif isinstance(messages, list):
            last_message = messages[-1].get('content', '') if messages else ''
            if "42" in last_message:
                return "The number you asked about is 42"
            elif "successful" in last_message.lower():
                return "LLM test successful"
            elif any(word in last_message.lower() for word in ["hello", "greeting"]):
                return "Hello, world"
            else:
                return "Mock conversation response"
        return "Mock LLM response"
    
    mock_llm.call = smart_mock_call
    mock_llm.invoke = smart_mock_call
    mock_llm.model = enhanced_llm_config['model']
    mock_llm.api_key = enhanced_llm_config['api_key']
    mock_llm.provider = enhanced_llm_config['provider']
    
    return mock_llm


@pytest.fixture
def disable_crew_memory():
    """Enhanced crew memory disabling that prevents Pydantic issues."""
    with patch.dict(os.environ, {"DISABLE_CREW_MEMORY": "true"}):
        # Additional patches to prevent CrewAI memory initialization
        from unittest.mock import patch
        
        def mock_crew_init(self, *args, **kwargs):
            # Extract kwargs safely
            kwargs_copy = kwargs.copy()
            
            # Disable memory and planning to avoid ChromaDB issues
            kwargs_copy['memory'] = False
            kwargs_copy['planning'] = False
            
            # Remove any settings that might cause _type issues
            if 'settings' in kwargs_copy:
                settings = kwargs_copy['settings']
                if isinstance(settings, dict) and '_type' not in settings:
                    settings['_type'] = 'hnsw'
            
            # Call original init with safe parameters
            return original_crew_init(self, *args, **kwargs_copy)
        
        try:
            from crewai import Crew
            original_crew_init = Crew.__init__
            
            with patch.object(Crew, '__init__', mock_crew_init):
                yield
        except ImportError:
            yield


@pytest.fixture
def enhanced_mock_crew_with_memory_fix():
    """Enhanced mock crew that handles memory integration issues."""
    mock_crew = MagicMock()
    
    def mock_kickoff_with_enhanced_memory(inputs=None, callbacks=None, **kwargs):
        # Check for missing required inputs
        if not inputs or not inputs.get('topic'):
            raise ValueError("Missing required input: topic")
        
        # Handle memory interactions safely
        if inputs.get('enable_memory', False):
            # Mock memory interaction without actual ChromaDB
            memory_context = f"Memory context for: {inputs['topic']}"
            return f"Research completed for {inputs['topic']} with memory context: {memory_context}"
        
        return f"Research completed for {inputs['topic']}"
    
    mock_crew.kickoff = mock_kickoff_with_enhanced_memory
    
    # Mock agents with enhanced tool handling
    mock_agent = MagicMock()
    mock_agent.role = "test_agent"
    mock_agent.tools = [MagicMock(name=f"enhanced_tool_{i}") for i in range(6)]
    mock_crew.agents = [mock_agent] * 4
    
    # Mock tasks with enhanced validation
    mock_task = MagicMock()
    mock_task.description = "enhanced_test_task"
    mock_task.guardrail = MagicMock(return_value=True)
    mock_task.max_retries = 2
    mock_task.human_input = False
    mock_crew.tasks = [mock_task] * 4
    
    return mock_crew


@pytest.fixture
def enhanced_input_mocking():
    """Enhanced input mocking that prevents StopIteration issues."""
    def create_input_mock(responses):
        """Create a properly configured input mock."""
        if isinstance(responses, str):
            responses = [responses]
        
        call_count = {'count': 0}
        
        def mock_input_func(prompt=""):
            if call_count['count'] < len(responses):
                response = responses[call_count['count']]
                call_count['count'] += 1
                return response
            else:
                # Instead of StopIteration, return a default response
                return "n"  # Default to 'no' for continuation prompts
        
        return mock_input_func
    
    return create_input_mock


@pytest.fixture
def dotenv_fix():
    """Fix dotenv frame assertion issues in tests."""
    def mock_find_dotenv():
        # Return a simple path instead of relying on frame inspection
        return ".env.test"
    
    def mock_load_dotenv(dotenv_path=None):
        # Mock successful loading without frame inspection
        return True
    
    with patch('dotenv.find_dotenv', mock_find_dotenv):
        with patch('dotenv.load_dotenv', mock_load_dotenv):
            yield


@pytest.fixture
def enhanced_chromadb_mock():
    """Enhanced ChromaDB mock that prevents all _type issues."""
    mock_chromadb = MagicMock()
    
    def enhanced_query(*args, **kwargs):
        """Enhanced query that returns properly structured results."""
        return {
            "documents": [["Machine Learning fundamentals", "AI testing methodology"]],
            "metadatas": [[{"source": "test_doc_1"}, {"source": "test_doc_2"}]],
            "ids": [["doc_1", "doc_2"]],
            "distances": [[0.1, 0.3]]
        }
    
    def enhanced_add(*args, **kwargs):
        """Enhanced add operation."""
        return {"success": True, "ids": kwargs.get("ids", ["doc_1"])}
    
    # Configure collection mock
    mock_collection = MagicMock()
    mock_collection.query = enhanced_query
    mock_collection.add = enhanced_add
    
    # Configure client mock
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection
    mock_client.list_collections.return_value = [{"name": "test_collection"}]
    
    mock_chromadb.Client.return_value = mock_client
    mock_chromadb.PersistentClient.return_value = mock_client
    
    with patch('chromadb.Client', return_value=mock_client):
        with patch('chromadb.PersistentClient', return_value=mock_client):
            with patch('chromadb.config.Settings') as mock_settings:
                # Ensure Settings doesn't have _type issues
                mock_settings.return_value = MagicMock(_type='hnsw')
                yield mock_chromadb


@pytest.fixture
def sample_test_inputs():
    """Enhanced sample inputs for testing."""
    return {
        'paper_query': 'AI Testing Methodologies',
        'current_year': 2024,
        'topic': 'AI Testing Methodologies',
        'enable_memory': False,  # Disable memory by default for stability
        'output_format': 'markdown',
        'max_results': 10
    }


@pytest.fixture
def enhanced_crew_agents():
    """Enhanced mock crew agents with better tool simulation."""
    def create_enhanced_agent(role):
        agent = MagicMock()
        agent.role = role
        agent.tools = []
        
        # Add role-specific tools
        if "historian" in role:
            agent.tools = [MagicMock(name=f"memory_tool_{i}") for i in range(4)]
        elif "research" in role:
            agent.tools = [MagicMock(name=f"research_tool_{i}") for i in range(3)]
        elif "archivist" in role:
            agent.tools = [MagicMock(name="sequential_thinking")]
        elif "publisher" in role:
            agent.tools = [MagicMock(name=f"publish_tool_{i}") for i in range(2)]
        
        agent.execute = MagicMock(return_value=f"Task completed by {role}")
        return agent
    
    return {
        "historian": create_enhanced_agent("memory and context historian"),
        "researcher": create_enhanced_agent("research specialist"),
        "synthesizer": create_enhanced_agent("knowledge synthesizer"),
        "validator": create_enhanced_agent("quality validator"),
        "archivist": create_enhanced_agent("analysis archivist"),
        "publisher": create_enhanced_agent("content publisher")
    } 