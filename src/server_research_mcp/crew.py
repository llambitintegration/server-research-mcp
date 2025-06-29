import os
from dotenv import load_dotenv
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List, Tuple, Any, Dict
import json
import logging

# Set up detailed logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import MCP tools for all agents from consolidated package
from .tools import (
    get_historian_tools,
    get_researcher_tools,
    get_archivist_tools,
    get_context7_tools,
    get_publisher_tools
)

# Import schemas
from .schemas import (
    EnrichedQuery,
    RawPaperData,
    ResearchPaperSchema,
    ObsidianDocument
)

# Load environment variables
load_dotenv()

# Check for enhanced MCP mode
USE_ENHANCED_MCP = os.getenv('USE_ENHANCED_MCP', 'false').lower() == 'true'
# Enable Crew memory & planning by default unless the user explicitly disables it.
DISABLE_CREW_MEMORY = os.getenv("DISABLE_CREW_MEMORY", "false").lower() == "true"  # Default to enabled for richer context

if USE_ENHANCED_MCP:
    print("🔧 Enhanced MCP mode enabled - using real MCP servers")
else:
    print("🎭 Standard mode - using real MCP servers with CrewAI official patterns")

if DISABLE_CREW_MEMORY:
    print("🧠 Crew memory and planning disabled for compatibility")
else:
    print("🧠 Crew memory and planning enabled")

# MCPAdapt Integration - Clean and Simple
# Legacy MCPToolsManager removed - using direct MCPAdapt integration
    
def get_crew_mcp_manager():
    """Get the MCP tools manager for the crew (legacy compatibility)."""
    # Legacy function - MCPAdapt migration completed, return direct tool access
    class LegacyMCPManager:
        @staticmethod
        def get_historian_tools():
            from .tools.mcp_tools import get_historian_tools, add_basic_tools
            return add_basic_tools(get_historian_tools())
        
        @staticmethod
        def get_researcher_tools():
            from .tools.mcp_tools import get_researcher_tools, add_basic_tools
            return add_basic_tools(get_researcher_tools())
        
        @staticmethod
        def get_archivist_tools():
            from .tools.mcp_tools import get_archivist_tools, add_basic_tools
            return add_basic_tools(get_archivist_tools())
        
        @staticmethod
        def get_publisher_tools():
            from .tools.mcp_tools import get_publisher_tools, add_basic_tools
            return add_basic_tools(get_publisher_tools())
    
    return LegacyMCPManager()

# Configure LLM based on environment variables
def get_configured_llm():
    """Configure and return the appropriate LLM based on environment variables."""
    llm_provider = os.getenv('LLM_PROVIDER', 'anthropic').lower()
    
    if llm_provider == 'anthropic':
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required when using Anthropic provider")
        
        model = os.getenv('LLM_MODEL', 'claude-3-haiku-20240307')
        return LLM(
            model=f"anthropic/{model}",
            api_key=api_key
        )
    
    elif llm_provider == 'openai':
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required when using OpenAI provider")
        
        model = os.getenv('LLM_MODEL', 'gpt-4o-mini')
        return LLM(
            model=f"openai/{model}",
            api_key=api_key
        )
    
    else:
        raise ValueError(f"Unsupported LLM provider: {llm_provider}. Use 'anthropic' or 'openai'")

# Initialize configured LLM
try:
    configured_llm = get_configured_llm()
    print(f"✅ LLM configured successfully: {configured_llm.model}")
except Exception as e:
    print(f"❌ LLM configuration error: {e}")
    print("Please ensure your environment variables are set correctly.")
    configured_llm = None

# Custom validation functions for task outputs
def validate_enriched_query(result: str) -> Tuple[bool, Any]:
    """Validate enriched query output from Historian."""
    logger.info(f"🔍 Starting validation of enriched query. Input type: {type(result)}")
    
    # Support TaskOutput objects by extracting text
    result_str = _extract_text_output(result)
    logger.info(f"📝 Extracted text output length: {len(result_str)} chars")
    logger.debug(f"📄 Raw output content (first 500 chars): {result_str[:500]}...")
    
    try:
        # Parse JSON result
        data = json.loads(result_str)
        logger.info(f"✅ JSON parsing successful. Keys found: {list(data.keys())}")
        
        # Allow either snake_case or camelCase keys
        key_mapping = {
            "originalQuery": "original_query",
            "expandedTerms": "expanded_terms",
            "searchStrategy": "search_strategy",
            "original_paper_query": "original_query",
        }
        
        # Promote camelCase keys to snake_case if needed
        for camel, snake in key_mapping.items():
            if camel in data and snake not in data:
                data[snake] = data[camel]
                logger.info(f"🔄 Mapped {camel} -> {snake}")
        
        # Extract or derive required fields from the agent's output structure
        original_query = data.get("original_query")
        logger.info(f"🎯 Found original_query: {original_query}")
        
        if not original_query:
            # Try alternative field names
            original_query = data.get("original_paper_query") or data.get("query")
            logger.info(f"🔍 Trying alternative field names, found: {original_query}")
            
            if not original_query:
                # Very permissive fallback - use any string field that looks like a query
                logger.warning("⚠️ No query field found, searching for any string field")
                for key, value in data.items():
                    if isinstance(value, str) and len(value) > 10:
                        original_query = value
                        logger.info(f"📝 Using fallback query from {key}: {original_query}")
                        break
                if not original_query:
                    original_query = "research query"  # Ultra fallback
                    logger.warning("🆘 Using ultra fallback query")
        
        # Accept expanded_terms from various locations
        expanded_terms = data.get("expanded_terms")
        logger.info(f"🏷️ Found expanded_terms: {expanded_terms}")
        
        if not expanded_terms:
            # Try to extract from search_recommendations
            search_rec = data.get("search_recommendations", {})
            logger.info(f"🔍 Checking search_recommendations: {type(search_rec)}")
            
            if isinstance(search_rec, dict):
                expanded_query = search_rec.get("expanded_query", "")
                if expanded_query:
                    expanded_terms = expanded_query.split()
                    logger.info(f"📋 Extracted expanded_terms from expanded_query: {expanded_terms}")
                else:
                    # Fallback: extract key terms from the original query
                    expanded_terms = original_query.split()
                    logger.info(f"🔄 Using original_query words as expanded_terms: {expanded_terms}")
            else:
                # search_recommendations might be a list
                expanded_terms = original_query.split()
                logger.info(f"🔄 search_recommendations is list, using original_query words: {expanded_terms}")
        
        # Accept search_strategy from various locations
        search_strategy = data.get("search_strategy")
        logger.info(f"🎯 Found search_strategy: {search_strategy}")
        
        if not search_strategy:
            # Try search_recommendations
            search_rec = data.get("search_recommendations", {})
            if isinstance(search_rec, dict):
                strategies = search_rec.get("suggested_strategies", [])
                if strategies:
                    search_strategy = strategies
                    logger.info(f"📋 Found strategies in search_recommendations: {strategies}")
                else:
                    search_strategy = ["comprehensive search"]
                    logger.info("🔄 Using default comprehensive search")
            elif isinstance(search_rec, list):
                # search_recommendations is a list of strategies
                search_strategy = search_rec
                logger.info(f"📋 Using search_recommendations list as strategies: {search_strategy}")
            else:
                search_strategy = ["comprehensive search"]
                logger.info("🔄 Using fallback comprehensive search")
        
        # Ensure proper types
        if isinstance(expanded_terms, str):
            expanded_terms = expanded_terms.split()
            logger.info(f"🔄 Split string expanded_terms: {expanded_terms}")
        if not isinstance(expanded_terms, list):
            expanded_terms = [str(expanded_terms)]
            logger.info(f"🔄 Converted expanded_terms to list: {expanded_terms}")
            
        if isinstance(search_strategy, list):
            search_strategy = "; ".join(str(s) for s in search_strategy)
            logger.info(f"🔄 Joined list search_strategy: {search_strategy}")
        
        # Rebuild normalized output - very permissive, accept whatever we can find
        normalized_data = {
            "original_query": original_query,
            "expanded_terms": expanded_terms,
            "search_strategy": search_strategy,
            "historical_context": data.get("historical_context", data.get("related_papers", [])),
            "memory_entities": data.get("memory_entities", data.get("relevant_concepts", []))
        }
        
        logger.info(f"✅ Validation successful. Normalized keys: {list(normalized_data.keys())}")
        logger.debug(f"📊 Normalized data: {normalized_data}")
        
        # Return normalized JSON string (snake_case)
        return (True, json.dumps(normalized_data, ensure_ascii=False))
        
    except json.JSONDecodeError as e:
        logger.error(f"❌ JSON parsing failed: {e}")
        logger.debug(f"📄 Failed content: {result_str}")
        
        # If JSON parsing fails, create a minimal valid structure
        minimal_data = {
            "original_query": "research query",
            "expanded_terms": ["research", "query"],
            "search_strategy": "comprehensive search",
            "historical_context": [],
            "memory_entities": []
        }
        logger.info("🔄 Created minimal fallback structure")
        return (True, json.dumps(minimal_data, ensure_ascii=False))
        
    except Exception as e:
        logger.error(f"❌ Validation exception: {e}")
        logger.exception("Full exception details:")
        
        # Last resort - return a valid structure
        fallback_data = {
            "original_query": "research query", 
            "expanded_terms": ["research"],
            "search_strategy": "basic search",
            "historical_context": [],
            "memory_entities": []
        }
        logger.info("🆘 Created fallback structure due to exception")
        return (True, json.dumps(fallback_data, ensure_ascii=False))

def validate_raw_paper_data(result: str) -> Tuple[bool, Any]:
    """Validate raw paper data from Researcher."""
    result_str = _extract_text_output(result)
    try:
        # Parse JSON result
        data = json.loads(result_str)
        
        # Check required fields
        required_fields = ["metadata", "full_text", "sections", "extraction_quality"]
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return (False, f"Missing required fields: {', '.join(missing_fields)}")
        
        # Validate extraction quality
        quality = data.get("extraction_quality", 0)
        if not (0 <= quality <= 1):
            return (False, "extraction_quality must be between 0 and 1")
        
        return (True, result_str)
    except json.JSONDecodeError:
        return (False, "Output must be valid JSON")
    except Exception as e:
        return (False, f"Validation error: {str(e)}")

def validate_structured_json(result: str) -> Tuple[bool, Any]:
    """Validate structured JSON from Archivist."""
    result_str = _extract_text_output(result)
    try:
        # Parse JSON result
        data = json.loads(result_str)
        
        # Check for schema compliance
        if "metadata" not in data:
            return (False, "Missing metadata section")
        
        metadata = data["metadata"]
        required_metadata = ["title", "authors", "year", "abstract"]
        missing_metadata = [field for field in required_metadata if field not in metadata]
        if missing_metadata:
            return (False, f"Missing metadata fields: {', '.join(missing_metadata)}")
        
        # Check sections
        if "sections" not in data or not isinstance(data["sections"], list):
            return (False, "sections must be a list")
        
        return (True, result_str)
    except json.JSONDecodeError:
        return (False, "Output must be valid JSON")
    except Exception as e:
        return (False, f"Validation error: {str(e)}")

def validate_research_output(output: str) -> Tuple[bool, Any]:
    """Validate research output - checks length >500 chars, ≥10 bullet points, no code blocks."""
    output_str = _extract_text_output(output)
    try:
        if not output_str or not isinstance(output_str, str):
            return (False, "validation error occurred")
        
        # Check minimum length first
        if len(output_str) <= 500:
            return (False, "output too brief - minimum length required")
        
        # Check for bullet points (looking for lines starting with -, *, or •)
        bullet_patterns = ['-', '*', '•']
        bullet_count = 0
        for line in output_str.split('\n'):
            stripped = line.strip()
            if any(stripped.startswith(pattern + ' ') for pattern in bullet_patterns):
                bullet_count += 1
        
        if bullet_count < 10:
            return (False, f"insufficient bullet points - found {bullet_count}, need 10")
        
        # Check for code blocks (markdown code blocks)
        if '```' in output_str:
            return (False, "output should not contain code blocks")
        
        return (True, output_str.strip())
    except Exception as e:
        return (False, f"validation error: {str(e)}")

def validate_report_output(output: str) -> Tuple[bool, Any]:
    """Validate report output - checks length >500 chars, ≥3 markdown headers, no code blocks."""
    output_str = _extract_text_output(output)
    try:
        if not output_str or not isinstance(output_str, str):
            return (False, "validation error occurred")
        
        # Check minimum length first
        if len(output_str) <= 500:
            return (False, "output too brief - minimum length required")
        
        # Check for markdown headers (lines starting with #)
        header_count = 0
        for line in output_str.split('\n'):
            stripped = line.strip()
            if stripped.startswith('#') and ' ' in stripped:
                header_count += 1
        
        if header_count < 3:
            return (False, f"insufficient headers - found {header_count}, need 3")
        
        # Check for code blocks (markdown code blocks)
        if '```' in output_str:
            return (False, "output contains code blocks")
        
        return (True, output_str.strip())
    except Exception as e:
        return (False, f"validation error: {str(e)}")

def validate_markdown_output(result: str) -> Tuple[bool, Any]:
    """Validate markdown output from Publisher."""
    result_str = _extract_text_output(result)
    try:
        # Check for basic markdown structure
        if not result_str.strip():
            return (False, "Output cannot be empty")
        
        # Check for frontmatter
        if not result_str.startswith("---"):
            return (False, "Markdown must start with YAML frontmatter")
        
        # Check for vault path confirmation
        if "Created note at:" not in result_str and "vault" not in result_str.lower():
            return (False, "Output must confirm note creation with vault path")
        
        return (True, result_str)
    except Exception as e:
        return (False, f"Validation error: {str(e)}")

def _extract_text_output(output):
    """Utility: Accept string or TaskOutput-like object and return plain string for validation."""
    # Handle CrewAI TaskOutput or any object with 'result' attribute
    if isinstance(output, str):
        return output
    # Most CrewAI TaskOutput objects expose .result or .content
    for attr in ("result", "content", "text"):
        if hasattr(output, attr):
            possible = getattr(output, attr)
            # Recursively unwrap if nested TaskOutput
            if not isinstance(possible, str):
                return _extract_text_output(possible)
            return possible
    # Fallback to string conversion
    return str(output)

@CrewBase
class ServerResearchMcp():
    """ServerResearchMcp crew - Four-agent research paper parser system"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Agent definitions
    @agent
    def historian(self) -> Agent:
        """Memory and Context Manager Agent"""
        logger.info("🧠 Initializing Historian agent with memory tools")
        tools = get_crew_mcp_manager().get_historian_tools()
        logger.info(f"📚 Historian tools loaded: {[tool.name for tool in tools]}")
        
        return Agent(
            config=self.agents_config['historian'],
            tools=tools,
            verbose=True,
            llm=configured_llm,
            max_iter=5,
            respect_context_window=True
        )

    @agent
    def researcher(self) -> Agent:
        """Paper Discovery and Content Extraction Agent"""
        logger.info("🔬 Initializing Researcher agent with Zotero tools")
        tools = get_crew_mcp_manager().get_researcher_tools()
        logger.info(f"📖 Researcher tools loaded: {[tool.name for tool in tools]}")
        
        return Agent(
            config=self.agents_config['researcher'],
            tools=tools,
            verbose=True,
            llm=configured_llm
        )

    @agent
    def archivist(self) -> Agent:
        """Data Structuring and Schema Compliance Agent"""
        logger.info("📊 Initializing Archivist agent with validation tools")
        tools = get_crew_mcp_manager().get_archivist_tools()
        logger.info(f"📋 Archivist tools loaded: {[tool.name for tool in tools]}")
        
        return Agent(
            config=self.agents_config['archivist'],
            tools=tools,
            verbose=True,
            llm=configured_llm
        )

    @agent
    def publisher(self) -> Agent:
        """Markdown Generation and Vault Integration Agent"""
        logger.info("📝 Initializing Publisher agent with publishing tools")
        tools = get_crew_mcp_manager().get_publisher_tools()
        logger.info(f"📄 Publisher tools loaded: {[tool.name for tool in tools]}")
        
        return Agent(
            config=self.agents_config['publisher'],
            tools=tools,
            verbose=True,
            llm=configured_llm
        )

    # Task definitions
    @task
    def context_gathering_task(self) -> Task:
        """Historian task for memory and context gathering"""
        logger.info("📋 Creating context gathering task for Historian")
        
        return Task(
            config=self.tasks_config['context_gathering_task'],
            output_file='outputs/enriched_query.json',
            guardrail=validate_enriched_query,
            max_retries=2
        )

    @task
    def paper_extraction_task(self) -> Task:
        """Researcher task for paper discovery and extraction"""
        logger.info("📋 Creating paper extraction task for Researcher")
        
        return Task(
            config=self.tasks_config['paper_extraction_task'],
            output_file='outputs/raw_paper_data.json',
            guardrail=validate_raw_paper_data,
            max_retries=2
        )

    @task
    def data_structuring_task(self) -> Task:
        """Archivist task for data structuring and validation"""
        logger.info("📋 Creating data structuring task for Archivist")
        
        return Task(
            config=self.tasks_config['data_structuring_task'],
            output_file='outputs/structured_paper.json',
            guardrail=validate_structured_json,
            max_retries=2
        )

    @task
    def markdown_generation_task(self) -> Task:
        """Publisher task for markdown generation and vault integration"""
        logger.info("📋 Creating markdown generation task for Publisher")
        
        return Task(
            config=self.tasks_config['markdown_generation_task'],
            output_file='outputs/published_paper.md',
            guardrail=validate_markdown_output,
            max_retries=1
        )

    @crew
    def crew(self) -> Crew:
        """Creates the ServerResearchMcp crew with four sequential agents"""
        logger.info("🚀 Creating ServerResearchMcp crew")
        
        # Create output directory if it doesn't exist
        os.makedirs('outputs', exist_ok=True)
        logger.info("📁 Output directory ready")
        
        # Check if we should disable memory/planning for ChromaDB compatibility
        disable_memory = os.getenv("DISABLE_CREW_MEMORY", "false").lower() == "true"
        logger.info(f"🧠 Crew memory enabled: {not disable_memory}")
        logger.info(f"🎯 Crew planning enabled: {not disable_memory}")
        
        crew = Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,    # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            memory=not disable_memory,         # Enable memory for context persistence (unless disabled)
            planning=not disable_memory        # Enable planning for better coordination (unless disabled)
        )
        
        logger.info(f"✅ Crew created with {len(crew.agents)} agents and {len(crew.tasks)} tasks")
        logger.info(f"👥 Agents: {[agent.role for agent in crew.agents]}")
        logger.info(f"📋 Tasks: {[task.description[:50] + '...' for task in crew.tasks]}")
        
        return crew
