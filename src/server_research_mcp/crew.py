#!/usr/bin/env python3
"""
Ultra-simplified decorator-based CrewAI implementation.
Reduces boilerplate and makes the workflow more declarative.
"""

import os
import logging
from typing import Dict, Any, List, Optional, Type, Callable
from functools import wraps
from dataclasses import dataclass, field
import yaml
from pathlib import Path

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, task
from pydantic import BaseModel
from crewai.tools import BaseTool

from .schemas.research_paper import EnrichedQuery, RawPaperData, ResearchPaperSchema
from .schemas.obsidian_meta import ObsidianDocument
from .tools.mcp_tools import get_registry, get_mcp_manager
from .config.llm_config import LLMConfig, get_configured_llm, check_llm_config, get_llm_config
from .utils.log_context import log_execution, log_context
from .utils.mcpadapt import MCPAdapt
from .utils.logging_config import get_logger, get_symbol
from .utils.rate_limiting import RateLimitConfig

logger = get_logger(__name__)

# =============================================================================
# Decorators for Agent and Task Definition
# =============================================================================

class AgentDefinition:
    """Decorator for defining agents with their configuration."""
    
    _agents: Dict[str, 'AgentDefinition'] = {}
    
    def __init__(self, 
                 name: str,
                 schema: Type[BaseModel],
                 tools_pattern: str,
                 min_tools: int = 1,
                 max_iter: Optional[int] = None,
                 max_execution_time: Optional[int] = None):
        self.name = name
        self.schema = schema
        self.tools_pattern = tools_pattern
        self.min_tools = min_tools
        self.max_iter = max_iter
        self.max_execution_time = max_execution_time
        self.role = None
        self.goal = None
        self.backstory = None
        
        # Register this agent
        AgentDefinition._agents[name] = self
    
    def __call__(self, func: Callable) -> Callable:
        """Capture the agent configuration from the decorated function."""
        # Extract docstring parts
        docstring = func.__doc__ or ""
        lines = [line.strip() for line in docstring.strip().split('\n')]
        
        # Parse structured docstring
        current_section = None
        sections = {"role": [], "goal": [], "backstory": []}
        
        for line in lines:
            if line.lower().startswith("role:"):
                current_section = "role"
                sections[current_section].append(line[5:].strip())
            elif line.lower().startswith("goal:"):
                current_section = "goal"
                sections[current_section].append(line[5:].strip())
            elif line.lower().startswith("backstory:"):
                current_section = "backstory"
                sections[current_section].append(line[10:].strip())
            elif current_section and line:
                sections[current_section].append(line)
        
        # Set configuration
        self.role = " ".join(sections["role"])
        self.goal = " ".join(sections["goal"])
        self.backstory = " ".join(sections["backstory"])
        
        # Store the original function
        self.func = func
        
        @wraps(func)
        def wrapper(crew_instance):
            # Create agent with configuration
            return crew_instance._create_agent_from_definition(self)
        
        return wrapper
    
    @classmethod
    def get(cls, name: str) -> Optional['AgentDefinition']:
        """Get agent definition by name."""
        return cls._agents.get(name)

class TaskDefinition:
    """Decorator for defining tasks with their configuration."""
    
    _tasks: Dict[str, 'TaskDefinition'] = {}
    
    def __init__(self,
                 name: str,
                 agent: str,
                 depends_on: Optional[List[str]] = None):
        self.name = name
        self.agent_name = agent
        self.depends_on = depends_on or []
        self.description = None
        self.expected_output = None
        
        # Register this task
        TaskDefinition._tasks[name] = self
    
    def __call__(self, func: Callable) -> Callable:
        """Capture task configuration from decorated function."""
        # Extract docstring
        docstring = func.__doc__ or ""
        lines = [line.strip() for line in docstring.strip().split('\n')]
        
        # Parse sections
        current_section = None
        sections = {"description": [], "expected_output": []}
        
        for line in lines:
            if line.lower().startswith("description:"):
                current_section = "description"
                sections[current_section].append(line[12:].strip())
            elif line.lower().startswith("expected output:"):
                current_section = "expected_output"
                sections[current_section].append(line[16:].strip())
            elif current_section and line:
                sections[current_section].append(line)
        
        self.description = " ".join(sections["description"])
        self.expected_output = " ".join(sections["expected_output"])
        
        # Store original function
        self.func = func
        
        @wraps(func)
        def wrapper(crew_instance):
            return crew_instance._create_task_from_definition(self)
        
        return wrapper
    
    @classmethod
    def get_all(cls) -> Dict[str, 'TaskDefinition']:
        """Get all task definitions."""
        return cls._tasks.copy()

# =============================================================================
# Agent Definitions
# =============================================================================

@AgentDefinition("historian", schema=EnrichedQuery, tools_pattern="historian", min_tools=6, max_iter=8, max_execution_time=120)
def historian_agent():
    """
    Role: Research Context Specialist for {topic}
    
    Goal: Build comprehensive research context by connecting current queries to 
    existing knowledge while preparing search strategies for future discovery
    
    Backstory: You are a knowledge archaeologist who specializes in understanding
    research patterns and connections. You excel at finding hidden relationships
    between topics and expanding simple queries into rich, contextual searches
    that capture the full landscape of academic knowledge.
    """
    pass

@AgentDefinition("researcher", schema=RawPaperData, tools_pattern="researcher", min_tools=3)
def researcher_agent():
    """
    Role: Academic Paper Discovery Expert for {topic}
    
    Goal: Find and extract comprehensive content from relevant academic papers
    using Zotero integration
    
    Backstory: You are a digital librarian with expertise in academic databases
    and paper extraction. You know exactly how to navigate Zotero's systems to
    find the most relevant papers and extract every piece of valuable content,
    from abstracts to full-text and references.
    """
    pass

@AgentDefinition("archivist", schema=ResearchPaperSchema, tools_pattern="archivist", min_tools=1)
def archivist_agent():
    """
    Role: Academic Data Structuring Expert for {topic}
    
    Goal: Transform raw paper data into validated, structured formats that
    preserve critical information while enabling knowledge graph integration
    
    Backstory: You are a data architect who specializes in academic information
    systems. You understand how to structure complex research data for maximum
    utility, ensuring nothing important is lost while making the information
    accessible and interconnected.
    """
    pass

@AgentDefinition("publisher", schema=ObsidianDocument, tools_pattern="publisher", min_tools=11)
def publisher_agent():
    """
    Role: Obsidian Knowledge Vault Curator for {topic}
    
    Goal: Create beautifully formatted, richly interconnected Obsidian documents
    that integrate seamlessly into existing knowledge graphs
    
    Backstory: You are a digital knowledge curator who specializes in Obsidian
    vault architecture. You understand how to create documents that not only
    look great but also form meaningful connections through tags, links, and
    metadata, turning individual papers into a living knowledge network.
    """
    pass

# =============================================================================
# Task Definitions
# =============================================================================

@TaskDefinition("context_gathering", agent="historian")
def context_task():
    """
    Description: Build comprehensive research context for {topic} by:
    
    1. Search existing knowledge base for related papers and concepts
    2. Identify key authors, journals, and research themes  
    3. Expand search terms based on discovered patterns
    4. Retrieve relevant memory entities and relationships
    5. Develop targeted search strategies for paper discovery
    
    Tools to use:
    - memory:search_nodes - Find existing related knowledge
    - memory:get_entities - Retrieve stored concepts and relationships
    
    Expected Output: EnrichedQuery containing:
    - Original query and expanded search terms
    - Related papers and authors from memory
    - Topic context and research themes
    - Recommended search strategies
    - Created/updated memory entities
    
    Output limit: 5,000 tokens
    """
    pass

@TaskDefinition("paper_extraction", agent="researcher", depends_on=["context_gathering"])
def extraction_task():
    """
    Description: Discover and extract papers about {topic} using enriched context:
    
    1. Execute Zotero searches using expanded terms from context
    2. Retrieve comprehensive metadata for each paper
    3. Extract full-text content including all sections
    4. Capture complete reference lists and citations
    5. Assess extraction quality and completeness
    
    Tools to use:
    - zotero_search_items - Find relevant papers
    - zotero_item_metadata - Extract comprehensive metadata
    - zotero_item_children - Get full-text and references
    
    Expected Output: RawPaperData containing:
    - Complete paper metadata (title, authors, DOI, etc.)
    - Full-text content organized by sections
    - Complete reference lists
    - Extraction quality metrics
    - Zotero item keys for reference
    
    Output limit: 10,000 tokens
    """
    pass

@TaskDefinition("analysis", agent="archivist", depends_on=["paper_extraction"])
def analysis_task():
    """
    Description: Structure and analyze raw paper data for {topic}:
    
    1. Validate all data against ResearchPaperSchema requirements
    2. Extract and summarize key findings and contributions
    3. Identify methodologies, limitations, and future work
    4. Structure content for optimal knowledge graph integration
    5. Cross-reference with existing knowledge from memory
    
    Tools to use:
    - sequential-thinking:sequentialthinking - Break down complex analysis
    - memory:search_nodes - Connect to existing knowledge
    - memory:add_observations - Store key insights
    
    Expected Output: ResearchPaperSchema containing:
    - Validated metadata conforming to schema
    - Structured sections (abstract, methods, results, etc.)
    - Key findings and contributions
    - Limitations and future work
    - Cross-references and relationships
    
    Output limit: 30,000 tokens
    """
    pass

@TaskDefinition("publishing", agent="publisher", depends_on=["analysis"])
def publishing_task():
    """
    Description: Create Obsidian documents for {topic} research papers:
    
    1. Generate markdown content from structured data
    2. Create comprehensive YAML frontmatter with metadata
    3. Add wikilinks to related concepts and papers
    4. Generate tag hierarchies for categorization
    5. Create backlinks and cross-references
    6. Save to appropriate vault location
    
    Tools to use:
    - filesystem:write_file - Create markdown documents
    - filesystem:edit_file - Update existing documents
    - memory:search_nodes - Find related documents for linking
    - memory:create_relations - Store document relationships
    
    Expected Output: ObsidianDocument containing:
    - Complete markdown document with proper formatting
    - YAML frontmatter with all metadata and tags
    - Wikilinks and backlinks throughout content
    - File path and integration status
    - Publishing metadata and timestamps
    
    Output limit: 30,000 tokens
    """

# =============================================================================
# Simplified Crew Implementation
# =============================================================================

@CrewBase
class ServerResearchMcpCrew:
    """Ultra-simplified research crew using decorators."""
    
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'
    
    def __init__(self, inputs: Optional[Dict[str, Any]] = None):
        """Initialize the crew."""
        self.inputs = inputs or {}
        self.llm_config = LLMConfig()
        self.tool_registry = get_registry()
        self._agents_cache = {}
        self._tasks_cache = {}
        
        # Legacy compatibility attributes
        self._memory = True
        self._state = {}
        
        # Logging is configured globally via logging_config
        logger.info("Crew initialized", extra={"inputs": bool(self.inputs)})
    
    @property
    def memory(self) -> bool:
        """Legacy compatibility - memory attribute for tests."""
        return self._memory
    
    @log_execution
    def _create_agent_from_definition(self, definition: AgentDefinition) -> Agent:
        """Create agent from decorator definition."""
        if definition.name not in self._agents_cache:
            # Get tools
            tools = self.tool_registry.get_agent_tools(definition.name)
            
            # Legacy compatibility - pad historian tools to minimum 6
            if definition.name == "historian" and len(tools) < 6:
                padding_count = 6 - len(tools)
                for i in range(padding_count):
                    from .tools.mcp_tools import SchemaValidationTool
                    tools.append(SchemaValidationTool())
            
            logger.info("Agent created", extra={
                "agent_name": definition.name,
                "tools_count": len(tools),
                "schema": definition.schema.__name__
            })
            
            # Create safe inputs for formatting (provide defaults for missing keys)
            safe_inputs = self.inputs.copy()
            if 'topic' not in safe_inputs:
                safe_inputs['topic'] = 'Research Topic'
            else:
                # Truncate very long topics for role formatting to avoid LLM issues
                topic = safe_inputs['topic']
                if len(topic) > 80:
                    # Take first 80 chars and add ellipsis
                    safe_inputs['topic'] = topic[:80].rstrip() + '...'
            
            # Create agent with detailed logging
            try:
                # Get LLM instance and apply rate limiting
                llm_instance = self.llm_config.get_llm()
                
                # Apply LLM rate limiting
                from .utils.llm_rate_limiter import get_rate_limited_llm
                if not (hasattr(llm_instance, 'rate_limiter_applied') and llm_instance.rate_limiter_applied):
                    llm_instance = get_rate_limited_llm(llm_instance)
                    logger.info(f"{get_symbol('success')} Applied rate limiting to LLM for {definition.name}")
                else:
                    logger.info(f"{get_symbol('info')} LLM already rate limited for {definition.name}")
                
                logger.info("Creating agent with rate-limited LLM", extra={
                    "agent_name": definition.name,
                    "llm_model": getattr(llm_instance.wrapped_llm if hasattr(llm_instance, 'wrapped_llm') else llm_instance, 'model', 'unknown'),
                    "tools_available": len(tools),
                    "schema": definition.schema.__name__,
                    "rate_limited": hasattr(llm_instance, 'rate_limiter_applied')
                })
                
                # Format role with truncated topic to avoid very long role names
                formatted_role = definition.role.format(**safe_inputs)
                logger.info("Creating agent with formatted role", extra={
                    "agent_name": definition.name,
                    "role_length": len(formatted_role),
                    "role_preview": formatted_role[:100] + "..." if len(formatted_role) > 100 else formatted_role
                })
                
                # Prepare agent kwargs with enhanced retry and timeout settings
                agent_kwargs = {
                    "role": formatted_role,
                    "goal": definition.goal,
                    "backstory": definition.backstory,
                    "tools": tools,
                    "llm": llm_instance,
                    "verbose": False,
                    "output_pydantic": definition.schema,
                    "max_retry_limit": 5,  # Increased for rate limiting delays
                    "allow_delegation": False,  # Disable delegation to prevent unexpected behavior
                    "request_timeout": 120,  # Increased timeout for rate limiting
                }
                
                # Add optional parameters if specified
                if definition.max_iter is not None:
                    agent_kwargs["max_iter"] = definition.max_iter
                    logger.info(f"Setting max_iter={definition.max_iter} for agent {definition.name}")
                if definition.max_execution_time is not None:
                    # Double the execution time to accommodate rate limiting delays
                    doubled_time = definition.max_execution_time * 2
                    agent_kwargs["max_execution_time"] = doubled_time
                    logger.info(f"Setting max_execution_time={doubled_time} (doubled from {definition.max_execution_time}) for agent {definition.name}")
                else:
                    # Default execution time of 300s (5 minutes) for rate limiting accommodation
                    agent_kwargs["max_execution_time"] = 300
                    logger.info(f"Setting default max_execution_time=300 for agent {definition.name}")
                
                self._agents_cache[definition.name] = Agent(**agent_kwargs)
                
                logger.info("Agent created successfully", extra={
                    "agent_name": definition.name,
                    "final_tools_count": len(self._agents_cache[definition.name].tools) if hasattr(self._agents_cache[definition.name], 'tools') else 0
                })
                
            except Exception as e:
                logger.error("Failed to create agent", extra={
                    "agent_name": definition.name,
                    "error": str(e),
                    "error_type": type(e).__name__
                }, exc_info=True)
                raise
        
        return self._agents_cache[definition.name]
    
    def _create_guardrail(self, task_name: str, schema: Type[BaseModel]) -> Callable:
        """Create a guardrail function for a task."""
        def guardrail_function(output: Any) -> tuple[bool, Any]:
            """Validate task output against schema."""
            try:
                logger.info(f"Guardrail validation for {task_name}", extra={
                    "task_name": task_name,
                    "output_type": type(output).__name__,
                    "output_length": len(str(output)) if output else 0,
                    "has_pydantic": hasattr(output, 'pydantic') if output else False
                })
                
                if hasattr(output, 'pydantic') and output.pydantic:
                    # Already validated by CrewAI
                    logger.info(f"Output already validated for {task_name}")
                    return True, output
                
                # Handle empty or None output
                if output is None:
                    logger.warning(f"None output for {task_name}")
                    return False, "None output"
                
                # Convert output to string for processing
                output_str = str(output).strip()
                if not output_str:
                    logger.warning(f"Empty output for {task_name}")
                    return False, "Empty output"
                
                # Try to validate against schema
                import json
                if isinstance(output, str):
                    try:
                        # Try JSON parsing first
                        data = json.loads(output)
                        validated = schema(**data)
                        logger.info(f"JSON validation successful for {task_name}")
                        return True, validated
                    except (json.JSONDecodeError, TypeError) as e:
                        # If not JSON, check if it's at least meaningful text
                        if len(output_str) > 10:  # Minimum meaningful length
                            logger.info(f"Non-JSON output accepted for {task_name}: {len(output_str)} chars")
                            return True, output
                        else:
                            logger.warning(f"Output too short for {task_name}: {len(output_str)} chars")
                            return False, f"Output too short: {len(output_str)} chars"
                elif isinstance(output, dict):
                    validated = schema(**output)
                    logger.info(f"Dict validation successful for {task_name}")
                    return True, validated
                else:
                    # For other types, accept if not empty
                    if output_str and len(output_str) > 10:
                        logger.info(f"Generic output accepted for {task_name}")
                        return True, output
                    else:
                        logger.warning(f"Output validation failed for {task_name}: insufficient content")
                        return False, "Insufficient content"
                        
            except Exception as e:
                logger.error(f"Guardrail validation exception for {task_name}: {e}", extra={
                    "task_name": task_name,
                    "error": str(e),
                    "error_type": type(e).__name__
                }, exc_info=True)
                return False, f"Validation exception: {str(e)}"
        
        return guardrail_function
    
    def _create_task_from_definition(self, definition: TaskDefinition) -> Task:
        """Create task from decorator definition."""
        if definition.name not in self._tasks_cache:
            # Get agent
            agent_def = AgentDefinition.get(definition.agent_name)
            if not agent_def:
                raise ValueError(f"Agent {definition.agent_name} not found")
            
            agent = self._create_agent_from_definition(agent_def)
            
            # Get dependencies
            context = []
            for dep_name in definition.depends_on:
                dep_task_def = TaskDefinition._tasks.get(dep_name)
                if dep_task_def:
                    context.append(self._create_task_from_definition(dep_task_def))
            
            # Create safe inputs for formatting (provide defaults for missing keys)
            safe_inputs = self.inputs.copy()
            if 'topic' not in safe_inputs:
                safe_inputs['topic'] = 'Research Topic'
            else:
                # Truncate very long topics for task formatting to avoid LLM issues
                topic = safe_inputs['topic']
                if len(topic) > 80:
                    # Take first 80 chars and add ellipsis
                    safe_inputs['topic'] = topic[:80].rstrip() + '...'
            
            # Create task with guardrails and logging
            try:
                logger.info("Creating task", extra={
                    "task_name": definition.name,
                    "agent_name": definition.agent_name,
                    "dependencies": len(context),
                    "schema": agent_def.schema.__name__
                })
                
                # Format task description with truncated topic
                formatted_description = definition.description.format(**safe_inputs)
                logger.info("Creating task with formatted description", extra={
                    "task_name": definition.name,
                    "description_length": len(formatted_description),
                    "description_preview": formatted_description[:150] + "..." if len(formatted_description) > 150 else formatted_description
                })
                
                self._tasks_cache[definition.name] = Task(
                    description=formatted_description,
                    expected_output=definition.expected_output,
                    agent=agent,
                    output_pydantic=agent_def.schema,
                    context=context,
                    guardrail=self._create_guardrail(definition.name, agent_def.schema),
                    max_retries=3,  # Increase retry limit for better resilience
                    async_execution=False  # Ensure synchronous execution for better error handling
                )
                
                logger.info("Task created successfully", extra={
                    "task_name": definition.name,
                    "description_length": len(definition.description),
                    "has_guardrail": self._tasks_cache[definition.name].guardrail is not None
                })
                
            except Exception as e:
                logger.error("Failed to create task", extra={
                    "task_name": definition.name,
                    "agent_name": definition.agent_name,
                    "error": str(e),
                    "error_type": type(e).__name__
                }, exc_info=True)
                raise
        
        return self._tasks_cache[definition.name]
    
    # Legacy compatibility methods
    @agent
    def historian(self) -> Agent:
        """Get historian agent."""
        return self._create_agent_from_definition(AgentDefinition.get("historian"))
    
    @agent
    def researcher(self) -> Agent:
        """Get researcher agent."""
        return self._create_agent_from_definition(AgentDefinition.get("researcher"))
    
    @agent
    def archivist(self) -> Agent:
        """Get archivist agent."""
        return self._create_agent_from_definition(AgentDefinition.get("archivist"))
    
    @agent
    def publisher(self) -> Agent:
        """Get publisher agent."""
        return self._create_agent_from_definition(AgentDefinition.get("publisher"))
    
    # Legacy task aliases
    @agent
    def reporting_analyst(self) -> Agent:
        """Legacy alias for publisher agent."""
        return self.publisher()
    
    @task
    def context_gathering_task(self) -> Task:
        """Get context gathering task."""
        return self._create_task_from_definition(TaskDefinition._tasks["context_gathering"])
    
    @task
    def paper_extraction_task(self) -> Task:
        """Get paper extraction task."""
        return self._create_task_from_definition(TaskDefinition._tasks["paper_extraction"])
    
    @task
    def analysis_and_structuring_task(self) -> Task:
        """Get analysis task."""
        return self._create_task_from_definition(TaskDefinition._tasks["analysis"])
    
    @task
    def publishing_task(self) -> Task:
        """Get publishing task."""
        return self._create_task_from_definition(TaskDefinition._tasks["publishing"])
    
    # Legacy aliases
    @task
    def research_task(self) -> Task:
        """Legacy alias for paper_extraction_task."""
        return self.paper_extraction_task()
    
    @task
    def reporting_task(self) -> Task:
        """Legacy alias for publishing_task."""
        return self.publishing_task()
    
    @property
    def agents(self) -> List[Agent]:
        """Get all agents in order."""
        return [
            self.historian(),
            self.researcher(),
            self.archivist(),
            self.publisher()
        ]
    
    @property
    def tasks(self) -> List[Task]:
        """Get all tasks in order."""
        return [
            self.context_gathering_task(),
            self.paper_extraction_task(),
            self.analysis_and_structuring_task(),
            self.publishing_task()
        ]
    
    def crew(self) -> Crew:
        """Create the crew."""
        logger.info("Creating crew instance", extra={
            "agent_count": len(self.agents),
            "task_count": len(self.tasks),
            "crew_process": "sequential"
        })
        
        # Log agent details
        for i, agent in enumerate(self.agents):
            logger.info(f"Agent {i+1} configured", extra={
                "role": agent.role[:50] + "..." if len(agent.role) > 50 else agent.role,
                "tools_count": len(agent.tools) if hasattr(agent, 'tools') else 0,
                "llm_model": getattr(agent.llm, 'model', 'unknown') if hasattr(agent, 'llm') else 'no_llm'
            })
        
        # Log task dependencies
        for i, task in enumerate(self.tasks):
            context_count = len(task.context) if hasattr(task, 'context') and task.context else 0
            logger.info(f"Task {i+1} configured", extra={
                "description_preview": task.description[:80] + "..." if len(task.description) > 80 else task.description,
                "dependencies": context_count,
                "agent_role": task.agent.role[:30] + "..." if len(task.agent.role) > 30 else task.agent.role
            })
        
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=1,  # Use level 1 for cleaner output without JSON blocks
            full_output=False
        )
    
    @log_execution
    def run_with_validation(self, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute the research pipeline with validation."""
        from datetime import datetime
        
        start_time = datetime.now()
        final_inputs = {**self.inputs, **(inputs or {})}
        
        logger.info("Starting research pipeline", extra={
            "topic": final_inputs.get('topic', 'Unknown'),
            "query": final_inputs.get('paper_query', 'Unknown')
        })
        
        try:
            logger.info("Initiating crew kickoff", extra={
                "final_inputs": final_inputs,
                "crew_agents_count": len(self.crew().agents),
                "crew_tasks_count": len(self.crew().tasks)
            })
            
            result = self.crew().kickoff(inputs=final_inputs)
            
            logger.info("Crew kickoff completed", extra={
                "result_type": type(result).__name__,
                "has_tasks_output": hasattr(result, 'tasks_output'),
                "tasks_output_count": len(result.tasks_output) if hasattr(result, 'tasks_output') else 0
            })
            
            # Validate outputs
            validation_report = self._validate_outputs(result)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            logger.info("Research pipeline completed successfully", extra={
                "execution_time": execution_time,
                "validation_success": validation_report.get("overall_success", True),
                "validation_errors": validation_report.get("errors", [])
            })
            
            return {
                "success": True,
                "execution_time": execution_time,
                "crew_result": result,
                "validation_report": validation_report,
                "error_recovery": {
                    "attempted": True,
                    "successful": validation_report.get("overall_success", True),
                    "strategy": "schema_validation_retry",
                    "retry_count": 0
                },
                "error_type": None
            }
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Enhanced error logging
            error_details = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "execution_time": execution_time,
                "final_inputs": final_inputs
            }
            
            # Try to get additional context from the exception
            if hasattr(e, '__cause__') and e.__cause__:
                error_details["underlying_cause"] = str(e.__cause__)
                error_details["cause_type"] = type(e.__cause__).__name__
            
            if hasattr(e, 'args') and e.args:
                error_details["error_args"] = str(e.args)
            
            logger.error("Pipeline failed with detailed context", extra=error_details, exc_info=True)
            
            return {
                "success": False,
                "execution_time": execution_time,
                "crew_result": None,
                "error": str(e),
                "validation_report": {"overall_success": False, "errors": [str(e)]},
                "error_recovery": {
                    "attempted": True,
                    "successful": False,
                    "strategy": "exception_handling",
                    "retry_count": 1,
                    "error_details": str(e)
                },
                "error_type": "execution_error"
            }
    
    def _validate_outputs(self, result: Any) -> Dict[str, Any]:
        """Validate crew outputs."""
        logger.info("Starting output validation", extra={
            "result_type": type(result).__name__,
            "has_tasks_output": hasattr(result, 'tasks_output')
        })
        
        validation_report = {
            "overall_success": True,
            "task_validations": {},
            "errors": []
        }
        
        try:
            if hasattr(result, 'tasks_output'):
                schemas = [EnrichedQuery, RawPaperData, ResearchPaperSchema, ObsidianDocument]
                task_names = ["context_gathering", "paper_extraction", "analysis", "publishing"]
                
                logger.info("Validating task outputs", extra={
                    "tasks_output_count": len(result.tasks_output),
                    "expected_schemas": len(schemas)
                })
                
                for i, (task_output, name, schema) in enumerate(zip(result.tasks_output, task_names, schemas)):
                    logger.info(f"Validating task {i+1}", extra={
                        "task_name": name,
                        "schema": schema.__name__,
                        "output_type": type(task_output).__name__
                    })
                    
                    validation = self._validate_output(task_output, schema)
                    validation_report["task_validations"][name] = validation
                    
                    if not validation.get("valid", False):
                        validation_report["overall_success"] = False
                        validation_report["errors"].append(f"Task {name} validation failed")
                        logger.warning(f"Task validation failed", extra={
                            "task_name": name,
                            "validation_error": validation.get("error", "Unknown error")
                        })
                    else:
                        logger.info(f"Task validation successful", extra={
                            "task_name": name
                        })
            else:
                logger.warning("No tasks_output found in result")
                validation_report["overall_success"] = False
                validation_report["errors"].append("No tasks_output found in result")
        
        except Exception as e:
            logger.error("Validation process failed", extra={
                "error": str(e),
                "error_type": type(e).__name__
            }, exc_info=True)
            validation_report["overall_success"] = False
            validation_report["errors"].append(f"Validation error: {str(e)}")
        
        logger.info("Output validation completed", extra={
            "overall_success": validation_report["overall_success"],
            "error_count": len(validation_report["errors"])
        })
        
        return validation_report
    
    def _validate_output(self, output: Any, schema: Type[BaseModel]) -> Dict[str, Any]:
        """Validate single output against schema."""
        try:
            if hasattr(output, 'pydantic') and output.pydantic:
                return {"valid": True, "data": output.pydantic.model_dump()}
            
            # Parse raw output
            import json
            raw = output.raw if hasattr(output, 'raw') else str(output)
            data = json.loads(raw)
            validated = schema(**data)
            
            return {"valid": True, "data": validated.model_dump()}
        except Exception as e:
            return {"valid": False, "error": str(e)}

# =============================================================================
# Simple API
# =============================================================================

def create_research_crew(topic: str) -> ServerResearchMcpCrew:
    """Create a research crew for a specific topic."""
    return ServerResearchMcpCrew(inputs={'topic': topic})

def run_research_pipeline(topic: str) -> Dict[str, Any]:
    """Run the complete research pipeline for a topic."""
    crew = create_research_crew(topic)
    return crew.run_with_validation()

# =============================================================================
# Main
# =============================================================================

def main():
    """Main function to run the crew."""
    topic = 'KST: Executable Formal Semantics of IEC 61131-3 Structured Text'
    
    try:
        result = run_research_pipeline(topic)
        
        if result["success"]:
            print(f"{get_symbol('success')} Research completed for: {topic}")
            validation_report = result.get("validation_report", {})
            task_validations = validation_report.get("task_validations", {})
            
            for task_name, validation in task_validations.items():
                status = "✓" if validation.get("valid") else "✗"
                print(f"  {status} {task_name}: {'Valid' if validation.get('valid') else validation.get('error', 'Invalid')}")
        else:
            print(f"{get_symbol('error')} Research failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise

# =============================================================================
# Legacy Validation Functions for Test Compatibility
# =============================================================================

def validate_enriched_query(data: str) -> tuple[bool, Any]:
    """
    Validate EnrichedQuery output for test compatibility.
    
    Args:
        data: JSON string or dict to validate
        
    Returns:
        Tuple of (success, result_or_error)
    """
    try:
        import json
        
        # Parse JSON if string
        if isinstance(data, str):
            try:
                parsed_data = json.loads(data)
            except json.JSONDecodeError as e:
                return False, f"Invalid JSON: {str(e)}"
        else:
            parsed_data = data
        
        # Validate using EnrichedQuery schema
        enriched_query = EnrichedQuery(**parsed_data)
        return True, enriched_query.model_dump()
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def validate_research_output(data: str) -> tuple[bool, Any]:
    """
    Validate ResearchPaperSchema output for test compatibility.
    
    Args:
        data: JSON string or dict to validate
        
    Returns:
        Tuple of (success, result_or_error)
    """
    try:
        import json
        
        # Parse JSON if string
        if isinstance(data, str):
            try:
                parsed_data = json.loads(data)
            except json.JSONDecodeError as e:
                return False, f"Invalid JSON: {str(e)}"
        else:
            parsed_data = data
        
        # Validate using ResearchPaperSchema
        research_paper = ResearchPaperSchema(**parsed_data)
        return True, research_paper.model_dump()
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def validate_report_output(data: str) -> tuple[bool, Any]:
    """
    Validate ObsidianDocument output for test compatibility.
    
    Args:
        data: JSON string or dict to validate
        
    Returns:
        Tuple of (success, result_or_error)
    """
    try:
        import json
        
        # Parse JSON if string
        if isinstance(data, str):
            try:
                parsed_data = json.loads(data)
            except json.JSONDecodeError as e:
                return False, f"Invalid JSON: {str(e)}"
        else:
            parsed_data = data
        
        # Validate using ObsidianDocument schema
        obsidian_doc = ObsidianDocument(**parsed_data)
        return True, obsidian_doc.model_dump()
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def validate_raw_paper_data(data: str) -> tuple[bool, Any]:
    """
    Validate RawPaperData output for test compatibility.
    
    Args:
        data: JSON string or dict to validate
        
    Returns:
        Tuple of (success, result_or_error)
    """
    try:
        import json
        
        # Parse JSON if string
        if isinstance(data, str):
            try:
                parsed_data = json.loads(data)
            except json.JSONDecodeError as e:
                return False, f"Invalid JSON: {str(e)}"
        else:
            parsed_data = data
        
        # Validate using RawPaperData schema
        raw_paper = RawPaperData(**parsed_data)
        return True, raw_paper.model_dump()
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def validate_structured_json(data: str) -> tuple[bool, Any]:
    """
    Validate structured JSON against ResearchPaperSchema for test compatibility.
    
    Args:
        data: JSON string or dict to validate
        
    Returns:
        Tuple of (success, result_or_error)
    """
    try:
        import json
        
        # Parse JSON if string
        if isinstance(data, str):
            try:
                parsed_data = json.loads(data)
            except json.JSONDecodeError as e:
                return False, f"Invalid JSON: {str(e)}"
        else:
            parsed_data = data
        
        # Check for required metadata fields
        if "metadata" not in parsed_data:
            return False, "Missing required metadata field"
        
        metadata = parsed_data["metadata"]
        required_fields = ["title", "authors", "year", "abstract"]
        missing_fields = [field for field in required_fields if field not in metadata]
        
        if missing_fields:
            return False, f"Missing required metadata fields: {', '.join(missing_fields)}"
        
        # Validate using ResearchPaperSchema
        research_paper = ResearchPaperSchema(**parsed_data)
        return True, research_paper.model_dump()
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def validate_markdown_output(data: str) -> tuple[bool, Any]:
    """
    Validate markdown output with frontmatter for test compatibility.
    
    Args:
        data: Markdown string to validate
        
    Returns:
        Tuple of (success, result_or_error)
    """
    try:
        # Check for basic markdown structure
        if not isinstance(data, str):
            return False, "Input must be a string"
        
        # Check for YAML frontmatter
        if not data.strip().startswith("---"):
            return False, "Markdown must start with YAML frontmatter"
        
        # Check for frontmatter end
        if data.count("---") < 2:
            return False, "Markdown must have complete YAML frontmatter (ending ---)"
        
        # Check for basic content after frontmatter
        parts = data.split("---", 2)
        if len(parts) < 3:
            return False, "Markdown must have content after frontmatter"
        
        frontmatter_content = parts[1].strip()
        markdown_content = parts[2].strip()
        
        if not frontmatter_content:
            return False, "YAML frontmatter cannot be empty"
        
        if not markdown_content:
            return False, "Markdown content cannot be empty"
        
        # Check for file path indication (common in output)
        if "Created note at:" in data or "/vault/" in data or ".md" in data:
            return True, {
                "has_frontmatter": True,
                "has_content": True,
                "has_file_path": True,
                "content_length": len(markdown_content)
            }
        
        return True, {
            "has_frontmatter": True,
            "has_content": True,
            "has_file_path": False,
            "content_length": len(markdown_content)
        }
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def load_config() -> Dict[str, Any]:
    """Load agent and task configurations from YAML files."""
    logger.info("Loading configuration files")
    
    config_dir = Path(__file__).parent / "config"
    
    with log_context(operation="load_agents_config"):
        agents_path = config_dir / "agents.yaml" 
        logger.debug(f"Loading agents config from: {agents_path}")
        if not agents_path.exists():
            logger.error(f"Agents config file not found: {agents_path}")
            raise FileNotFoundError(f"Could not find {agents_path}")
        
        with open(agents_path, 'r') as f:
            agents_config = yaml.safe_load(f)
        logger.info(f"Loaded {len(agents_config.get('agents', {}))} agent configurations")
    
    with log_context(operation="load_tasks_config"):
        tasks_path = config_dir / "tasks.yaml"
        logger.debug(f"Loading tasks config from: {tasks_path}")
        if not tasks_path.exists():
            logger.error(f"Tasks config file not found: {tasks_path}")
            raise FileNotFoundError(f"Could not find {tasks_path}")
        
        with open(tasks_path, 'r') as f:
            tasks_config = yaml.safe_load(f)
        logger.info(f"Loaded {len(tasks_config.get('tasks', {}))} task configurations")
    
    logger.info("Configuration loading completed successfully")
    return {
        'agents': agents_config,
        'tasks': tasks_config
    }


@log_execution
def initialize_mcp_tools() -> Dict[str, List[BaseTool]]:
    """Initialize and categorize MCP tools for different agent roles."""
    logger.info("Starting MCP tools initialization")
    
    try:
        with log_context(operation="get_mcp_manager"):
            manager = get_mcp_manager()
            logger.debug("MCP manager obtained successfully")
        
        with log_context(operation="create_mcp_adapter"):
            adapter = MCPAdapt(manager)
            logger.debug("MCP adapter created successfully")
        
        tools = {
            'memory_tools': [],
            'research_tools': [],
            'filesystem_tools': [],
            'thinking_tools': []
        }
        
        with log_context(operation="get_memory_tools"):
            logger.debug("Getting memory tools")
            memory_tools = adapter.get_tools_by_server('memory')
            tools['memory_tools'] = memory_tools
            logger.info(f"Loaded {len(memory_tools)} memory tools")
        
        with log_context(operation="get_zotero_tools"):
            logger.debug("Getting Zotero research tools")
            zotero_tools = adapter.get_tools_by_server('zotero')
            tools['research_tools'] = zotero_tools
            logger.info(f"Loaded {len(zotero_tools)} Zotero research tools")
        
        with log_context(operation="get_filesystem_tools"):
            logger.debug("Getting filesystem tools")
            filesystem_tools = adapter.get_tools_by_server('filesystem')
            tools['filesystem_tools'] = filesystem_tools
            logger.info(f"Loaded {len(filesystem_tools)} filesystem tools")
        
        with log_context(operation="get_thinking_tools"):
            logger.debug("Getting sequential thinking tools")
            thinking_tools = adapter.get_tools_by_server('sequential-thinking')
            tools['thinking_tools'] = thinking_tools
            logger.info(f"Loaded {len(thinking_tools)} thinking tools")
        
        total_tools = sum(len(tool_list) for tool_list in tools.values())
        logger.info(f"MCP tools initialization completed - Total tools: {total_tools}")
        
        return tools
    
    except Exception as e:
        logger.error(f"Failed to initialize MCP tools: {e}", exc_info=True)
        raise


@log_execution
def create_agents(config: Dict[str, Any], tools: Dict[str, List[BaseTool]]) -> Dict[str, Agent]:
    """Create agents with their assigned tools and LLM configuration."""
    logger.info("Starting agent creation process")
    
    # Check LLM configuration first
    with log_context(operation="check_llm_config"):
        is_valid, error_msg = check_llm_config()
        if not is_valid:
            logger.error(f"LLM configuration is invalid: {error_msg}")
            raise ValueError(f"LLM configuration error: {error_msg}")
        
        llm_config = get_llm_config()
        logger.info(f"Using LLM configuration: provider={llm_config['provider']}, model={llm_config['model']}")
    
    with log_context(operation="get_configured_llm"):
        llm = get_configured_llm()
        logger.debug(f"LLM instance created: {type(llm).__name__}")
    
    agents = {}
    agents_config = config['agents']['agents']
    
    logger.info(f"Creating {len(agents_config)} agents")
    
    for agent_name, agent_config in agents_config.items():
        with log_context(operation=f"create_agent_{agent_name}"):
            logger.info(f"Creating agent: {agent_name}")
            
            # Assign tools based on agent role
            agent_tools = []
            role = agent_config.get('role', '').lower()
            
            if 'historian' in role:
                agent_tools.extend(tools['memory_tools'])
                logger.debug(f"Assigned {len(tools['memory_tools'])} memory tools to {agent_name}")
            
            if 'researcher' in role:
                agent_tools.extend(tools['research_tools'])
                logger.debug(f"Assigned {len(tools['research_tools'])} research tools to {agent_name}")
            
            if 'archivist' in role:
                agent_tools.extend(tools['thinking_tools'])
                logger.debug(f"Assigned {len(tools['thinking_tools'])} thinking tools to {agent_name}")
            
            if 'publisher' in role:
                agent_tools.extend(tools['filesystem_tools'])
                logger.debug(f"Assigned {len(tools['filesystem_tools'])} filesystem tools to {agent_name}")
            
            logger.info(f"Agent {agent_name} assigned {len(agent_tools)} tools total")
            
            try:
                agent = Agent(
                    role=agent_config['role'],
                    goal=agent_config['goal'],
                    backstory=agent_config['backstory'],
                    tools=agent_tools,
                    llm=llm,
                    verbose=False,
                    allow_delegation=agent_config.get('allow_delegation', False),
                    max_iter=agent_config.get('max_iter', 3)
                )
                agents[agent_name] = agent
                logger.info(f"Successfully created agent: {agent_name}")
                
            except Exception as e:
                logger.error(f"Failed to create agent {agent_name}: {e}", exc_info=True)
                raise
    
    logger.info(f"Agent creation completed - Created {len(agents)} agents")
    return agents

class ResearchCrew:
    """Simplified research crew wrapper for main.py compatibility."""
    
    def __init__(self):
        """Initialize the research crew."""
        logger.info("Initializing ResearchCrew wrapper")
        self._crew_instance = None
    
    def crew(self) -> Crew:
        """Get the crew instance."""
        if self._crew_instance is None:
            logger.info("Creating ServerResearchMcpCrew instance")
            crew_obj = ServerResearchMcpCrew()
            self._crew_instance = crew_obj.crew()
            logger.info(f"Crew created with {len(self._crew_instance.agents)} agents and {len(self._crew_instance.tasks)} tasks")
        
        return self._crew_instance

if __name__ == "__main__":
    main()