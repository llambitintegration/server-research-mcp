"""
Unified MCP Tools System for Server Research MCP
Provides extensible, plug-and-play tools for different agents and crews.
"""
from crewai.tools import BaseTool
from typing import Type, Dict, Any, List, Optional
from pydantic import BaseModel, Field, field_validator
import subprocess
import json
import os
import asyncio
import logging
from datetime import datetime

# Import MCP Manager and base classes
from .mcp_manager import get_mcp_manager
from .mcp_base_tool import MCPBaseTool
from .tool_factory import mcp_tool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Input Schemas - Define the contracts for each tool
# =============================================================================

class MemorySearchInput(BaseModel):
    """Input schema for Memory Search Tool."""
    query: str = Field(..., description="Search query to find relevant knowledge in memory graph")

    def to_mcp_format(self) -> dict:
        """Convert to MCP server expected format."""
        return {"query": str(self.query) if self.query else ""}

class MemoryCreateEntityInput(BaseModel):
    """Input schema for Memory Create Entity Tool.

    Supports two usage patterns:
    1. Single-entity mode: Provide ``name``, ``entity_type``, and ``observations``.
    2. Batch mode: Provide a list of entities via the ``entities`` field.
    At least one of these patterns must be satisfied.
    """
    # Single-entity fields (all optional)
    name: Optional[str] = Field(None, description="Name of the entity to create (single-entity mode)")
    entity_type: Optional[str] = Field(None, description="Type of the entity (e.g., 'paper', 'author', 'topic', 'concept') in single-entity mode")
    observations: Optional[List[str]] = Field(None, description="List of observations about this entity in single-entity mode")

    # Batch mode field
    entities: Optional[List[dict]] = Field(None, description="List of entity dictionaries for batch creation")

    @field_validator("entities", mode="after")
    def _validate_either_single_or_batch(cls, v, info):  # noqa: N805
        data = info.data  # type: ignore[attr-defined]
        single_fields_provided = all(data.get(k) is not None for k in ("name", "entity_type", "observations"))
        batch_provided = v is not None

        if not batch_provided and not single_fields_provided:
            raise ValueError("Provide either 'entities' for batch mode or 'name', 'entity_type', and 'observations' for single-entity mode.")

        return v

    def to_mcp_format(self) -> dict:
        """Convert to MCP server expected format."""
        if self.entities is not None:
            return {"entities": self.entities}
        else:
            return {
                "name": self.name,
                "entity_type": self.entity_type,
                "observations": self.observations or []
            }

class MemoryAddObservationInput(BaseModel):
    """Input schema for Memory Add Observation Tool."""
    entity_name: str = Field(..., description="Name of the entity to add observations to")
    observations: List[str] = Field(..., description="List of new observations to add")

    def to_mcp_format(self) -> dict:
        """Convert to MCP server expected format."""
        return {
            "entityName": self.entity_name,
            "observations": self.observations
        }

class Context7ResolveInput(BaseModel):
    """Input schema for Context7 Library Resolution Tool."""
    library_name: str = Field(..., description="Library name to search for and resolve to Context7-compatible ID")

    def to_mcp_format(self) -> dict:
        """Convert to MCP server expected format."""
        return {"libraryName": self.library_name}

class Context7DocsInput(BaseModel):
    """Input schema for Context7 Documentation Tool."""
    context7_library_id: str = Field(..., description="Context7-compatible library ID (e.g., '/mongodb/docs')")
    topic: str = Field(default="", description="Topic to focus documentation on (optional)")
    tokens: int = Field(default=10000, description="Maximum tokens of documentation to retrieve")

    def to_mcp_format(self) -> dict:
        """Convert to MCP server expected format."""
        return {
            "context7CompatibleLibraryID": self.context7_library_id,
            "topic": self.topic,
            "tokens": max(self.tokens, 10000)  # Context7 enforces minimum 10000 tokens
        }

class ZoteroSearchInput(BaseModel):
    """Input schema for Zotero Search Tool."""
    query: str = Field(..., description="Search query for finding papers in Zotero")
    search_type: str = Field("everything", description="Search type: 'everything', 'title', 'author', 'tag'")
    limit: int = Field(10, description="Maximum number of results to return")

    def to_mcp_format(self) -> dict:
        """Convert to MCP server expected format."""
        return {
            "query": self.query,
            "searchType": self.search_type,
            "limit": self.limit
        }

class ZoteroExtractInput(BaseModel):
    """Input schema for Zotero Extract Tool."""
    item_key: str = Field(..., description="Zotero item key to extract")
    include_pdf: bool = Field(True, description="Whether to extract PDF content")
    include_annotations: bool = Field(True, description="Whether to include annotations")

    def to_mcp_format(self) -> dict:
        """Convert to MCP server expected format."""
        return {
            "key": self.item_key,
            "includePdf": self.include_pdf,
            "includeAnnotations": self.include_annotations
        }

class SequentialThinkingInput(BaseModel):
    """Input schema for Sequential Thinking Tool."""
    thought: str = Field(..., description="Current thinking step for multi-step reasoning")
    thought_number: int = Field(..., description="Current thought number in sequence")
    total_thoughts: int = Field(..., description="Estimated total thoughts needed")
    next_thought_needed: bool = Field(..., description="Whether another thought step is needed")

    def to_mcp_format(self) -> dict:
        """Convert to MCP server expected format."""
        return {
            "thought": self.thought,
            "thoughtNumber": self.thought_number,
            "totalThoughts": self.total_thoughts,
            "nextThoughtNeeded": self.next_thought_needed
        }

# =============================================================================
# Memory Tools (Historian Agent) - Using clean inheritance pattern
# =============================================================================

class MemorySearchTool(MCPBaseTool):
    name: str = "memory_search"
    description: str = (
        "Search the knowledge graph memory for relevant information. "
        "Use this to find existing knowledge about research topics, entities, and relationships."
    )
    server_name: str = "memory"
    mcp_tool_name: str = "search_nodes"
    args_schema: Type[BaseModel] = MemorySearchInput

class MemoryCreateEntityTool(MCPBaseTool):
    name: str = "memory_create_entity"
    description: str = (
        "Create new entities in the knowledge graph memory. Use this to store important research topics, sources, and concepts. "
        "Accepts either single-entity fields (name, entity_type, observations) or a batch via the 'entities' list."
    )
    server_name: str = "memory"
    mcp_tool_name: str = "create_entities"
    args_schema: Type[BaseModel] = MemoryCreateEntityInput

class MemoryAddObservationTool(MCPBaseTool):
    name: str = "memory_add_observation"
    description: str = (
        "Add new observations to existing entities in the knowledge graph memory. "
        "Use this to update entities with new information discovered during research."
    )
    server_name: str = "memory"
    mcp_tool_name: str = "add_observations"
    args_schema: Type[BaseModel] = MemoryAddObservationInput

# =============================================================================
# Context7 Tools (Available for future agents) - Using factory pattern
# =============================================================================

@mcp_tool(
    tool_name="context7_resolve_library",
    server="context7",
    mcp_method="resolve-library-id",
    schema=Context7ResolveInput,
    description="Resolve a library/package name to a Context7-compatible library ID. Use this to find library documentation for research topics."
)
class Context7ResolveTool:
    """Context7 library resolution tool."""
    pass

@mcp_tool(
    tool_name="context7_get_docs",
    server="context7",
    mcp_method="get-library-docs",
    schema=Context7DocsInput,
    description="Fetch up-to-date documentation for a library using Context7. Use this to get detailed technical documentation for research."
)
class Context7DocsTool:
    """Context7 documentation retrieval tool."""
    pass

# =============================================================================
# Research Tools (For future Researcher agent)
# =============================================================================

@mcp_tool(
    tool_name="zotero_search",
    server="zotero",
    mcp_method="search",
    schema=ZoteroSearchInput,
    description="Search Zotero library for research papers using various criteria. Returns list of matching papers with metadata."
)
class ZoteroSearchTool:
    """Zotero search tool."""
    pass

@mcp_tool(
    tool_name="zotero_extract",
    server="zotero", 
    mcp_method="get_item",
    schema=ZoteroExtractInput,
    description="Extract full content from a Zotero item including PDF text, metadata, and annotations. Returns comprehensive paper data."
)
class ZoteroExtractTool:
    """Zotero extraction tool."""
    pass

@mcp_tool(
    tool_name="sequential_thinking",
    server="sequential-thinking",
    mcp_method="append_thought",
    schema=SequentialThinkingInput,
    description="Use structured multi-step reasoning for complex analysis. Break down complex research questions into manageable thinking steps."
)
class SequentialThinkingTool:
    """Sequential thinking tool."""
    pass

# =============================================================================
# Legacy Tools (Preserved for backward compatibility)
# =============================================================================

class SchemaValidationInput(BaseModel):
    """Input schema for Schema Validation Tool."""
    data: Dict[str, Any] = Field(..., description="Data to validate against schema")
    schema_type: str = Field("research_paper", description="Schema type to validate against")

class IntelligentSummaryInput(BaseModel):
    """Input schema for Intelligent Summary Tool."""
    content: str = Field(..., description="Content to summarize")
    max_length: int = Field(500, description="Maximum summary length in characters")
    preserve_technical: bool = Field(True, description="Whether to preserve technical terms")

class FileSystemReadInput(BaseModel):
    """Input schema for FileSystem Read Tool."""
    file_path: str = Field(..., description="Path to file to read")

class FileSystemWriteInput(BaseModel):
    """Input schema for FileSystem Write Tool."""
    file_path: str = Field(..., description="Path where to write file")
    content: str = Field(..., description="Content to write to file")
    create_dirs: bool = Field(True, description="Whether to create directories if they don't exist")

class ObsidianCreateNoteInput(BaseModel):
    """Input schema for Obsidian Create Note Tool."""
    title: str = Field(..., description="Note title")
    content: str = Field(..., description="Note content in markdown")
    folder: str = Field("Papers", description="Folder within vault")
    tags: List[str] = Field(default_factory=list, description="Tags to add")

class ObsidianLinkGeneratorInput(BaseModel):
    """Input schema for Obsidian Link Generator Tool."""
    source_note: str = Field(..., description="Source note title")
    target_notes: List[str] = Field(..., description="Target notes to link to")
    link_type: str = Field("related", description="Type of link: 'related', 'cites', 'cited_by'")

class ObsidianTemplateInput(BaseModel):
    """Input schema for Obsidian Template Tool."""
    template_name: str = Field("research_paper", description="Template to use")
    data: Dict[str, Any] = Field(..., description="Data to fill template with")

# Legacy tool implementations (simplified)
class SchemaValidationTool(BaseTool):
    name: str = "schema_validation"
    description: str = "Validate data against research paper JSON schema. Ensures all required fields are present and properly formatted."
    args_schema: Type[BaseModel] = SchemaValidationInput

    def _run(self, data: Dict[str, Any], schema_type: str = "research_paper") -> str:
        """Validate data against schema."""
        try:
            from ..schemas import ResearchPaperSchema
            if schema_type == "research_paper":
                validated = ResearchPaperSchema(**data)
                return json.dumps({
                    "valid": True,
                    "data": validated.model_dump(),
                    "errors": []
                }, default=str)
            else:
                return json.dumps({
                    "valid": False,
                    "errors": [f"Unknown schema type: {schema_type}"],
                    "data": None
                })
        except Exception as e:
            logger.error(f"Schema validation error: {e}")
            return json.dumps({
                "valid": False,
                "errors": [str(e)],
                "data": None
            })

# =============================================================================
# Tool Collections by Agent/Purpose - Plug-and-Play Interface
# =============================================================================

def get_historian_tools() -> List[BaseTool]:
    """Get memory-focused tools for the Historian agent."""
    return [
        MemorySearchTool(),
        MemoryCreateEntityTool(),
        MemoryAddObservationTool()
    ]

def get_context7_tools() -> List[BaseTool]:
    """Get Context7 tools for documentation and library research."""
    return [
        Context7ResolveTool(),
        Context7DocsTool()
    ]

def get_researcher_tools() -> List[BaseTool]:
    """Get research-focused tools for the Researcher agent."""
    return [
        ZoteroSearchTool(),
        ZoteroExtractTool()
    ]

def get_archivist_tools() -> List[BaseTool]:
    """Get data structuring tools for the Archivist agent."""
    return [
        SequentialThinkingTool(),
        SchemaValidationTool()
    ]

def get_publisher_tools() -> List[BaseTool]:
    """Return tools for the Publisher agent (none defined yet, kept for compatibility)."""
    return []

def get_all_mcp_tools() -> Dict[str, List[BaseTool]]:
    """Get all tools organized by category for easy access."""
    return {
        "historian": get_historian_tools(),
        "context7": get_context7_tools(),
        "researcher": get_researcher_tools(),
        "archivist": get_archivist_tools(),
        "publisher": get_publisher_tools(),
    }

# Backward compatibility
historian_mcp_tools = get_historian_tools() 