"""
Example: Adding a new MCP server (vector-db) with the refactored pattern.

This example shows how to add support for a hypothetical vector database
MCP server in just a few lines of code.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from .tool_factory import mcp_tool, create_tools_from_mcp_schema

# ====== Step 1: Define Input Schemas ======
# These define the contract for each tool

class VectorSearchInput(BaseModel):
    """Input for vector similarity search."""
    query: str = Field(..., description="Query text for semantic search")
    collection: str = Field(..., description="Collection name to search in")
    limit: int = Field(default=10, description="Maximum number of results")
    threshold: float = Field(default=0.7, description="Similarity threshold (0-1)")

class VectorInsertInput(BaseModel):
    """Input for inserting vectors."""
    text: str = Field(..., description="Text to embed and store")
    collection: str = Field(..., description="Collection name to insert into")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")

class VectorDeleteInput(BaseModel):
    """Input for deleting vectors."""
    ids: List[str] = Field(..., description="Vector IDs to delete")
    collection: str = Field(..., description="Collection name to delete from")

class VectorCollectionInput(BaseModel):
    """Input for collection operations."""
    name: str = Field(..., description="Collection name")
    dimension: Optional[int] = Field(default=768, description="Vector dimension")


# ====== Step 2: Create Tools (5 lines each!) ======

# Using the decorator pattern
@mcp_tool(
    tool_name="vector_search",
    server="vector-db",
    mcp_method="search",
    schema=VectorSearchInput,
    description="Search for similar vectors using semantic similarity"
)
class VectorSearchTool:
    """Semantic search in vector database."""
    pass

# Using the factory pattern
VectorInsertTool = mcp_tool(
    tool_name="vector_insert",
    server="vector-db",
    mcp_method="insert",
    schema=VectorInsertInput,
    description="Insert text embeddings into vector database"
)(None)

VectorDeleteTool = mcp_tool(
    tool_name="vector_delete",
    server="vector-db",
    mcp_method="delete",
    schema=VectorDeleteInput,
    description="Delete vectors from the database"
)(None)

VectorCreateCollectionTool = mcp_tool(
    tool_name="vector_create_collection",
    server="vector-db",
    mcp_method="create_collection",
    schema=VectorCollectionInput,
    description="Create a new vector collection"
)(None)


# ====== Step 3: Export Tools ======

def get_vector_db_tools():
    """Get all vector database MCP tools."""
    return [
        VectorSearchTool(),
        VectorInsertTool(),
        VectorDeleteTool(),
        VectorCreateCollectionTool(),
    ]


# ====== Alternative: Auto-generate from MCP Schema ======

def auto_generate_vector_tools():
    """
    Example of auto-generating tools from MCP schema.
    This would be called when the MCP server provides its schema.
    """
    # This would come from the MCP server's describe() method
    mock_mcp_schema = {
        "tools": [
            {
                "name": "search",
                "description": "Search for similar vectors",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "collection": {"type": "string", "description": "Collection name"},
                        "limit": {"type": "integer", "default": 10},
                        "threshold": {"type": "number", "default": 0.7}
                    },
                    "required": ["query", "collection"]
                }
            },
            {
                "name": "insert",
                "description": "Insert vectors",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "collection": {"type": "string"},
                        "metadata": {"type": "object"}
                    },
                    "required": ["text", "collection"]
                }
            }
        ]
    }
    
    # Auto-generate all tools from schema
    return create_tools_from_mcp_schema(
        server_name="vector-db",
        tool_definitions=mock_mcp_schema["tools"]
    )


# ====== Usage Example ======

if __name__ == "__main__":
    # Get manually defined tools
    manual_tools = get_vector_db_tools()
    print(f"Manually defined tools: {len(manual_tools)}")
    for tool in manual_tools:
        print(f"  - {tool.name}: {tool.description}")
    
    # Get auto-generated tools
    auto_tools = auto_generate_vector_tools()
    print(f"\nAuto-generated tools: {len(auto_tools)}")
    for tool in auto_tools:
        print(f"  - {tool.name}: {tool.description}")
    
    # Use a tool
    search_tool = manual_tools[0]
    result = search_tool._run(
        query="machine learning papers",
        collection="research_papers",
        limit=5
    )
    print(f"\nSearch result: {result}")


"""
Summary: Adding a new MCP server (vector-db) required:

1. Define Pydantic schemas for inputs (standard practice)
2. Create tools with 5 lines each using decorator/factory
3. That's it!

Compare to old pattern: Would have required ~150 lines of boilerplate.

Benefits:
- Consistent interface across all MCP servers
- Automatic error handling and logging
- Ready for auto-generation from MCP schemas
- Type-safe with full IDE support
- Easy to test and maintain
"""