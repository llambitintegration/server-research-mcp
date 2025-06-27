"""
Test suite for the Research Paper Parser crew.
"""

import pytest
import os
import json
import tempfile
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Import components to test
from server_research_mcp.crew import ServerResearchMcp
from server_research_mcp.schemas import (
    EnrichedQuery,
    RawPaperData,
    ResearchPaperSchema,
    Author,
    PaperMetadata,
    ObsidianFrontmatter,
    ObsidianDocument
)
from server_research_mcp.tools.mcp_tools import (
    get_historian_tools,
    get_researcher_tools,
    get_archivist_tools,
    get_publisher_tools
)


class TestSchemas:
    """Test schema models."""
    
    def test_author_model(self):
        """Test Author model creation."""
        author = Author(
            name="John Doe",
            affiliation="University X",
            email="john@example.com",
            orcid="0000-0000-0000-0000",
            is_corresponding=True
        )
        assert author.name == "John Doe"
        assert author.is_corresponding == True
    
    def test_enriched_query_model(self):
        """Test EnrichedQuery model."""
        query = EnrichedQuery(
            original_query="machine learning",
            expanded_terms=["AI", "deep learning", "neural networks"],
            search_strategy="comprehensive search with semantic expansion",
            related_papers=[{"title": "Related Paper", "key": "ABC123"}],
            known_authors=[{"name": "Expert Author", "area": "ML"}],
            topic_context={"field": "computer science"},
            memory_entities=["ml_topic", "ai_concepts"]
        )
        assert len(query.expanded_terms) == 3
        assert query.search_strategy != ""
    
    def test_obsidian_document_to_markdown(self):
        """Test Obsidian document markdown generation."""
        frontmatter = ObsidianFrontmatter(
            title="Test Paper",
            authors=["Author One", "Author Two"],
            year=2023,
            tags=["research", "ai", "test"],
            journal="Test Journal",
            doi="10.1234/test",
            keywords=["keyword1", "keyword2"]
        )
        
        doc = ObsidianDocument(
            frontmatter=frontmatter,
            content="# Test Paper\n\nThis is the content.",
            vault_path="/vault/Papers/test-paper.md"
        )
        
        markdown = doc.to_markdown()
        assert "---" in markdown
        assert "title: Test Paper" in markdown
        assert "tags:" in markdown
        assert "- research" in markdown
        assert "# Test Paper" in markdown


class TestMCPTools:
    """Test MCP tool collections."""
    
    def test_historian_tools(self):
        """Test historian tools are properly configured."""
        tools = get_historian_tools()
        assert len(tools) == 5
        tool_names = [tool.name for tool in tools]
        assert "memory_search" in tool_names
        assert "memory_create_entity" in tool_names
        assert "context7_resolve_library" in tool_names
    
    def test_researcher_tools(self):
        """Test researcher tools are properly configured."""
        tools = get_researcher_tools()
        assert len(tools) == 3
        tool_names = [tool.name for tool in tools]
        assert "zotero_search" in tool_names
        assert "zotero_extract" in tool_names
        assert "sequential_thinking" in tool_names
    
    def test_archivist_tools(self):
        """Test archivist tools are properly configured."""
        tools = get_archivist_tools()
        assert len(tools) == 5
        tool_names = [tool.name for tool in tools]
        assert "schema_validation" in tool_names
        assert "intelligent_summary" in tool_names
        assert "filesystem_read" in tool_names
    
    def test_publisher_tools(self):
        """Test publisher tools are properly configured."""
        tools = get_publisher_tools()
        assert len(tools) == 5
        tool_names = [tool.name for tool in tools]
        assert "obsidian_create_note" in tool_names
        assert "obsidian_link_generator" in tool_names
        assert "filesystem_write" in tool_names


class TestCrew:
    """Test crew configuration and execution."""
    
    @patch.dict(os.environ, {
        'ANTHROPIC_API_KEY': 'test-key',
        'LLM_PROVIDER': 'anthropic',
        'OBSIDIAN_VAULT_PATH': '/test/vault'
    })
    def test_crew_initialization(self):
        """Test crew can be initialized."""
        crew = ServerResearchMcp()
        assert crew is not None
        
        # Test agent creation
        historian = crew.historian()
        assert historian is not None
        assert len(historian.tools) > 0
        
        researcher = crew.researcher()
        assert researcher is not None
        
        archivist = crew.archivist()
        assert archivist is not None
        
        publisher = crew.publisher()
        assert publisher is not None
    
    @patch.dict(os.environ, {
        'ANTHROPIC_API_KEY': 'test-key',
        'LLM_PROVIDER': 'anthropic',
        'OBSIDIAN_VAULT_PATH': '/test/vault'
    })
    def test_task_creation(self):
        """Test task creation."""
        crew = ServerResearchMcp()
        
        # Test task creation
        context_task = crew.context_gathering_task()
        assert context_task is not None
        assert context_task.output_file == 'outputs/enriched_query.json'
        
        extraction_task = crew.paper_extraction_task()
        assert extraction_task is not None
        
        structuring_task = crew.data_structuring_task()
        assert structuring_task is not None
        
        generation_task = crew.markdown_generation_task()
        assert generation_task is not None


class TestValidation:
    """Test validation functions."""
    
    def test_enriched_query_validation(self):
        """Test enriched query validation."""
        from server_research_mcp.crew import validate_enriched_query
        
        # Valid query
        valid_data = json.dumps({
            "original_query": "test",
            "expanded_terms": ["term1", "term2"],
            "search_strategy": "test strategy"
        })
        is_valid, result = validate_enriched_query(valid_data)
        assert is_valid == True
        
        # Invalid query - missing field
        invalid_data = json.dumps({
            "original_query": "test"
        })
        is_valid, error = validate_enriched_query(invalid_data)
        assert is_valid == False
        assert "Missing required fields" in error
    
    def test_raw_paper_validation(self):
        """Test raw paper data validation."""
        from server_research_mcp.crew import validate_raw_paper_data
        
        # Valid data
        valid_data = json.dumps({
            "metadata": {"title": "Test"},
            "full_text": "Paper content",
            "sections": {"intro": "Introduction"},
            "extraction_quality": 0.9
        })
        is_valid, result = validate_raw_paper_data(valid_data)
        assert is_valid == True
        
        # Invalid quality score
        invalid_data = json.dumps({
            "metadata": {"title": "Test"},
            "full_text": "Paper content",
            "sections": {"intro": "Introduction"},
            "extraction_quality": 1.5  # Invalid - should be 0-1
        })
        is_valid, error = validate_raw_paper_data(invalid_data)
        assert is_valid == False
        assert "between 0 and 1" in error


class TestIntegration:
    """Integration tests for the full pipeline."""
    
    @patch.dict(os.environ, {
        'ANTHROPIC_API_KEY': 'test-key',
        'LLM_PROVIDER': 'anthropic',
        'OBSIDIAN_VAULT_PATH': '/test/vault',
        'DRY_RUN': 'true'  # Enable dry run for testing
    })
    def test_crew_execution_dry_run(self):
        """Test crew execution in dry run mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test inputs
            inputs = {
                "paper_query": "test paper",
                "topic": "artificial intelligence",
                "current_year": 2024,
                "enriched_query": "{}",
                "raw_paper_data": "{}",
                "structured_json": "{}"
            }
            
            # Mock the crew execution
            crew = ServerResearchMcp()
            
            # Verify crew structure
            crew_instance = crew.crew()
            assert crew_instance is not None
            assert len(crew_instance.agents) == 4  # Four agents
            assert len(crew_instance.tasks) == 4    # Four tasks
            assert crew_instance.process.value == "sequential"
            assert crew_instance.memory == True
            assert crew_instance.planning == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])