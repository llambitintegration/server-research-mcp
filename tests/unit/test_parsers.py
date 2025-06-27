"""
Parser and schema validation tests - consolidated from test_research_paper_parser.py
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
    
    def test_research_paper_schema(self):
        """Test ResearchPaperSchema validation."""
        paper_data = {
            "title": "Test Paper",
            "abstract": "This is a test abstract for validation.",
            "authors": [
                {"name": "Author One", "affiliation": "University A"},
                {"name": "Author Two", "affiliation": "University B"}
            ],
            "sections": [
                {"title": "Introduction", "content": "Introduction content here."},
                {"title": "Methods", "content": "Methods content here."},
                {"title": "Results", "content": "Results content here."},
                {"title": "Conclusion", "content": "Conclusion content here."}
            ],
            "metadata": {
                "year": 2024,
                "journal": "Test Journal",
                "doi": "10.1234/test"
            }
        }
        
        paper = ResearchPaperSchema(**paper_data)
        assert paper.title == "Test Paper"
        assert len(paper.authors) == 2
        assert len(paper.sections) >= 4
    
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


class TestParserValidation:
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
    
    def test_raw_paper_data_validation(self):
        """Test raw paper data validation."""
        raw_data = RawPaperData(
            source="test",
            title="Test Paper",
            content="Paper content here...",
            metadata={"year": 2024}
        )
        
        assert raw_data.source == "test"
        assert raw_data.title == "Test Paper"
        assert raw_data.metadata["year"] == 2024
    
    def test_paper_metadata_validation(self):
        """Test paper metadata validation."""
        metadata = PaperMetadata(
            year=2024,
            journal="Test Journal",
            volume="12",
            issue="3",
            pages="123-135",
            doi="10.1234/test",
            url="https://example.com/paper"
        )
        
        assert metadata.year == 2024
        assert metadata.journal == "Test Journal"
        assert metadata.doi == "10.1234/test"


class TestContentParsing:
    """Test content parsing functionality."""
    
    def test_markdown_parsing(self):
        """Test markdown content parsing."""
        markdown_content = """
        # Introduction
        
        This is the introduction section.
        
        ## Background
        
        Some background information.
        
        # Methods
        
        Description of methods used.
        
        # Results
        
        The results of the study.
        
        # Conclusion
        
        Final thoughts and conclusions.
        """
        
        # Simulate parsing logic
        sections = []
        current_section = None
        
        for line in markdown_content.strip().split('\n'):
            line = line.strip()
            if line.startswith('#'):
                if current_section:
                    sections.append(current_section)
                
                level = len(line) - len(line.lstrip('#'))
                title = line.lstrip('# ').strip()
                current_section = {
                    "title": title,
                    "content": "",
                    "level": level
                }
            elif current_section and line:
                current_section["content"] += line + "\n"
        
        if current_section:
            sections.append(current_section)
        
        assert len(sections) >= 4
        assert any(s["title"] == "Introduction" for s in sections)
        assert any(s["title"] == "Methods" for s in sections)
        assert any(s["title"] == "Results" for s in sections)
        assert any(s["title"] == "Conclusion" for s in sections)
    
    def test_reference_extraction(self):
        """Test reference extraction from text."""
        text_with_refs = """
        This is a paper about machine learning (Smith et al., 2023).
        Another important work is by Johnson (2022).
        See also: Brown, A., & Davis, B. (2021). "Deep Learning Methods."
        """
        
        # Simple reference extraction pattern
        import re
        
        patterns = [
            r'\(([A-Z][a-z]+ et al\., \d{4})\)',
            r'([A-Z][a-z]+ \(\d{4}\))',
            r'([A-Z][a-z]+, [A-Z]\., & [A-Z][a-z]+, [B-Z]\. \(\d{4}\))'
        ]
        
        extracted_refs = []
        for pattern in patterns:
            matches = re.findall(pattern, text_with_refs)
            extracted_refs.extend(matches)
        
        assert len(extracted_refs) >= 2
    
    def test_author_name_parsing(self):
        """Test parsing of author names from different formats."""
        author_strings = [
            "John Smith",
            "Smith, John",
            "J. Smith",
            "Smith, J.",
            "John A. Smith",
            "Smith, John A."
        ]
        
        def parse_author_name(name_str):
            """Simple author name parser."""
            if ',' in name_str:
                parts = name_str.split(',')
                last_name = parts[0].strip()
                first_name = parts[1].strip() if len(parts) > 1 else ""
            else:
                parts = name_str.strip().split()
                if len(parts) >= 2:
                    first_name = ' '.join(parts[:-1])
                    last_name = parts[-1]
                else:
                    first_name = ""
                    last_name = parts[0] if parts else ""
            
            return {
                "first_name": first_name,
                "last_name": last_name,
                "full_name": f"{first_name} {last_name}".strip()
            }
        
        parsed_authors = [parse_author_name(name) for name in author_strings]
        
        # All should have last names
        assert all(author["last_name"] for author in parsed_authors)
        
        # Most should have first names
        with_first_names = [a for a in parsed_authors if a["first_name"]]
        assert len(with_first_names) >= len(author_strings) - 2 