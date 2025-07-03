"""
Comprehensive schema validation tests using real data.
Tests schemas against actual output data to ensure compatibility.
"""

import pytest
import json
from pathlib import Path
from pydantic import ValidationError
from datetime import datetime
from typing import Dict, Any

from server_research_mcp.schemas.research_paper import (
    ResearchPaperSchema, 
    PaperMetadata, 
    Author,
    PaperSection
)
from server_research_mcp.schemas.obsidian_meta import (
    ObsidianDocument,
    ObsidianFrontmatter
)


class TestSchemaCompatibility:
    """Test schema compatibility with real data."""
    
    @pytest.fixture
    def structured_paper_data(self):
        """Load real structured paper data."""
        data_path = Path("outputs/structured_paper.json")
        if not data_path.exists():
            pytest.skip("structured_paper.json not found - skipping real data validation")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def test_real_data_compatibility(self, structured_paper_data):
        """Test if schemas can handle real output data."""
        data = structured_paper_data
        
        # Test metadata compatibility
        metadata_data = data["metadata"]
        
        # Convert authors to Author objects
        authors = []
        for author_data in metadata_data["authors"]:
            author = Author(
                name=author_data["name"],
                affiliation=author_data.get("affiliation"),
                is_corresponding=False  # Default value
            )
            authors.append(author)
        
        # Create metadata with required fields
        metadata = PaperMetadata(
            title=metadata_data["title"],
            authors=authors,
            year=metadata_data["year"],
            journal=metadata_data.get("journal"),
            doi=metadata_data.get("doi"),
            abstract=metadata_data.get("abstract", "Abstract not available"),
            volume=metadata_data.get("volume"),
            pages=metadata_data.get("pages"),
            keywords=data.get("keywords", [])
        )
        
        # Test sections compatibility - handle both 'text' and 'content' fields
        sections = []
        for section_data in data["sections"]:
            # Map 'text' to 'content' field for compatibility
            content = section_data.get("text", section_data.get("content", ""))
            section = PaperSection(
                title=section_data["title"],
                content=content,
                summary=None  # Optional field
            )
            sections.append(section)
        
        # Create complete paper schema
        key_findings = data.get("key_findings", [])
        if isinstance(key_findings, str):
            key_findings = [key_findings]
        elif not isinstance(key_findings, list):
            key_findings = []
            
        paper = ResearchPaperSchema(
            metadata=metadata,
            sections=sections,
            key_findings=key_findings,
            extraction_timestamp=datetime.now()
        )
        
        # Assertions
        assert paper.metadata.title == metadata_data["title"]
        assert len(paper.metadata.authors) == len(metadata_data["authors"])
        assert len(paper.sections) == len(data["sections"])
        assert paper.metadata.year == metadata_data["year"]
        
        print(f"✅ Successfully validated real data with {len(paper.sections)} sections")
    
    def test_schema_flexibility(self):
        """Test schema flexibility with various input formats."""
        # Test minimal valid data
        minimal_data = {
            "metadata": {
                "title": "Minimal Test Paper",
                "authors": [{"name": "Test Author"}],
                "year": 2024,
                "abstract": "Minimal abstract"
            },
            "sections": [
                {"title": "Introduction", "content": "Test content"}
            ]
        }
        
        paper = ResearchPaperSchema(**minimal_data)
        assert paper.metadata.title == "Minimal Test Paper"
        assert len(paper.sections) == 1
        
        print("✅ Schema flexibility validated")
    
    def test_obsidian_frontmatter_validation(self):
        """Test Obsidian frontmatter validation."""
        # Valid frontmatter
        valid_frontmatter = ObsidianFrontmatter(
            title="Test Paper",
            authors=["Author 1", "Author 2"],
            year=2024
        )
        assert valid_frontmatter.title == "Test Paper"
        assert len(valid_frontmatter.authors) == 2
        
        print("✅ Obsidian frontmatter validation passed")


class TestSchemaRobustness:
    """Test schema robustness and edge cases."""
    
    def test_unicode_handling(self):
        """Test handling of Unicode characters."""
        unicode_data = {
            "metadata": {
                "title": "测试论文 with émojis 🧪",
                "authors": [{"name": "José García-Martínez"}],
                "year": 2024,
                "abstract": "Abstract with special chars: α, β, γ, ∑, ∫"
            },
            "sections": [
                {"title": "Introducción", "content": "Content with ñ, ü, ç"}
            ]
        }
        
        paper = ResearchPaperSchema(**unicode_data)
        assert "测试论文" in paper.metadata.title
        assert "García" in paper.metadata.authors[0].name
        assert "α" in paper.metadata.abstract
        
        print("✅ Unicode handling validated") 
