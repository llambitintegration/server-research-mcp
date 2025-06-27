"""Pydantic models for Obsidian document structures."""

from typing import List, Dict, Optional, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field


class ObsidianFrontmatter(BaseModel):
    """Obsidian YAML frontmatter model."""
    # Required fields
    title: str = Field(..., description="Document title")
    authors: List[str] = Field(..., description="List of author names")
    year: int = Field(..., description="Publication year")
    created: datetime = Field(default_factory=datetime.now, description="Note creation date")
    modified: datetime = Field(default_factory=datetime.now, description="Last modification date")
    
    # Common metadata
    tags: List[str] = Field(default_factory=list, description="Obsidian tags")
    aliases: List[str] = Field(default_factory=list, description="Alternative names/references")
    cssclass: Optional[str] = Field(None, description="CSS class for styling")
    
    # Paper-specific metadata
    journal: Optional[str] = Field(None, description="Journal name")
    doi: Optional[str] = Field(None, description="Digital Object Identifier")
    arxiv_id: Optional[str] = Field(None, description="arXiv identifier")
    paper_type: str = Field("research", description="Type of paper")
    keywords: List[str] = Field(default_factory=list, description="Paper keywords")
    
    # Knowledge graph connections
    related_papers: List[str] = Field(default_factory=list, description="Wiki-links to related papers")
    cited_by: List[str] = Field(default_factory=list, description="Papers that cite this one")
    cites: List[str] = Field(default_factory=list, description="Papers this one cites")
    author_pages: List[str] = Field(default_factory=list, description="Links to author pages")
    topic_pages: List[str] = Field(default_factory=list, description="Links to topic pages")
    
    # Custom fields
    key_findings: List[str] = Field(default_factory=list, description="Main findings")
    quality_score: float = Field(1.0, description="Paper quality/relevance score")
    read_status: str = Field("unread", description="Reading status: unread, reading, read")
    importance: str = Field("medium", description="Importance level: low, medium, high")
    
    # Additional metadata
    custom_fields: Dict[str, Any] = Field(default_factory=dict, description="Additional custom fields")


class ObsidianDocument(BaseModel):
    """Complete Obsidian document model."""
    frontmatter: ObsidianFrontmatter = Field(..., description="YAML frontmatter")
    content: str = Field(..., description="Markdown content body")
    vault_path: str = Field(..., description="Path within Obsidian vault")
    
    def to_markdown(self) -> str:
        """Convert to complete markdown document with frontmatter."""
        # Build YAML frontmatter
        yaml_lines = ["---"]
        
        # Serialize frontmatter fields
        fm = self.frontmatter.dict(exclude_none=True, exclude={'custom_fields'})
        
        # Handle datetime fields
        fm['created'] = fm['created'].strftime("%Y-%m-%d %H:%M:%S")
        fm['modified'] = fm['modified'].strftime("%Y-%m-%d %H:%M:%S")
        
        # Add standard fields
        for key, value in fm.items():
            if isinstance(value, list):
                if value:  # Only include non-empty lists
                    yaml_lines.append(f"{key}:")
                    for item in value:
                        yaml_lines.append(f"  - {item}")
            else:
                yaml_lines.append(f"{key}: {value}")
        
        # Add custom fields
        if self.frontmatter.custom_fields:
            for key, value in self.frontmatter.custom_fields.items():
                if isinstance(value, list):
                    yaml_lines.append(f"{key}:")
                    for item in value:
                        yaml_lines.append(f"  - {item}")
                else:
                    yaml_lines.append(f"{key}: {value}")
        
        yaml_lines.append("---")
        yaml_lines.append("")  # Empty line after frontmatter
        
        # Combine frontmatter and content
        return "\n".join(yaml_lines) + "\n" + self.content


class ObsidianLink(BaseModel):
    """Model for Obsidian wiki-links."""
    target: str = Field(..., description="Target note/page name")
    display_text: Optional[str] = Field(None, description="Display text if different from target")
    
    def to_wiki_link(self) -> str:
        """Convert to Obsidian wiki-link format."""
        if self.display_text:
            return f"[[{self.target}|{self.display_text}]]"
        return f"[[{self.target}]]"


class ObsidianTag(BaseModel):
    """Model for Obsidian tags."""
    name: str = Field(..., description="Tag name without # prefix")
    nested: bool = Field(False, description="Whether this is a nested tag")
    
    def to_tag(self) -> str:
        """Convert to Obsidian tag format."""
        return f"#{self.name}"