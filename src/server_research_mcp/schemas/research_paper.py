"""Pydantic models for research paper data structures."""

from typing import List, Dict, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator


class Author(BaseModel):
    """Author information model."""
    name: str = Field(..., description="Full name of the author")
    affiliation: Optional[str] = Field(None, description="Author's institutional affiliation")
    email: Optional[str] = Field(None, description="Author's email address")
    orcid: Optional[str] = Field(None, description="ORCID identifier")
    is_corresponding: bool = Field(False, description="Whether this is the corresponding author")


class Reference(BaseModel):
    """Reference/citation model."""
    id: str = Field(..., description="Unique reference ID")
    title: str = Field(..., description="Title of the referenced work")
    authors: List[str] = Field(..., description="List of author names")
    year: Optional[int] = Field(None, description="Publication year")
    journal: Optional[str] = Field(None, description="Journal or conference name")
    doi: Optional[str] = Field(None, description="DOI of the reference")
    citation_count: int = Field(0, description="Number of times cited in this paper")


class PaperSection(BaseModel):
    """Paper section content model."""
    title: str = Field(..., description="Section title")
    content: str = Field(..., description="Full text content of the section")
    summary: Optional[str] = Field(None, description="AI-generated summary of the section")
    subsections: List['PaperSection'] = Field(default_factory=list, description="Nested subsections")
    figures: List[Dict[str, str]] = Field(default_factory=list, description="Figures in this section")
    tables: List[Dict[str, str]] = Field(default_factory=list, description="Tables in this section")


class PaperMetadata(BaseModel):
    """Paper metadata model."""
    title: str = Field(..., description="Paper title")
    authors: List[Author] = Field(..., description="List of authors")
    year: int = Field(..., description="Publication year")
    journal: Optional[str] = Field(None, description="Journal name")
    volume: Optional[str] = Field(None, description="Journal volume")
    issue: Optional[str] = Field(None, description="Journal issue")
    pages: Optional[str] = Field(None, description="Page numbers")
    doi: Optional[str] = Field(None, description="Digital Object Identifier")
    arxiv_id: Optional[str] = Field(None, description="arXiv identifier")
    pmid: Optional[str] = Field(None, description="PubMed ID")
    keywords: List[str] = Field(default_factory=list, description="Paper keywords")
    abstract: str = Field(..., description="Paper abstract")
    publication_date: Optional[datetime] = Field(None, description="Publication date")
    paper_type: str = Field("research", description="Type of paper (research, review, etc.)")


class ResearchPaperSchema(BaseModel):
    """Complete research paper schema."""
    metadata: PaperMetadata = Field(..., description="Paper metadata")
    sections: List[PaperSection] = Field(..., description="Paper sections")
    references: List[Reference] = Field(default_factory=list, description="References cited")
    key_findings: List[str] = Field(default_factory=list, description="Key findings extracted")
    contributions: List[str] = Field(default_factory=list, description="Main contributions")
    limitations: List[str] = Field(default_factory=list, description="Identified limitations")
    future_work: List[str] = Field(default_factory=list, description="Future work suggestions")
    quality_metrics: Dict[str, float] = Field(default_factory=dict, description="Extraction quality scores")
    extraction_timestamp: datetime = Field(default_factory=datetime.now, description="When extraction occurred")
    

class EnrichedQuery(BaseModel):
    """Enriched query from Historian agent."""
    original_query: str = Field(..., description="Original user query")
    expanded_terms: List[str] = Field(..., description="Expanded search terms")
    related_papers: List[Dict[str, str]] = Field(default_factory=list, description="Related papers from memory")
    known_authors: List[Dict[str, str]] = Field(default_factory=list, description="Known authors in this area")
    topic_context: Dict[str, Any] = Field(default_factory=dict, description="Topic context from memory")
    search_strategy: str = Field(..., description="Recommended search strategy")
    memory_entities: List[str] = Field(default_factory=list, description="Created/updated memory entities")


class RawPaperData(BaseModel):
    """Raw paper data from Researcher agent."""
    zotero_key: Optional[str] = Field(None, description="Zotero item key")
    metadata: Dict[str, Any] = Field(..., description="Raw metadata extracted")
    full_text: str = Field(..., description="Complete extracted text")
    sections: Dict[str, str] = Field(..., description="Text organized by sections")
    references_raw: List[str] = Field(default_factory=list, description="Raw reference strings")
    figures: List[Dict[str, Any]] = Field(default_factory=list, description="Figure information")
    tables: List[Dict[str, Any]] = Field(default_factory=list, description="Table information")
    annotations: List[Dict[str, str]] = Field(default_factory=list, description="Annotations and highlights")
    extraction_method: str = Field(..., description="Method used for extraction")
    extraction_quality: float = Field(..., description="Quality score of extraction (0-1)")


# Enable forward references
PaperSection.model_rebuild()