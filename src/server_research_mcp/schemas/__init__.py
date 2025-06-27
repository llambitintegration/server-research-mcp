"""Schemas module for server-research-mcp.

This module contains Pydantic schemas for data validation.
"""

# Export commonly used schemas for easy import
from .research_paper import EnrichedQuery, RawPaperData, ResearchPaperSchema, Author, PaperMetadata
from .obsidian_meta import ObsidianDocument, ObsidianFrontmatter

__all__ = [
    'EnrichedQuery',
    'RawPaperData',
    'ResearchPaperSchema',
    'Author',
    'PaperMetadata',
    'ObsidianDocument',
    'ObsidianFrontmatter',
]