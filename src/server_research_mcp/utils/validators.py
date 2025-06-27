"""Validation utilities for server-research-mcp."""

from typing import Tuple, Any


def validate_context_gathering_output(result: str) -> Tuple[bool, Any]:
    """Validate context gathering output meets knowledge foundation standards."""
    try:
        # Check for minimum content length
        if len(result.strip()) < 300:
            return (False, "Context gathering output too brief - needs comprehensive knowledge foundation")
        
        # Check for required sections
        required_sections = ["executive summary", "concepts", "knowledge"]
        missing_sections = [section for section in required_sections 
                          if section.lower() not in result.lower()]
        if missing_sections:
            return (False, f"Missing required sections: {', '.join(missing_sections)}")
        
        # Check for markdown formatting
        if "#" not in result:
            return (False, "Output should use markdown headers for section structure")
        
        # Avoid code blocks as specified
        if "```" in result:
            return (False, "Output should not contain code block markers (```)")
        
        return (True, result.strip())
    except Exception as e:
        return (False, f"Validation error: {str(e)}")


def validate_research_topic(topic: str) -> Tuple[bool, str]:
    """Validate research topic input."""
    topic = topic.strip()
    
    if not topic:
        return (False, "Research topic cannot be empty")
    
    if len(topic) < 3:
        return (False, "Research topic too short - please provide more detail")
    
    if len(topic) > 200:
        return (False, "Research topic too long - please be more concise")
    
    return (True, topic)