"""Validation utilities for server-research-mcp."""

from typing import Tuple, Any
import re


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


def validate_research_output(data: str) -> Tuple[bool, Any]:
    """
    Validate research output text format.
    
    Args:
        data: Research output text to validate
        
    Returns:
        Tuple of (success, result_or_error_message)
    """
    if data is None:
        return False, "output too brief - minimum length required"
    
    if not isinstance(data, str):
        return False, "output too brief - minimum length required"
    
    data = data.strip()
    
    # Check minimum length (at least 200 characters)
    if len(data) < 200:
        return False, "output too brief - minimum length required"
    
    # Count bullet points (•, -, *, or unicode bullets)
    bullet_pattern = r'^[\s]*[•\-\*]'
    bullet_lines = [line for line in data.split('\n') if re.match(bullet_pattern, line)]
    
    # Require at least 10 bullet points for research output
    if len(bullet_lines) < 10:
        return False, "output too brief - minimum length required"
    
    return True, data.strip()


def validate_report_output(data: str) -> Tuple[bool, Any]:
    """
    Validate report output text format.
    
    Args:
        data: Report output text to validate
        
    Returns:
        Tuple of (success, result_or_error_message)
    """
    if data is None:
        return False, "output too brief - minimum length required"
    
    if not isinstance(data, str):
        return False, "output too brief - minimum length required"
    
    data = data.strip()
    
    # Check minimum length (at least 500 characters for reports)
    if len(data) < 500:
        return False, "output too brief"
    
    # Count markdown headers (# ## ###)
    header_pattern = r'^[\s]*#{1,6}\s+'
    header_lines = [line for line in data.split('\n') if re.match(header_pattern, line)]
    
    # Require at least 3 headers for report output
    if len(header_lines) < 3:
        if len(header_lines) == 1:
            return False, "requires at least 3 headers, found 1"
        return False, "not enough headers for structured report"
    
    # Check for code blocks (not allowed in reports)
    if '```' in data:
        return False, "reports should not contain code blocks"
    
    return True, data.strip()


def count_bullet_points(text: str) -> int:
    """Count bullet points in text."""
    if not text:
        return 0
    
    bullet_pattern = r'^[\s]*[•\-\*]'
    bullet_lines = [line for line in text.split('\n') if re.match(bullet_pattern, line)]
    return len(bullet_lines)


def count_markdown_headers(text: str) -> int:
    """Count markdown headers in text."""
    if not text:
        return 0
    
    header_pattern = r'^[\s]*#{1,6}\s+'
    header_lines = [line for line in text.split('\n') if re.match(header_pattern, line)]
    return len(header_lines)