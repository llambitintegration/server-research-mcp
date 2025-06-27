"""
Type definitions for MCP server interfaces.

This module defines common types used across protocol interfaces to ensure
consistency and type safety.
"""

from typing import Dict, Any, List, Union

# Common types used across protocols
ToolResponse = Dict[str, Any]
"""Response from an MCP tool call - typically contains status, data, and metadata."""

ToolParameters = Dict[str, Any]
"""Parameters passed to MCP tools - key-value pairs of arguments."""

EntityData = Dict[str, Union[str, List[str]]]
"""Data structure for memory entities - contains name, type, observations, etc.""" 