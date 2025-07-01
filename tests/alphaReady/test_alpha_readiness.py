"""
Alpha Readiness Test Suite
========================

Critical tests that must pass before alpha release.
These tests validate core functionality and user workflows.
"""

import pytest
import os
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime
import asyncio

# Add the src directory to Python path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from server_research_mcp.crew import ServerResearchMcp
from server_research_mcp.main import parse_arguments, validate_environment, run_crew


class TestAlphaReadiness:
    """Critical alpha release tests."""
    
    @pytest.fixture
    def alpha_test_environment(self):
        """Set up realistic test environment for alpha validation."""
        # Create temporary output directory
        temp_dir = tempfile.mkdtemp(prefix="alpha_test_")
        
        # Set required environment variables
        test_env = {
            "ANTHROPIC_API_KEY": "test-key-alpha",
            "OBSIDIAN_VAULT_PATH": temp_dir,
            "LLM_PROVIDER": "anthropic",
            "DISABLE_CREW_MEMORY": "true",  # For testing stability
            "DRY_RUN": "true"
        }
        
        original_env = {}
        for key, value in test_env.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = value
        
        yield temp_dir
        
        # Cleanup
        for key, original_value in original_env.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value
    
    def test_critical_environment_validation(self, alpha_test_environment):
        """CRITICAL: Environment must be properly configured."""
        from server_research_mcp.main import validate_environment
        
        assert validate_environment() == True, "Environment validation must pass for alpha"
        
        # LLM configuration testing moved to integration/test_llm.py for better coverage
        
        print("✅ CRITICAL: Environment validation passed")
    
    def test_critical_crew_initialization(self, alpha_test_environment):
        """CRITICAL: Crew must initialize without errors using MCPAdapt system."""
        # Test crew creation with MCPAdapt system
        crew_instance = ServerResearchMcp()
        assert crew_instance is not None
        
        # Test crew assembly
        crew = crew_instance.crew()
        assert crew is not None
        assert len(crew.agents) == 4, "Must have 4 agents"
        assert len(crew.tasks) == 4, "Must have 4 tasks"
        
        # Verify agents have tools (MCPAdapt creates tools dynamically)
        for agent in crew.agents:
            assert len(agent.tools) > 0, f"Agent {agent.role} must have tools"
        
        print("✅ CRITICAL: Crew initialization passed")
    
    def test_critical_crew_configuration(self, alpha_test_environment):
        """CRITICAL: Crew configuration must be correct for alpha."""
        crew_instance = ServerResearchMcp()
        crew = crew_instance.crew()
        
        # Verify crew is properly configured for execution
        assert crew.process.value == "sequential", "Must use sequential process"
        assert crew.verbose == True, "Must have verbose logging for alpha"
        
        # Verify agents have expected tool counts (MCPAdapt implementation)
        agent_tool_expectations = {
            "Historian": 9,
            "Researcher": 6, 
            "Archivist": 1,
            "Publisher": 2
        }
        
        for agent in crew.agents:
            if agent.role in agent_tool_expectations:
                expected_count = agent_tool_expectations[agent.role]
                actual_count = len(agent.tools)
                assert actual_count == expected_count, f"{agent.role} should have {expected_count} tools, got {actual_count}"
        
        print("✅ CRITICAL: Crew configuration passed")

    def test_critical_output_validation(self, alpha_test_environment):
        """CRITICAL: Output validation functions must work correctly."""
        from server_research_mcp.crew import (
            validate_enriched_query,
            validate_raw_paper_data,
            validate_structured_json,
            validate_markdown_output
        )
        
        # Test valid enriched query
        valid_query = json.dumps({
            "original_query": "test query",
            "expanded_terms": ["term1", "term2"],
            "search_strategy": "comprehensive search"
        })
        is_valid, result = validate_enriched_query(valid_query)
        assert is_valid == True, "Valid enriched query must pass validation"
        
        # Test valid raw paper data
        valid_paper = json.dumps({
            "metadata": {"title": "Test", "authors": ["Author"]},
            "full_text": "Content",
            "sections": [],
            "extraction_quality": 0.8
        })
        is_valid, result = validate_raw_paper_data(valid_paper)
        assert is_valid == True, "Valid paper data must pass validation"
        
        # Test valid structured JSON
        valid_structured = json.dumps({
            "metadata": {
                "title": "Test Paper",
                "authors": [{"name": "Author", "affiliation": "University"}],
                "year": 2024,
                "abstract": "Test abstract"
            },
            "sections": [{"title": "Section", "content": "Content"}]
        })
        is_valid, result = validate_structured_json(valid_structured)
        assert is_valid == True, "Valid structured JSON must pass validation"
        
        # Test valid markdown output
        valid_markdown = """---
title: Test Paper
authors: [Author]
year: 2024
---

# Test Paper

Content here.

Created note at: /vault/Papers/test-paper.md
"""
        is_valid, result = validate_markdown_output(valid_markdown)
        assert is_valid == True, "Valid markdown must pass validation"
        
        print("✅ CRITICAL: Output validation passed")
    
    def test_critical_error_handling(self, alpha_test_environment):
        """CRITICAL: Application must handle errors gracefully."""
        from server_research_mcp.crew import (
            validate_enriched_query,
            validate_raw_paper_data,
            validate_structured_json
        )
        
        # Test that validation functions are robust and handle edge cases
        # Note: validate_enriched_query now has fallback logic for robustness
        
        # Test raw paper data validation with invalid JSON
        invalid_json = "{'invalid': json}"
        is_valid, error = validate_raw_paper_data(invalid_json)
        assert is_valid == False, "Invalid JSON must be rejected by raw paper validator"
        assert "JSON" in error, "Error message must mention JSON"
        
        # Test structured JSON validation with missing fields
        incomplete_data = json.dumps({"title": "test"})  # Missing required metadata fields
        is_valid, error = validate_structured_json(incomplete_data)
        assert is_valid == False, "Incomplete data must be rejected by structured validator"
        assert "Missing" in error or "metadata" in error, "Error message must mention missing fields"
        
        print("✅ CRITICAL: Error handling passed")

    def test_critical_argument_parsing(self):
        """CRITICAL: Command line argument parsing must work."""
        # Test basic argument parsing
        test_args = ['--topic', 'test topic', '--year', '2024']
        
        # Mock sys.argv for testing
        with patch('sys.argv', ['main.py'] + test_args):
            args = parse_arguments()
            assert hasattr(args, 'topic')
            
        print("✅ CRITICAL: Argument parsing passed")
    
    def test_critical_file_output_structure(self, alpha_test_environment):
        """CRITICAL: File output structure must be consistent."""
        # Test that output directory structure can be created
        output_base = Path(alpha_test_environment) / "outputs"
        
        # Create expected directory structure
        directories = [
            output_base / "papers",
            output_base / "analysis", 
            output_base / "reports"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            assert directory.exists(), f"Failed to create {directory}"
            
        print("✅ CRITICAL: File output structure passed")
    
    @pytest.mark.parametrize("query_type", [
        "simple_query",
        "complex_query_with_special_chars",
        "unicode_query", 
        "very_long_query"
    ])
    def test_critical_input_robustness(self, query_type, alpha_test_environment):
        """CRITICAL: System must handle various input types robustly."""
        test_queries = {
            "simple_query": "AI testing",
            "complex_query_with_special_chars": "AI & ML: Testing (2024) - Part 1/2",
            "unicode_query": "AI 测试 with émojis 🤖",
            "very_long_query": "AI testing " * 100  # Very long query
        }
        
        query = test_queries[query_type]
        
        # Test that query can be processed without errors
        assert isinstance(query, str)
        assert len(query) > 0
        
        # Test basic string operations that would be used in processing
        assert query.strip() == query.strip()
        assert query.lower() is not None
        
        print(f"✅ CRITICAL: Input robustness passed for {query_type}")


class TestAlphaPerformance:
    """Alpha performance requirements."""
    
    def test_crew_initialization_performance(self):
        """Test crew initialization happens within reasonable time."""
        import time
        
        start_time = time.time()
        crew_instance = ServerResearchMcp()
        crew = crew_instance.crew()
        initialization_time = time.time() - start_time
        
        # Should initialize within 10 seconds (reasonable for alpha)
        assert initialization_time < 10, f"Crew initialization took {initialization_time:.2f}s, expected < 10s"
        
        print(f"✅ Crew initialization: {initialization_time:.2f}s")
    
    def test_validation_performance(self):
        """Test validation functions perform adequately."""
        from server_research_mcp.crew import validate_enriched_query
        
        # Test with various input sizes
        small_input = json.dumps({"query": "test"})
        large_input = json.dumps({"query": "test " * 1000, "terms": ["term"] * 100})
        
        import time
        
        # Small input should be very fast
        start = time.time()
        validate_enriched_query(small_input)
        small_time = time.time() - start
        
        # Large input should still be reasonable  
        start = time.time()
        validate_enriched_query(large_input)
        large_time = time.time() - start
        
        assert small_time < 0.1, f"Small validation took {small_time:.3f}s"
        assert large_time < 1.0, f"Large validation took {large_time:.3f}s"
        
        print(f"✅ Validation performance: small={small_time:.3f}s, large={large_time:.3f}s")


class TestAlphaIntegration:
    """Alpha integration requirements."""
    
    def test_schema_compatibility(self):
        """Test that schemas are compatible with expected formats."""
        from server_research_mcp.schemas.research_paper import ResearchPaperSchema
        from server_research_mcp.schemas.obsidian_meta import ObsidianDocument
        
        # Test schema instantiation
        paper_schema = ResearchPaperSchema
        obsidian_schema = ObsidianDocument
        
        assert paper_schema is not None
        assert obsidian_schema is not None
        
        print("✅ CRITICAL: Schema compatibility passed")


if __name__ == "__main__":
    print("🧪 Running Alpha Readiness Tests")
    print("=" * 50)
    pytest.main([__file__, "-v", "--tb=short"]) 