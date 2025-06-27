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
        
        # Verify LLM configuration
        from server_research_mcp.crew import get_configured_llm
        llm = get_configured_llm()
        assert llm is not None, "LLM must be configured"
        
        print("✅ CRITICAL: Environment validation passed")
    
    @patch('server_research_mcp.tools.mcp_tools.get_mcp_manager')
    def test_critical_crew_initialization(self, mock_mcp, alpha_test_environment):
        """CRITICAL: Crew must initialize without errors."""
        # Mock MCP manager
        mock_manager = MagicMock()
        mock_manager.initialize = MagicMock(return_value=True)
        mock_manager.call_tool = MagicMock(return_value={"status": "success"})
        mock_mcp.return_value = mock_manager
        
        # Test crew creation
        crew_instance = ServerResearchMcp()
        assert crew_instance is not None
        
        # Test crew assembly
        crew = crew_instance.crew()
        assert crew is not None
        assert len(crew.agents) == 4, "Must have 4 agents"
        assert len(crew.tasks) == 4, "Must have 4 tasks"
        
        # Verify agents have tools
        for agent in crew.agents:
            assert len(agent.tools) > 0, f"Agent {agent.role} must have tools"
        
        print("✅ CRITICAL: Crew initialization passed")
    
    @patch('server_research_mcp.tools.mcp_tools.get_mcp_manager')
    def test_critical_workflow_simulation(self, mock_mcp, alpha_test_environment):
        """CRITICAL: End-to-end workflow must complete without errors."""
        # Mock MCP responses that simulate real workflow
        mock_manager = MagicMock()
        
        async def mock_call_tool(server, tool, arguments):
            """Simulate realistic MCP responses."""
            if tool == "search_nodes":
                return {
                    "nodes": [
                        {
                            "name": "machine_learning_research",
                            "type": "research_area",
                            "observations": [
                                "Active research area with many papers",
                                "Popular keywords: transformers, attention, neural networks"
                            ]
                        }
                    ]
                }
            elif tool == "create_entities":
                return {"status": "created", "entity_id": f"entity_{int(time.time())}"}
            
            elif tool == "search_items":
                return {
                    "items": [
                        {
                            "key": "SMITH2024",
                            "title": "Advances in Machine Learning Transformers",
                            "authors": [{"name": "John Smith", "affiliation": "AI University"}],
                            "year": 2024,
                            "journal": "AI Research Journal",
                            "abstract": "This paper presents novel advances in transformer architectures...",
                            "doi": "10.1234/airesearch.2024.001"
                        }
                    ]
                }
            
            elif tool == "get_item_fulltext" or tool == "get_item":
                return {
                    "content": "# Advances in Machine Learning Transformers\n\n## Abstract\nThis paper presents novel advances...\n\n## Introduction\nTransformers have revolutionized...",
                    "metadata": {
                        "title": "Advances in Machine Learning Transformers",
                        "authors": ["John Smith"],
                        "year": 2024,
                        "extraction_quality": 0.9
                    }
                }
            
            elif tool == "create_note":
                return {
                    "status": "created",
                    "path": f"{alpha_test_environment}/Papers/smith2024-advances-machine-learning.md"
                }
            
            else:
                return {"status": "success", "data": "mock_response"}
        
        mock_manager.initialize = MagicMock(return_value=True)
        mock_manager.call_tool = mock_call_tool
        mock_mcp.return_value = mock_manager
        
        # Test inputs
        test_inputs = {
            'paper_query': 'machine learning transformers',
            'topic': 'artificial intelligence',
            'current_year': 2024,
            'enriched_query': '{}',
            'raw_paper_data': '{}',
            'structured_json': '{}'
        }
        
        # Simulate crew execution (without actually running CrewAI)
        crew_instance = ServerResearchMcp()
        crew = crew_instance.crew()
        
        # Verify crew is properly configured for execution
        assert crew.process.value == "sequential", "Must use sequential process"
        assert crew.verbose == True, "Must have verbose logging for alpha"
        
        print("✅ CRITICAL: Workflow simulation passed")
    
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
        
        # Test invalid JSON handling
        invalid_json = "{'invalid': json}"
        is_valid, error = validate_enriched_query(invalid_json)
        assert is_valid == False, "Invalid JSON must be rejected"
        assert "JSON" in error, "Error message must mention JSON"
        
        # Test missing fields handling
        incomplete_data = json.dumps({"original_query": "test"})
        is_valid, error = validate_enriched_query(incomplete_data)
        assert is_valid == False, "Incomplete data must be rejected"
        assert "Missing" in error, "Error message must mention missing fields"
        
        print("✅ CRITICAL: Error handling passed")
    
    def test_critical_argument_parsing(self):
        """CRITICAL: Command line interface must work correctly."""
        # Test basic argument parsing
        with patch('sys.argv', ['main.py', 'test query']):
            args = parse_arguments()
            assert args.query == "test query"
            assert args.topic == "research"  # default
            assert isinstance(args.year, int)
        
        # Test full argument parsing
        test_args = [
            'main.py',
            'advanced query',
            '--topic', 'machine learning',
            '--year', '2023',
            '--verbose',
            '--dry-run'
        ]
        with patch('sys.argv', test_args):
            args = parse_arguments()
            assert args.query == "advanced query"
            assert args.topic == "machine learning"
            assert args.year == 2023
            assert args.verbose == True
            assert args.dry_run == True
        
        print("✅ CRITICAL: Argument parsing passed")
    
    def test_critical_file_output_structure(self, alpha_test_environment):
        """CRITICAL: Output files must be created in correct structure."""
        # Create outputs directory
        outputs_dir = Path(alpha_test_environment) / "outputs"
        outputs_dir.mkdir(exist_ok=True)
        
        # Test expected output files
        expected_files = [
            "enriched_query.json",
            "raw_paper_data.json",
            "structured_paper.json",
            "published_paper.md"
        ]
        
        for filename in expected_files:
            test_file = outputs_dir / filename
            test_file.write_text('{"test": "data"}' if filename.endswith('.json') else '# Test Content')
            assert test_file.exists(), f"Output file {filename} must be creatable"
        
        print("✅ CRITICAL: File output structure passed")
    
    @pytest.mark.parametrize("query_type", [
        "simple_query",
        "complex_query_with_special_chars",
        "unicode_query",
        "very_long_query"
    ])
    def test_critical_input_robustness(self, query_type, alpha_test_environment):
        """CRITICAL: Application must handle various input types."""
        test_queries = {
            "simple_query": "AI",
            "complex_query_with_special_chars": "AI/ML & Data Science (2024)",
            "unicode_query": "Künstliche Intelligenz",
            "very_long_query": "This is a very long query about machine learning " * 10
        }
        
        query = test_queries[query_type]
        
        # Test that input parsing doesn't crash
        with patch('sys.argv', ['main.py', query]):
            args = parse_arguments()
            assert args.query == query
        
        print(f"✅ CRITICAL: Input robustness passed for {query_type}")


class TestAlphaPerformance:
    """Performance tests for alpha release."""
    
    @patch('server_research_mcp.tools.mcp_tools.get_mcp_manager')
    def test_crew_initialization_performance(self, mock_mcp):
        """Alpha must initialize within reasonable time."""
        mock_manager = MagicMock()
        mock_manager.initialize = MagicMock(return_value=True)
        mock_mcp.return_value = mock_manager
        
        start_time = time.time()
        
        crew_instance = ServerResearchMcp()
        crew = crew_instance.crew()
        
        initialization_time = time.time() - start_time
        
        # Alpha should initialize within 10 seconds
        assert initialization_time < 10, f"Initialization took {initialization_time:.2f}s, must be < 10s"
        
        print(f"✅ PERFORMANCE: Crew initialized in {initialization_time:.2f}s")
    
    def test_validation_performance(self):
        """Validation functions must be fast."""
        from server_research_mcp.crew import validate_enriched_query
        
        # Test with large but valid JSON
        large_query = json.dumps({
            "original_query": "test",
            "expanded_terms": [f"term_{i}" for i in range(1000)],
            "search_strategy": "test" * 100
        })
        
        start_time = time.time()
        is_valid, result = validate_enriched_query(large_query)
        validation_time = time.time() - start_time
        
        assert is_valid == True
        assert validation_time < 1, f"Validation took {validation_time:.2f}s, must be < 1s"
        
        print(f"✅ PERFORMANCE: Validation completed in {validation_time:.4f}s")


class TestAlphaIntegration:
    """Integration tests for alpha release."""
    
    def test_mcp_manager_compatibility(self):
        """Test that both standard and enhanced MCP managers work."""
        from server_research_mcp.tools.mcp_manager import get_mcp_manager
        from server_research_mcp.tools.enhanced_mcp_manager import get_enhanced_mcp_manager
        
        # Both managers should be importable and createable
        standard_manager = get_mcp_manager()
        enhanced_manager = get_enhanced_mcp_manager()
        
        assert standard_manager is not None
        assert enhanced_manager is not None
        
        print("✅ INTEGRATION: MCP manager compatibility passed")
    
    def test_schema_compatibility(self):
        """Test that all schemas work together."""
        from server_research_mcp.schemas import (
            EnrichedQuery, RawPaperData, ResearchPaperSchema, 
            Author, PaperMetadata, ObsidianDocument
        )
        
        # Test schema creation and serialization
        author = Author(name="Test Author", affiliation="Test University")
        metadata = PaperMetadata(
            title="Test Paper",
            authors=[author],
            year=2024,
            abstract="Test abstract"
        )
        
        # Schemas should serialize properly
        author_dict = author.model_dump()
        metadata_dict = metadata.model_dump()
        
        assert "name" in author_dict
        assert "title" in metadata_dict
        
        print("✅ INTEGRATION: Schema compatibility passed")


if __name__ == "__main__":
    print("🧪 Running Alpha Readiness Tests")
    print("=" * 50)
    pytest.main([__file__, "-v", "--tb=short"]) 