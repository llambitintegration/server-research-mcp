"""Tests for main module functionality."""

import os
import sys
import pytest
import asyncio
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from server_research_mcp.main import (
    parse_arguments,
    validate_environment,
    setup_output_directory,
    main
)


class TestArgumentParsing:
    """Test command line argument parsing."""
    
    def test_parse_arguments_basic(self):
        """Test parsing with basic arguments."""
        test_args = ["machine learning"]
        
        with patch('sys.argv', ['main.py'] + test_args):
            args = parse_arguments()
            
            assert args.topic == "machine learning"
            assert args.output_dir == "outputs"
            assert args.verbose is False
            assert args.dry_run is False
    
    def test_parse_arguments_all_options(self):
        """Test parsing with all options."""
        test_args = [
            "quantum computing",
            "--output-dir", "custom_output", 
            "--verbose",
            "--dry-run"
        ]
        
        with patch('sys.argv', ['main.py'] + test_args):
            args = parse_arguments()
            
            assert args.topic == "quantum computing"
            assert args.output_dir == "custom_output"
            assert args.verbose is True
            assert args.dry_run is True


class TestEnvironmentValidation:
    """Test environment validation."""
    
    @patch('server_research_mcp.config.llm_config.check_llm_config')
    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.exists')
    def test_validate_environment_success(self, mock_exists, mock_mkdir, mock_check_llm):
        """Test successful environment validation."""
        mock_check_llm.return_value = (True, "")
        mock_exists.return_value = True
        
        result = validate_environment()
        assert result is True
    
    @patch('server_research_mcp.config.llm_config.check_llm_config')
    def test_validate_environment_llm_failure(self, mock_check_llm):
        """Test environment validation with LLM failure."""
        # Force the mock to return failure
        mock_check_llm.return_value = (False, "No API key")
        
        # Also temporarily remove the environment variables that conftest.py sets
        # to ensure our mock is actually used
        with patch.dict(os.environ, {}, clear=True):
            result = validate_environment()
            assert result is False


class TestOutputDirectory:
    """Test output directory setup."""
    
    def test_setup_output_directory_success(self, tmp_path):
        """Test successful output directory setup."""
        test_dir = str(tmp_path / "test_output")
        
        result = setup_output_directory(test_dir)
        
        assert result.exists()
        assert result.is_dir()
        assert str(result).endswith("test_output")
    
    def test_setup_output_directory_existing(self, tmp_path):
        """Test setup with existing directory."""
        test_dir = str(tmp_path)
        
        result = setup_output_directory(test_dir)
        
        assert result.exists()
        assert result.is_dir()


class TestMainFunction:
    """Test main function behavior."""
    
    @patch('server_research_mcp.main.run_research')
    @patch('server_research_mcp.main.initialize_research_crew')
    @patch('server_research_mcp.main.setup_output_directory')
    @patch('server_research_mcp.main.validate_environment')
    @patch('server_research_mcp.main.parse_arguments')
    async def test_main_dry_run(self, mock_parse, mock_validate, mock_setup, 
                               mock_init_crew, mock_run):
        """Test main function with dry run."""
        # Setup mocks
        mock_args = MagicMock()
        mock_args.dry_run = True
        mock_args.topic = "test"
        mock_args.output_dir = "outputs"
        
        mock_parse.return_value = mock_args
        mock_validate.return_value = True
        mock_setup.return_value = Path("outputs")
        
        # Run main
        await main()
        
        # Verify dry run stops before crew initialization
        mock_parse.assert_called_once()
        mock_validate.assert_called_once()
        mock_setup.assert_called_once()
        mock_init_crew.assert_not_called()
        mock_run.assert_not_called()
    
    @patch('sys.exit')
    @patch('server_research_mcp.main.setup_output_directory')
    @patch('server_research_mcp.main.initialize_research_crew')
    @patch('server_research_mcp.main.run_research')
    @patch('server_research_mcp.main.validate_environment')
    @patch('server_research_mcp.main.parse_arguments')
    async def test_main_validation_failure(self, mock_parse, mock_validate, mock_run, mock_init, mock_setup, mock_exit):
        """Test main function with validation failure."""
        # Setup mocks
        mock_args = MagicMock()
        mock_args.dry_run = False
        mock_args.topic = "test topic"
        mock_args.output_dir = "outputs"
        
        mock_parse.return_value = mock_args
        mock_validate.return_value = False  # This should cause immediate exit
        
        # Make sure sys.exit actually stops execution by raising an exception
        mock_exit.side_effect = SystemExit(1)
        
        # Test that main() exits when validation fails
        with pytest.raises(SystemExit):
            await main()
        
        # Verify that validation was called
        mock_validate.assert_called_once()
        
        # Verify that subsequent functions were NOT called
        mock_setup.assert_not_called()
        mock_init.assert_not_called()
        mock_run.assert_not_called()
        
        # Verify sys.exit was called with code 1
        mock_exit.assert_called_once_with(1)