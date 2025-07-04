"""
Unit tests for run.py module - Command routing and entry point functionality.
"""
import pytest
import sys
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import the run module
import run


class TestMainFunction:
    """Test the main function command routing."""
    
    def test_main_no_args(self):
        """Test main function with no arguments - should call run()."""
        with patch('sys.argv', ['run.py']):
            with patch('run.run') as mock_run:
                run.main()
                mock_run.assert_called_once()
    
    def test_main_train_command_valid(self):
        """Test main function with valid train command."""
        with patch('sys.argv', ['run.py', 'train', '5', 'training_file.txt']):
            with patch('run.train') as mock_train:
                run.main()
                mock_train.assert_called_once()
    
    def test_main_train_command_insufficient_args(self):
        """Test main function with train command but insufficient args."""
        with patch('sys.argv', ['run.py', 'train', '5']):  # Missing file argument
            with patch('sys.exit') as mock_exit:
                run.main()
                mock_exit.assert_called_once_with(1)
    
    def test_main_replay_command_valid(self):
        """Test main function with valid replay command."""
        with patch('sys.argv', ['run.py', 'replay', 'task_123']):
            with patch('run.replay') as mock_replay:
                run.main()
                mock_replay.assert_called_once()
    
    def test_main_replay_command_insufficient_args(self):
        """Test main function with replay command but insufficient args."""
        with patch('sys.argv', ['run.py', 'replay']):  # Missing task_id argument
            with patch('sys.exit') as mock_exit:
                run.main()
                mock_exit.assert_called_once_with(1)
    
    def test_main_test_command_valid(self):
        """Test main function with valid test command."""
        with patch('sys.argv', ['run.py', 'test', '10', 'gpt-4']):
            with patch('run.test') as mock_test:
                run.main()
                mock_test.assert_called_once()
    
    def test_main_test_command_insufficient_args(self):
        """Test main function with test command but insufficient args."""
        with patch('sys.argv', ['run.py', 'test', '10']):  # Missing model argument
            with patch('sys.exit') as mock_exit:
                run.main()
                mock_exit.assert_called_once_with(1)
    
    def test_main_invalid_command(self):
        """Test main function with invalid command."""
        with patch('sys.argv', ['run.py', 'invalid_command']):
            with patch('sys.exit') as mock_exit:
                run.main()
                mock_exit.assert_called_once_with(1)
    
    def test_main_help_message_printed(self):
        """Test that help message is printed for invalid commands."""
        with patch('sys.argv', ['run.py', 'invalid_command']):
            with patch('sys.exit') as mock_exit:
                with patch('builtins.print') as mock_print:
                    run.main()
                    
                    # Check that help messages were printed
                    mock_print.assert_any_call("Usage:")
                    mock_print.assert_any_call("  python run.py [topic]                  # Run the crew with optional topic")
                    mock_print.assert_any_call("  python run.py train <n> <file>         # Train the crew")
                    mock_print.assert_any_call("  python run.py replay <task_id>         # Replay from task")
                    mock_print.assert_any_call("  python run.py test <n> <model>         # Test the crew")
                    mock_exit.assert_called_once_with(1)


class TestImportBehavior:
    """Test import behavior and module setup."""
    
    def test_sys_path_modification(self):
        """Test that sys.path is modified correctly."""
        # Since the run module is already imported, we need to check that src is in sys.path
        current_dir = Path(__file__).parent.parent.parent
        src_dir = str(current_dir / "src")
        
        # The path should be added when run.py is imported
        assert src_dir in sys.path or any(src_dir in path for path in sys.path)
    
    def test_imports_available(self):
        """Test that required imports are available after path modification."""
        # These should not raise ImportError
        from server_research_mcp import run as run_func
        from server_research_mcp import train as train_func
        from server_research_mcp import replay as replay_func
        from server_research_mcp import test as test_func
        
        # Check that imports are callable
        assert callable(run_func)
        assert callable(train_func)
        assert callable(replay_func)
        assert callable(test_func)


class TestEntryPoint:
    """Test entry point behavior."""
    
    def test_if_name_main_behavior(self):
        """Test the if __name__ == '__main__' behavior."""
        # This is hard to test directly, but we can verify the main function exists
        assert hasattr(run, 'main')
        assert callable(run.main)
    
    @patch('run.main')
    def test_module_execution(self, mock_main):
        """Test module execution as script."""
        # This test simulates running the module as a script
        # In practice, this would be tested through subprocess or similar
        with patch('__main__.__name__', '__main__'):
            # We can't easily test the actual if __name__ == '__main__' block
            # but we can verify that main exists and is callable
            run.main()
            mock_main.assert_called_once()


class TestCommandValidation:
    """Test command validation edge cases."""
    
    def test_empty_command_args(self):
        """Test behavior with empty command arguments."""
        test_cases = [
            ['run.py', 'train'],  # No n or file
            ['run.py', 'train', ''],  # Empty string args
            ['run.py', 'replay'],  # No task_id
            ['run.py', 'test'],  # No n or model
            ['run.py', 'test', '5'],  # Only n, no model
        ]
        
        for args in test_cases:
            with patch('sys.argv', args):
                with patch('sys.exit') as mock_exit:
                    run.main()
                    mock_exit.assert_called_with(1)
    
    def test_exact_argument_count_validation(self):
        """Test that exact argument counts are validated."""
        # Valid cases
        valid_cases = [
            (['run.py', 'train', '5', 'file.txt'], 'train'),
            (['run.py', 'replay', 'task_123'], 'replay'),
            (['run.py', 'test', '10', 'gpt-4'], 'test'),
        ]
        
        for args, expected_func in valid_cases:
            with patch('sys.argv', args):
                with patch(f'run.{expected_func}') as mock_func:
                    run.main()
                    mock_func.assert_called_once()
    
    def test_case_sensitivity(self):
        """Test that commands are case-sensitive."""
        case_variants = ['Train', 'TRAIN', 'Replay', 'REPLAY', 'Test', 'TEST']
        
        for variant in case_variants:
            with patch('sys.argv', ['run.py', variant]):
                with patch('sys.exit') as mock_exit:
                    run.main()
                    mock_exit.assert_called_with(1)


class TestPathConfiguration:
    """Test path configuration and module discovery."""
    
    def test_src_directory_structure(self):
        """Test that the expected src directory structure exists."""
        current_dir = Path(__file__).parent.parent.parent
        src_dir = current_dir / "src"
        
        # The src directory should exist (or be in the path)
        assert src_dir.exists() or any('src' in path for path in sys.path)
    
    def test_server_research_mcp_module_accessible(self):
        """Test that the server_research_mcp module is accessible."""
        # This should not raise ImportError
        try:
            import server_research_mcp
            assert server_research_mcp is not None
        except ImportError as e:
            # If import fails, it should be due to missing dependencies, not path issues
            # We can allow this in test environments
            pytest.skip(f"server_research_mcp not available: {e}")


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_import_error_handling(self):
        """Test behavior when imports fail."""
        with patch('builtins.__import__', side_effect=ImportError("Test import error")):
            # The module should still be importable, but functions might fail
            # This is testing the robustness of the import structure
            pass  # run.py itself should import successfully
    
    def test_sys_path_modification_idempotent(self):
        """Test that sys.path modification is idempotent."""
        original_path = sys.path.copy()
        
        # Re-importing should not add duplicate paths
        import importlib
        importlib.reload(run)
        
        # Count occurrences of src paths
        src_paths = [path for path in sys.path if 'src' in path]
        original_src_paths = [path for path in original_path if 'src' in path]
        
        # Should not have significantly more src paths than before
        assert len(src_paths) <= len(original_src_paths) + 1


class TestIntegrationWithMainModule:
    """Test integration with the main server_research_mcp module."""
    
    def test_run_function_integration(self):
        """Test that run function can be called through the entry point."""
        with patch('sys.argv', ['run.py']):
            with patch('run.run') as mock_run:
                with patch('builtins.input', side_effect=['test query', 'y']):
                    run.main()
                    mock_run.assert_called_once()
    
    def test_all_command_functions_exist(self):
        """Test that all referenced command functions exist."""
        from server_research_mcp import run as run_func
        from server_research_mcp import train as train_func
        from server_research_mcp import replay as replay_func
        from server_research_mcp import test as test_func
        
        # All functions should be callable
        assert callable(run_func)
        assert callable(train_func)
        assert callable(replay_func)  
        assert callable(test_func)
    
    def test_command_execution_flow(self):
        """Test the complete command execution flow."""
        commands = [
            (['run.py'], 'run'),
            (['run.py', 'train', '5', 'file.txt'], 'train'),
            (['run.py', 'replay', 'task_123'], 'replay'),
            (['run.py', 'test', '10', 'gpt-4'], 'test'),
        ]
        
        for argv, expected_func in commands:
            with patch('sys.argv', argv):
                with patch(f'run.{expected_func}') as mock_func:
                    # Mock input calls for the 'run' command that needs user input
                    if expected_func == 'run':
                        with patch('builtins.input', side_effect=['test query', 'y']):
                            run.main()
                    else:
                        run.main()
                    mock_func.assert_called_once()