#!/usr/bin/env python3
"""Run the Server Research MCP crew."""

import sys
import os
from pathlib import Path

# Add src directory to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

from server_research_mcp import run, train, replay, test


def main():
    """Main entry point that routes to appropriate command."""
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "train" and len(sys.argv) >= 4:
            train()
        elif command == "replay" and len(sys.argv) >= 3:
            replay()
        elif command == "test" and len(sys.argv) >= 4:
            test()
        elif command in ["train", "replay", "test"]:
            # Known command but insufficient arguments
            print("Usage:")
            print("  python run.py [topic]                  # Run the crew with optional topic")
            print("  python run.py train <n> <file>         # Train the crew")
            print("  python run.py replay <task_id>         # Replay from task")
            print("  python run.py test <n> <model>         # Test the crew")
            sys.exit(1)
        elif command.startswith('-') or command in ['help', '--help', '-h']:
            # Help or option flags
            print("Usage:")
            print("  python run.py [topic]                  # Run the crew with optional topic")
            print("  python run.py train <n> <file>         # Train the crew")
            print("  python run.py replay <task_id>         # Replay from task")
            print("  python run.py test <n> <model>         # Test the crew")
            sys.exit(0 if command in ['help', '--help', '-h'] else 1)
        elif len(command.split()) == 1 and len(command) < 20 and command not in [
            'machine', 'learning', 'quantum', 'research', 'analysis', 'study', 'transformers',
            'attention', 'sparse', 'neural', 'networks', 'deep', 'ai', 'artificial', 'intelligence'
        ] and not any(char in command for char in [' ', '.', ',', ':', ';', '!', '?', '"', "'"]):
            # Single suspicious word that looks like an invalid command
            # Allow common research terms and phrases with spaces/punctuation to pass through
            print("Usage:")
            print("  python run.py [topic]                  # Run the crew with optional topic")
            print("  python run.py train <n> <file>         # Train the crew")
            print("  python run.py replay <task_id>         # Replay from task")
            print("  python run.py test <n> <model>         # Test the crew")
            sys.exit(1)
        else:
            # Treat as topic and run the main application
            from server_research_mcp.main import main as main_func
            import asyncio
            asyncio.run(main_func())
    else:
        # Default: run the crew (will prompt for topic interactively)
        run()


if __name__ == "__main__":
    main()