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
        else:
            print("Usage:")
            print("  python run.py                      # Run the crew")
            print("  python run.py train <n> <file>     # Train the crew")
            print("  python run.py replay <task_id>     # Replay from task")
            print("  python run.py test <n> <model>     # Test the crew")
            sys.exit(1)
    else:
        # Default: run the crew
        run()


if __name__ == "__main__":
    main()