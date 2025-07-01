#!/usr/bin/env python
import sys
import os
import warnings
import tempfile
from dotenv import load_dotenv
from datetime import datetime
from typing import Dict, Any
import argparse
import json
from unittest.mock import patch

# Load environment variables first
load_dotenv()

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def setup_chromadb_environment():
    """Set up ChromaDB environment to prevent _type KeyError."""
    # Set ChromaDB configuration if not already set
    if not os.getenv("CHROMADB_PATH"):
        # Create a persistent ChromaDB directory for the application
        chromadb_dir = os.path.join(os.getcwd(), "data", "chromadb")
        os.makedirs(chromadb_dir, exist_ok=True)
        os.environ["CHROMADB_PATH"] = chromadb_dir
    
    # Allow ChromaDB to reset collections if needed
    if not os.getenv("CHROMADB_ALLOW_RESET"):
        os.environ["CHROMADB_ALLOW_RESET"] = "true"
    
    # Disable memory to avoid ChromaDB issues for now
    if not os.getenv("DISABLE_CREW_MEMORY"):
        os.environ["DISABLE_CREW_MEMORY"] = "true"
        print("⚠️  CrewAI memory disabled to avoid ChromaDB compatibility issues")
    
    print(f"✅ ChromaDB configured: {os.environ['CHROMADB_PATH']}")

def patch_chromadb_config():
    """Patch ChromaDB configuration to prevent _type KeyError."""
    try:
        import chromadb
        
        # Set default configuration to prevent _type KeyError
        original_settings = chromadb.config.Settings
        
        def patched_settings(*args, **kwargs):
            # Add default _type if not present
            if '_type' not in kwargs:
                kwargs['_type'] = 'hnsw'
            return original_settings(*args, **kwargs)
        
        chromadb.config.Settings = patched_settings
        print("✅ ChromaDB configuration patched")
        
    except ImportError:
        print("⚠️  ChromaDB not available for patching")
    except Exception as e:
        print(f"⚠️  ChromaDB patching failed: {e}")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Research Paper Parser - Extract and format papers from Zotero to Obsidian"
    )
    
    parser.add_argument(
        "query",
        type=str,
        nargs='?',  # Optional; will prompt if missing
        help="Paper identifier (Zotero key, DOI, title, or search query)"
    )
    
    parser.add_argument(
        "--topic",
        type=str,
        default="research",
        help="Research topic or area (default: research)"
    )
    
    parser.add_argument(
        "--year",
        type=int,
        default=datetime.now().year,
        help=f"Current year for context (default: {datetime.now().year})"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for results (default: outputs)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without actually creating Obsidian notes"
    )
    
    # Automatically answer yes to prompts (for automation)
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Automatically answer yes to confirmation prompts"
    )
    
    return parser.parse_args()

def validate_environment():
    """Validate required environment variables."""
    required_vars = [
        "ANTHROPIC_API_KEY",  # or OPENAI_API_KEY
        "OBSIDIAN_VAULT_PATH"
    ]
    
    optional_vars = [
        "ZOTERO_API_KEY",
        "ZOTERO_LIBRARY_ID",
        "LLM_PROVIDER",
        "LLM_MODEL"
    ]
    
    missing_required = []
    
    # Check for at least one LLM API key
    if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")):
        missing_required.append("ANTHROPIC_API_KEY or OPENAI_API_KEY")
    
    # Check other required variables
    if not os.getenv("OBSIDIAN_VAULT_PATH"):
        missing_required.append("OBSIDIAN_VAULT_PATH")
    
    if missing_required:
        print("❌ Missing required environment variables:")
        for var in missing_required:
            print(f"   - {var}")
        print("\nPlease set these in your .env file or environment")
        return False
    
    # Report optional variables
    print("✅ Environment validation passed")
    print("\nOptional variables (for full functionality):")
    for var in optional_vars:
        value = os.getenv(var)
        if value:
            print(f"   ✓ {var}: {'set' if var.endswith('KEY') else value}")
        else:
            print(f"   ✗ {var}: not set")
    
    return True

def get_user_input() -> str:
    """Get and validate user input for research topic."""
    import sys
    
    while True:
        try:
            topic = input("🎯 Enter your research topic: ").strip()
            if not topic:
                continue  # Loop until valid input provided
            
            # Confirmation loop
            while True:
                confirm = input(f"📋 Research topic: '{topic}'\n   Continue? (y/n): ").strip().lower()
                if confirm in ['y', 'yes']:
                    return topic
                elif confirm in ['n', 'no']:
                    sys.exit(0)
                # Invalid confirmation - loop again without message
                
        except KeyboardInterrupt:
            raise  # Re-raise KeyboardInterrupt for test handling

def run_crew(inputs: Dict[str, Any], verbose: bool = False):
    """Run the research paper parser crew."""
    print("\n🚀 Starting Research Paper Parser Crew...")
    print(f"   Query: {inputs['paper_query']}")
    print(f"   Topic: {inputs['topic']}")
    print(f"   Year: {inputs['current_year']}")
    
    try:
        # Import here to avoid circular imports
        from server_research_mcp.crew import ServerResearchMcp
        
        # Initialize and run the crew - MCPAdapt handles its own context management
        crew_instance = ServerResearchMcp()
        crew = crew_instance.crew()  # Get the crew object
        
        # Run with inputs
        result = crew.kickoff(inputs=inputs)
        
        print("\n✅ Crew execution completed successfully!")
        return result
        
    except Exception as e:
        print(f"\n❌ Error during crew execution: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return None

def run():
    """Entry point for run_crew script defined in pyproject.toml."""
    # Set up ChromaDB environment before any other initialization
    setup_chromadb_environment()
    patch_chromadb_config()
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Run with the parsed arguments
    return main_with_args(args)

def main_with_args(args):
    """Main function that accepts parsed arguments."""
    # Load environment variables
    load_dotenv()
    
    # Set up ChromaDB environment before CrewAI initialization
    setup_chromadb_environment()
    patch_chromadb_config()
    
    # Validate environment
    if not validate_environment():
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prompt for paper query if not provided via CLI
    if not args.query:
        try:
            args.query = input("Enter the paper query or identifier (e.g., DOI, title): ").strip()
        except EOFError:
            args.query = ""

        if not args.query:
            print("❌ Paper query is required. Aborting.")
            sys.exit(1)

    # Prepare inputs for the crew
    inputs = {
        "paper_query": args.query,
        "topic": args.topic,
        "current_year": args.year,
        "enriched_query": "{}",  # Will be filled by Historian
        "raw_paper_data": "{}",  # Will be filled by Researcher
        "structured_json": "{}"  # Will be filled by Archivist
    }
    
    # Set environment variable for dry run if specified
    if args.dry_run:
        os.environ["DRY_RUN"] = "true"
        print("\n⚠️  Running in DRY RUN mode - no files will be created")
    
    # Confirm before accessing memory/context if not auto-confirmed
    if not (args.yes or os.getenv("AUTO_YES")):
        try:
            proceed = input(
                "The Memory/Context agent will access your knowledge graph. Continue? [y/N]: "
            ).strip().lower()
        except EOFError:
            proceed = "n"

        if proceed != "y":
            print("Aborting at user request.")
            sys.exit(0)
    else:
        # Set env var for downstream scripts if flag used
        os.environ["AUTO_YES"] = "true"

    # Run the crew
    result = run_crew(inputs, verbose=args.verbose)
    
    if result:
        print("\n📊 Results Summary:")
        print(f"   - Enriched Query: {args.output_dir}/enriched_query.json")
        
        # Check if Obsidian note was created
        vault_path = os.getenv("OBSIDIAN_VAULT_PATH")
        if vault_path and not args.dry_run:
            print(f"\n📝 Obsidian note created in vault: {vault_path}")
    else:
        print("\n⚠️  Crew execution did not complete successfully")
        sys.exit(1)

def main():
    """Main entry point when called directly."""
    # Parse arguments
    args = parse_arguments()
    
    # Run with the parsed arguments
    return main_with_args(args)

def train():
    """Train the crew using CrewAI's native training functionality."""
    try:
        from server_research_mcp.crew import ServerResearchMcp
        
        # Default training parameters
        n_iterations = 2
        inputs = {"paper_query": "machine learning transformers", "topic": "research", "current_year": datetime.now().year}
        filename = "trained_model.pkl"
        
        print(f"🎯 Starting training for {n_iterations} iterations...")
        
        # Initialize and train the crew
        crew_instance = ServerResearchMcp()
        crew = crew_instance.crew()
        
        crew.train(
            n_iterations=n_iterations,
            inputs=inputs,
            filename=filename
        )
        
        print(f"✅ Training completed! Model saved as {filename}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """Replay the crew execution from a specific task using CrewAI's native replay functionality."""
    try:
        from server_research_mcp.crew import ServerResearchMcp
        import subprocess
        
        # Get task ID from command line args or use a default
        import sys
        if len(sys.argv) >= 3:
            task_id = sys.argv[2]
        else:
            # If no task ID provided, show available tasks
            print("📋 Retrieving latest task outputs...")
            subprocess.run(["crewai", "log-tasks-outputs"], check=True)
            print("\n❓ Please provide a task ID to replay:")
            print("Usage: python -m server_research_mcp replay <task_id>")
            return
        
        print(f"🔄 Replaying from task: {task_id}")
        
        # Optional inputs for replay
        inputs = {"paper_query": "machine learning transformers", "topic": "research", "current_year": datetime.now().year}
        
        # Initialize and replay the crew
        crew_instance = ServerResearchMcp()
        crew = crew_instance.crew()
        
        crew.replay(task_id=task_id, inputs=inputs)
        
        print("✅ Replay completed!")
        
    except subprocess.CalledProcessError as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {e}")

def test():
    """Test the crew using CrewAI's native testing functionality."""
    try:
        from server_research_mcp.crew import ServerResearchMcp
        import sys
        
        # Default test parameters
        n_iterations = 2
        model = "gpt-4o-mini"
        
        # Parse command line arguments if provided
        if len(sys.argv) >= 3:
            try:
                n_iterations = int(sys.argv[2])
            except ValueError:
                print("⚠️  Invalid number of iterations, using default: 2")
                
        if len(sys.argv) >= 4:
            model = sys.argv[3]
        
        print(f"🧪 Testing crew with {n_iterations} iterations using {model}...")
        
        # Initialize and test the crew
        crew_instance = ServerResearchMcp()
        crew = crew_instance.crew()
        
        # Use CrewAI's built-in test method
        # Note: This might require the crew to have test configuration
        crew.test(n_iterations=n_iterations, model=model)
        
        print("✅ Testing completed! Check output for performance metrics.")
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        raise Exception(f"An error occurred while testing the crew: {e}")

if __name__ == "__main__":
    main()
