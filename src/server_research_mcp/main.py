#!/usr/bin/env python3
"""Main entry point for the MCP Server Research application."""

import os
import sys
import argparse
import asyncio
from typing import Dict, Any, Optional
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from server_research_mcp.crew import ServerResearchMcpCrew
from server_research_mcp.config.llm_config import check_llm_config, get_llm_config
from server_research_mcp.utils.logging_config import setup_logging, get_logger, crew_execution, get_symbol
from server_research_mcp.utils.log_context import log_execution, log_context

# Initialize logging
setup_logging()
logger = get_logger(__name__)

@log_execution
def validate_environment() -> bool:
    """Validate that all required environment variables are set."""
    logger.info("Starting environment validation")
    
    with log_context(operation="check_llm_config"):
        is_valid, error_msg = check_llm_config()
        if not is_valid:
            logger.error(f"LLM configuration validation failed: {error_msg}")
            return False
        
        llm_config = get_llm_config()
        logger.info(f"{get_symbol('success')} LLM configuration valid: provider={llm_config['provider']}, model={llm_config['model']}")
    
    # Check for required directories
    required_dirs = ['outputs', 'knowledge', 'checkpoints']
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            logger.warning(f"Creating missing directory: {dir_path}")
            dir_path.mkdir(parents=True, exist_ok=True)
        else:
            logger.debug(f"{get_symbol('success')} Directory exists: {dir_path}")
    
    logger.info("Environment validation completed successfully")
    return True


@log_execution
def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    logger.info("Parsing command line arguments")
    
    parser = argparse.ArgumentParser(
        description="MCP Server Research - AI-powered research crew with MCP tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m server_research_mcp.main "machine learning transformers"
  python -m server_research_mcp.main "quantum computing applications" --output-dir ./research_outputs
        """
    )
    
    parser.add_argument(
        "topic",
        help="Research topic or query to investigate"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to store research outputs (default: outputs)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true", 
        help="Validate configuration without running research"
    )
    
    args = parser.parse_args()
    
    logger.info(f"Parsed arguments: topic='{args.topic}', output_dir='{args.output_dir}', verbose={args.verbose}, dry_run={args.dry_run}")
    
    return args


@log_execution
def setup_output_directory(output_dir: str) -> Path:
    """Create and validate output directory."""
    logger.info(f"Setting up output directory: {output_dir}")
    
    output_path = Path(output_dir)
    
    try:
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"{get_symbol('success')} Output directory ready: {output_path.absolute()}")
        
        # Test write permissions
        test_file = output_path / ".write_test"
        test_file.write_text("test")
        test_file.unlink()
        logger.debug(f"{get_symbol('success')} Output directory is writable")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to setup output directory: {e}", exc_info=True)
        raise


@log_execution
async def initialize_research_crew(inputs: Optional[Dict[str, Any]] = None) -> ServerResearchMcpCrew:
    """Initialize the research crew with comprehensive logging."""
    logger.info("Initializing research crew")
    
    try:
        with log_context(operation="create_research_crew"):
            crew = ServerResearchMcpCrew(inputs=inputs)  # Pass inputs during initialization
            logger.info(f"{get_symbol('success')} Research crew instance created")
        
        with log_context(operation="load_crew_config"):
            # This will trigger the crew initialization
            crew_instance = crew.crew()
            logger.info(f"{get_symbol('success')} Crew initialized with {len(crew_instance.agents)} agents")
            
            # Log agent details
            for i, agent in enumerate(crew_instance.agents):
                logger.info(f"  Agent {i+1}: {agent.role} with {len(agent.tools)} tools")
        
        return crew
        
    except Exception as e:
        logger.error(f"Failed to initialize research crew: {e}", exc_info=True)
        raise


@log_execution
async def run_research(crew: ServerResearchMcpCrew, topic: str, output_dir: Path) -> Dict[str, Any]:
    """Execute the research workflow."""
    logger.info(f"Starting research workflow for topic: '{topic}'")
    
    try:
        with log_context(operation="prepare_research_inputs"):
            # Handle both string topics and Namespace objects
            if hasattr(topic, 'topic'):
                # It's a Namespace object from argparse
                inputs = {
                    'topic': topic.topic,
                    'output_dir': str(output_dir),
                    'research_depth': 'comprehensive'
                }
                # Add paper_query if available for backward compatibility
                if hasattr(topic, 'query'):
                    inputs['paper_query'] = topic.query
            else:
                # It's a string topic
                inputs = {
                    'topic': topic,
                    'output_dir': str(output_dir),
                    'research_depth': 'comprehensive'
                }
            logger.info(f"Research inputs prepared: {inputs}")
        
        with log_context(operation="execute_crew_kickoff"):
            logger.info("Executing crew kickoff...")
            with crew_execution():
                result = crew.crew().kickoff(inputs=inputs)
            logger.info(f"{get_symbol('success')} Crew execution completed successfully")
        
        # Process and log results
        if hasattr(result, 'raw'):
            logger.info(f"Research completed. Raw result length: {len(str(result.raw))} characters")
        else:
            logger.info(f"Research completed. Result type: {type(result)}")
        
        return {
            'status': 'success',
            'result': result,
            'topic': topic,
            'output_dir': str(output_dir)
        }
        
    except Exception as e:
        logger.error(f"Research workflow failed: {e}", exc_info=True)
        return {
            'status': 'error',
            'error': str(e),
            'topic': topic,
            'output_dir': str(output_dir)
        }


def get_user_input() -> str:
    """Get research topic from user input with validation."""
    while True:
        try:
            topic = input("Enter your research topic: ").strip()
            if not topic:
                print("Please enter a valid research topic.")
                continue
                
            while True:
                confirmation = input(f"Research topic: '{topic}' - Continue? (y/n): ").strip().lower()
                if confirmation in ['y', 'yes']:
                    return topic
                elif confirmation in ['n', 'no']:
                    print("Research cancelled by user.")
                    sys.exit(0)
                else:
                    print("Please enter 'y' for yes or 'n' for no.")
                
        except KeyboardInterrupt:
            print("\nResearch cancelled by user.")
            raise  # Re-raise KeyboardInterrupt for tests that expect it


async def run_crew(topic: str, output_dir: str = "outputs") -> dict:
    """Run the research crew with given topic."""
    logger.info(f"Running crew for topic: {topic}")
    
    try:
        # Setup output directory
        output_path = setup_output_directory(output_dir)
        
        # Initialize and run crew with topic inputs
        crew_inputs = {'topic': topic}
        crew = await initialize_research_crew(inputs=crew_inputs)
        result = await run_research(crew, topic, output_path)
        
        return result
        
    except Exception as e:
        logger.error(f"Crew execution failed: {e}", exc_info=True)
        return {
            'status': 'error',
            'error': str(e)
        }


def main_with_args(args) -> Dict[str, Any]:
    """Run main function with provided arguments instead of parsing from command line.
    
    This is useful for testing and programmatic usage.
    Returns a dictionary with status and result information.
    """
    logger.info(f"{get_symbol('rocket')} Starting MCP Server Research application")
    
    try:
        # Validate environment
        if not validate_environment():
            logger.error(f"{get_symbol('error')} Environment validation failed")
            return {'status': 'error', 'error': 'Environment validation failed'}
        
        # Setup output directory
        output_dir = setup_output_directory(args.output_dir)
        
        # Dry run check
        if args.dry_run:
            logger.info(f"{get_symbol('success')} Dry run completed - configuration is valid")
            return {'status': 'success', 'message': 'Dry run completed successfully'}
        
        # Initialize crew with topic inputs
        try:
            crew_inputs = {'topic': args.topic}
            crew = asyncio.run(initialize_research_crew(inputs=crew_inputs))
        except Exception as e:
            logger.error(f"Failed to initialize research crew: {e}")
            return {'status': 'error', 'error': f'Failed to initialize research crew: {e}'}
        
        # Run research
        result = asyncio.run(run_research(crew, args.topic, output_dir))
        
        if result['status'] == 'success':
            logger.info(f"{get_symbol('party')} Research completed successfully!")
            return {'status': 'success', 'result': result, 'output_dir': str(output_dir)}
        else:
            logger.error(f"{get_symbol('error')} Research failed: {result['error']}")
            return {'status': 'error', 'error': result['error']}
            
    except KeyboardInterrupt:
        logger.info(f"{get_symbol('wave')} Application interrupted by user")
        return {'status': 'cancelled', 'message': 'Application interrupted by user'}
    except Exception as e:
        logger.error(f"{get_symbol('boom')} Unexpected error in main: {e}", exc_info=True)
        return {'status': 'error', 'error': str(e)}


def run():
    """Legacy compatibility function - runs main with default arguments."""
    import asyncio
    asyncio.run(main())


async def main():
    """Main application entry point."""
    logger.info(f"{get_symbol('rocket')} Starting MCP Server Research application")
    
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Validate environment
        if not validate_environment():
            logger.error(f"{get_symbol('error')} Environment validation failed")
            sys.exit(1)
        
        # Setup output directory
        output_dir = setup_output_directory(args.output_dir)
        
        # Dry run check
        if args.dry_run:
            logger.info(f"{get_symbol('success')} Dry run completed - configuration is valid")
            return
        
        # Initialize crew with topic inputs
        crew_inputs = {'topic': args.topic}
        crew = await initialize_research_crew(inputs=crew_inputs)
        
        # Run research
        result = await run_research(crew, args.topic, output_dir)
        
        if result['status'] == 'success':
            logger.info(f"{get_symbol('party')} Research completed successfully!")
            print(f"Research results saved to: {output_dir}")
        else:
            logger.error(f"{get_symbol('error')} Research failed: {result['error']}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info(f"{get_symbol('wave')} Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"{get_symbol('boom')} Unexpected error in main: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
