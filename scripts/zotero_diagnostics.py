#!/usr/bin/env python3
"""
Consolidated Zotero MCP Diagnostic Tool

This tool consolidates functionality from multiple diagnostic scripts:
- check_zotero.py: Environment and credential checks
- debug_zotero_params.py: Parameter passing simulation
- test_zotero_debug.py: Direct tool testing with debug logging
- test_zotero_integration.py: MCP server startup and integration testing
- test_simple_researcher.py: Simple CrewAI agent testing
- test_researcher_agent.py: Full researcher agent testing

Usage:
    python scripts/zotero_diagnostics.py env              # Check environment setup
    python scripts/zotero_diagnostics.py params           # Debug parameter passing
    python scripts/zotero_diagnostics.py tools            # Test tools directly
    python scripts/zotero_diagnostics.py server           # Test MCP server startup
    python scripts/zotero_diagnostics.py agent            # Test CrewAI integration
    python scripts/zotero_diagnostics.py all              # Run all diagnostics
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
from typing import Dict, List, Optional


class ZoteroDiagnostics:
    """Consolidated Zotero MCP diagnostic tool."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        if verbose:
            logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')
        else:
            logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)

    def check_environment(self) -> bool:
        """Check environment variables and dependencies."""
        print("🔍 ZOTERO ENVIRONMENT CHECK")
        print("=" * 40)
        
        success = True
        
        # Check environment variables
        api_key = os.getenv("ZOTERO_API_KEY")
        library_id = os.getenv("ZOTERO_LIBRARY_ID")
        
        print(f"ZOTERO_API_KEY: {'SET' if api_key else 'MISSING'}")
        if api_key:
            print(f"  Length: {len(api_key)} characters")
            print(f"  Format: {'Valid' if len(api_key) >= 20 and api_key.isalnum() else 'Invalid'}")
            if not (len(api_key) >= 20 and api_key.isalnum()):
                success = False
        else:
            success = False
        
        print(f"ZOTERO_LIBRARY_ID: {'SET' if library_id else 'MISSING'}")
        if library_id:
            print(f"  Value: {library_id}")
            print(f"  Format: {'Valid' if library_id.isdigit() else 'Invalid'}")
            if not library_id.isdigit():
                success = False
        else:
            success = False
        
        print()
        
        # Check uvx availability
        print("🔧 UVX AVAILABILITY CHECK")
        print("=" * 30)
        try:
            result = subprocess.run(["uvx", "--version"], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"✅ uvx available: {result.stdout.strip()}")
            else:
                print(f"❌ uvx error: {result.stderr.strip()}")
                success = False
        except FileNotFoundError:
            print("❌ uvx not found - install uv package manager")
            success = False
        except Exception as e:
            print(f"❌ uvx check failed: {e}")
            success = False
        
        print()
        
        # Check zotero-mcp package
        print("📦 ZOTERO-MCP PACKAGE CHECK")
        print("=" * 35)
        try:
            result = subprocess.run(["uvx", "zotero-mcp", "--help"], capture_output=True, text=True, timeout=15)
            if result.returncode == 0 or "zotero" in result.stderr.lower():
                print("✅ zotero-mcp package available")
            else:
                print("⚠️ zotero-mcp package may need installation")
                if self.verbose:
                    print(f"Output: {(result.stdout or result.stderr)[:200]}...")
        except Exception as e:
            print(f"❌ zotero-mcp check failed: {e}")
            success = False
        
        print()
        
        # Show recommendations if needed
        if not success:
            print("💡 RECOMMENDATIONS")
            print("=" * 20)
            
            if not api_key:
                print("• Set ZOTERO_API_KEY environment variable")
            elif not (len(api_key) >= 20 and api_key.isalnum()):
                print("• Check ZOTERO_API_KEY format (should be 40-character alphanumeric)")
            
            if not library_id:
                print("• Set ZOTERO_LIBRARY_ID environment variable")
            elif not library_id.isdigit():
                print("• Check ZOTERO_LIBRARY_ID format (should be numeric)")
            
            if not api_key or not library_id:
                print("• Visit https://www.zotero.org/settings/keys to create API credentials")
                print("• Get your library ID from the URL when viewing your library")
        
        return success

    def debug_parameter_passing(self) -> bool:
        """Debug parameter passing through MCPToolWrapper."""
        print("🧪 DEBUGGING ZOTERO PARAMETER PASSING")
        print("=" * 50)
        
        # Simulate CrewAI call
        crewai_input = '{"query": "machine learning", "qmode": "everything", "limit": 5}'
        print(f"CrewAI input: {crewai_input}")
        
        # Simulate MCPToolWrapper parameter processing
        args = (crewai_input,)
        kwargs = {}
        
        print(f"Initial - args: {args}, kwargs: {kwargs}")
        
        # Step 1: CrewAI sometimes wraps kwargs in a single "properties" dict
        if len(kwargs) == 1 and 'properties' in kwargs:
            kwargs = kwargs['properties']
            print(f"After properties unwrap - args: {args}, kwargs: {kwargs}")

        # Step 2: CrewAI often passes **all** parameters as a *single* JSON string
        if len(args) == 1 and isinstance(args[0], str):
            arg_str = args[0]
            print(f"Processing JSON string: {arg_str}")
            
            try:
                # Some CrewAI versions double-encode JSON
                for i in range(2):  # at most two decoding passes
                    if isinstance(arg_str, str):
                        try:
                            loaded = json.loads(arg_str)
                            print(f"JSON decode pass {i+1}: {loaded}")
                        except json.JSONDecodeError:
                            loaded = arg_str
                            print(f"Not JSON at pass {i+1}: {loaded}")
                            break
                        arg_str = loaded
                    else:
                        loaded = arg_str
                        print(f"Final loaded: {loaded}")
                        break

                # After potential double-decoding we have `loaded`
                if isinstance(loaded, dict):
                    print(f"Loaded is dict: {loaded}")
                    if 'query' in loaded and len(loaded) == 1:
                        kwargs.setdefault('query', loaded['query'])
                        args = tuple()
                        print(f"Single query conversion - args: {args}, kwargs: {kwargs}")
                    else:
                        for k, v in loaded.items():
                            kwargs.setdefault(k, v)
                        args = tuple()
                        print(f"Multi-param conversion - args: {args}, kwargs: {kwargs}")
                else:
                    args = (loaded,)
                    print(f"Loaded not dict - args: {args}, kwargs: {kwargs}")
            except json.JSONDecodeError:
                print("JSON decode failed")

        print(f"After JSON processing - args: {args}, kwargs: {kwargs}")

        # CRITICAL FIX check
        tool_name = "zotero_search_items"
        if 'search' in tool_name.lower() and kwargs:
            print(f"Zotero search tool detected: {tool_name}, keeping kwargs format")
        elif not args and len(kwargs) == 1 and 'query' in kwargs:
            print(f"Converting single query kwarg to positional: {kwargs['query']}")
            args = (kwargs['query'],)
            kwargs = {}
            print(f"After conversion: args={args}, kwargs={kwargs}")

        print(f"Final parameters - args: {args}, kwargs: {kwargs}")
        
        # Validation check
        if 'search' in tool_name.lower() and not args and not kwargs:
            print("❌ VALIDATION FAILED: No parameters provided")
            return False
        else:
            print("✅ VALIDATION PASSED: Parameters available")
            return True

    def test_tools_directly(self) -> bool:
        """Test tools directly with debug logging."""
        print("🔍 TESTING DIRECT TOOL CALLS")
        print("=" * 35)
        
        try:
            # Enable debug logging for specific modules
            logging.getLogger('src.server_research_mcp.tools.mcp_tools').setLevel(logging.DEBUG)
            logging.getLogger('src.server_research_mcp.utils.mcpadapt').setLevel(logging.DEBUG)
            
            from src.server_research_mcp.tools.mcp_tools import get_researcher_tools
            
            # Get tools
            tools = get_researcher_tools()
            search_tool = None
            
            print(f"📊 Total researcher tools: {len(tools)}")
            for i, tool in enumerate(tools):
                print(f"  {i+1}. {tool.name}")
                if 'search_items' in tool.name:
                    search_tool = tool
            
            if not search_tool:
                print("❌ Zotero search tool not found")
                return False
            
            print(f"✅ Found search tool: {search_tool.name}")
            
            # Test 1: Direct call with string (simulates CrewAI)
            print("\n🧪 Test 1: Direct call with JSON string")
            test_input = '{"query": "machine learning", "qmode": "everything", "limit": 3}'
            print(f"Input: {test_input}")
            
            try:
                result = search_tool._run(test_input)
                print(f"✅ Success! Result length: {len(str(result))}")
                if self.verbose:
                    print(f"Result preview: {str(result)[:200]}...")
                return True
            except Exception as e:
                print(f"❌ Failed: {e}")
                if self.verbose:
                    import traceback
                    traceback.print_exc()
                return False
                
        except Exception as e:
            print(f"❌ Test setup failed: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return False

    async def test_mcp_server_startup(self) -> bool:
        """Test MCP server startup and tool loading."""
        print("🔧 TESTING MCP SERVER STARTUP")
        print("=" * 35)
        
        try:
            from src.server_research_mcp.utils.mcpadapt import MCPAdapt, CrewAIAdapter
            from mcp import StdioServerParameters
            
            # Get credentials
            api_key = os.getenv("ZOTERO_API_KEY")
            library_id = os.getenv("ZOTERO_LIBRARY_ID")
            
            if not api_key or not library_id:
                print("❌ Missing Zotero credentials")
                return False
            
            print(f"✅ Credentials found - API key: {len(api_key)} chars, Library ID: {library_id}")
            
            # Configure Zotero server
            zotero_config = StdioServerParameters(
                command="uvx",
                args=["zotero-mcp"],
                env={
                    "ZOTERO_LOCAL": "false",
                    "ZOTERO_API_KEY": api_key,
                    "ZOTERO_LIBRARY_ID": library_id
                }
            )
            
            print("🚀 Starting Zotero MCP server...")
            
            try:
                with MCPAdapt([zotero_config], CrewAIAdapter()) as tools:
                    print(f"✅ Server started successfully!")
                    print(f"📊 Total tools loaded: {len(tools)}")
                    
                    if tools:
                        if self.verbose:
                            print("\n🔧 Available tools:")
                            for i, tool in enumerate(tools):
                                print(f"  {i+1}. {tool.name}")
                        
                        # Test tool execution
                        zotero_tools = [tool for tool in tools if 'zotero' in tool.name.lower()]
                        
                        if zotero_tools:
                            test_tool = zotero_tools[0]
                            print(f"🧪 Testing tool: {test_tool.name}")
                            
                            try:
                                result = test_tool._run("test search")
                                print(f"✅ Tool execution successful!")
                                if self.verbose:
                                    print(f"📋 Result preview: {str(result)[:200]}...")
                                return True
                            except Exception as e:
                                print(f"❌ Tool execution failed: {e}")
                                return False
                        else:
                            print("⚠️ No Zotero-specific tools found")
                            return False
                    else:
                        print("❌ No tools loaded from server")
                        return False
                        
            except Exception as e:
                print(f"❌ Server startup failed: {e}")
                if self.verbose:
                    import traceback
                    traceback.print_exc()
                return False
        
        except ImportError as e:
            print(f"❌ Import error: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return False

    def test_crewai_agent(self, simple: bool = True) -> bool:
        """Test CrewAI agent integration."""
        if simple:
            print("🧪 TESTING SIMPLE CREWAI INTEGRATION")
            print("=" * 40)
        else:
            print("🧪 TESTING FULL CREWAI AGENT")
            print("=" * 35)
        
        # Check for API key
        if not os.getenv('OPENAI_API_KEY'):
            print('⚠️ Skipping - OPENAI_API_KEY not set')
            print('   Set OPENAI_API_KEY environment variable to test with LLM')
            return False
        
        try:
            from crewai import Agent, Task, Crew, Process, LLM
            from src.server_research_mcp.tools.mcp_tools import get_researcher_tools
            
            # Get tools
            tools = get_researcher_tools()
            zotero_tools = [t for t in tools if 'zotero' in t.name.lower()]
            
            if not zotero_tools:
                print('❌ No Zotero tools available for testing')
                return False
            
            print(f'✅ Found {len(zotero_tools)} Zotero tools')
            if self.verbose:
                for tool in zotero_tools:
                    print(f'  • {tool.name}')
            
            # Create LLM and agent
            llm = LLM(model='openai/gpt-4o-mini', api_key=os.getenv('OPENAI_API_KEY'))
            
            if simple:
                # Use only one tool for simplicity
                search_tool = next((t for t in zotero_tools if 'search_items' in t.name), zotero_tools[0])
                
                researcher = Agent(
                    role='Simple Researcher',
                    goal='Use zotero_search_items tool once and report results',
                    backstory='I am a simple researcher who uses tools efficiently',
                    tools=[search_tool],
                    verbose=self.verbose,
                    llm=llm,
                    max_iter=2,
                    allow_delegation=False
                )
                
                task = Task(
                    description='''Use the zotero_search_items tool to search for "machine learning" with limit=3.
                    Then report what you found and stop.''',
                    expected_output='Brief summary of search results',
                    agent=researcher
                )
            else:
                researcher = Agent(
                    role='Research Specialist',
                    goal='Search Zotero library for research papers efficiently',
                    backstory='Expert at finding academic papers using Zotero search tools with concise reporting',
                    tools=zotero_tools,
                    verbose=self.verbose,
                    llm=llm,
                    max_iter=5,
                    max_execution_time=120
                )
                
                task = Task(
                    description="""Search for papers about "machine learning" using the zotero_search_items tool. 
                    
                    Instructions:
                    1. Use zotero_search_items with query "machine learning"
                    2. Summarize the found papers (titles, authors, types)
                    3. Provide a concise summary and finish
                    
                    Be efficient - complete this in 2-3 tool calls maximum.""",
                    expected_output='Concise summary of machine learning papers found in the Zotero library',
                    agent=researcher
                )
            
            # Create crew
            crew = Crew(
                agents=[researcher],
                tasks=[task],
                process=Process.sequential,
                verbose=self.verbose,
                memory=False,
                planning=False
            )
            
            print(f'\n🚀 Executing {"simple" if simple else "full"} researcher crew...')
            try:
                result = crew.kickoff()
                print('\n✅ CREW EXECUTION COMPLETED!')
                if self.verbose:
                    print('📋 Final Result:')
                    print(result.raw if hasattr(result, 'raw') else str(result))
                return True
                
            except Exception as e:
                print(f'\n❌ CREW EXECUTION FAILED: {e}')
                if self.verbose:
                    import traceback
                    traceback.print_exc()
                return False
                
        except Exception as e:
            print(f'❌ Setup failed: {e}')
            if self.verbose:
                import traceback
                traceback.print_exc()
            return False

    async def run_all_diagnostics(self) -> Dict[str, bool]:
        """Run all diagnostic tests and return results."""
        print("🔬 COMPREHENSIVE ZOTERO MCP DIAGNOSTIC")
        print("=" * 50)
        
        results = {}
        
        print("\n1. Environment Check")
        print("-" * 20)
        results['environment'] = self.check_environment()
        
        print("\n2. Parameter Passing")
        print("-" * 20)
        results['parameters'] = self.debug_parameter_passing()
        
        print("\n3. Direct Tool Testing")
        print("-" * 22)
        results['tools'] = self.test_tools_directly()
        
        print("\n4. MCP Server Startup")
        print("-" * 22)
        results['server'] = await self.test_mcp_server_startup()
        
        print("\n5. CrewAI Integration")
        print("-" * 21)
        results['agent'] = self.test_crewai_agent(simple=True)
        
        # Summary
        print("\n" + "="*50)
        print("🎯 DIAGNOSTIC SUMMARY")
        print("="*50)
        
        for test_name, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{test_name.upper():20} {status}")
        
        total_passed = sum(results.values())
        total_tests = len(results)
        print(f"\nOverall: {total_passed}/{total_tests} tests passed")
        
        if total_passed == total_tests:
            print("🎉 All diagnostics passed! Zotero MCP integration is working correctly.")
        else:
            print("⚠️ Some diagnostics failed. Check the detailed output above for troubleshooting.")
        
        return results


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Consolidated Zotero MCP Diagnostic Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/zotero_diagnostics.py env              # Check environment setup
  python scripts/zotero_diagnostics.py params           # Debug parameter passing
  python scripts/zotero_diagnostics.py tools            # Test tools directly
  python scripts/zotero_diagnostics.py server           # Test MCP server startup
  python scripts/zotero_diagnostics.py agent            # Test CrewAI integration
  python scripts/zotero_diagnostics.py all              # Run all diagnostics
  python scripts/zotero_diagnostics.py all --verbose    # Run all with verbose output
        """
    )
    
    parser.add_argument(
        'mode',
        choices=['env', 'params', 'tools', 'server', 'agent', 'all'],
        help='Diagnostic mode to run'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output and debug logging'
    )
    
    parser.add_argument(
        '--full-agent',
        action='store_true',
        help='Run full agent test instead of simple version (for agent mode)'
    )
    
    args = parser.parse_args()
    
    # Create diagnostics instance
    diagnostics = ZoteroDiagnostics(verbose=args.verbose)
    
    # Run selected diagnostic
    if args.mode == 'env':
        success = diagnostics.check_environment()
    elif args.mode == 'params':
        success = diagnostics.debug_parameter_passing()
    elif args.mode == 'tools':
        success = diagnostics.test_tools_directly()
    elif args.mode == 'server':
        success = asyncio.run(diagnostics.test_mcp_server_startup())
    elif args.mode == 'agent':
        success = diagnostics.test_crewai_agent(simple=not args.full_agent)
    elif args.mode == 'all':
        results = asyncio.run(diagnostics.run_all_diagnostics())
        success = all(results.values())
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 