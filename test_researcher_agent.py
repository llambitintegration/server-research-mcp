#!/usr/bin/env python3
"""Test researcher agent with Zotero tools - limited iterations."""

import os
import sys

def test_researcher_agent():
    """Test researcher agent with Zotero tools."""
    
    # Check for API key
    if not os.getenv('OPENAI_API_KEY'):
        print('⚠️ Skipping - OPENAI_API_KEY not set')
        print('   Set OPENAI_API_KEY environment variable to test with LLM')
        return
    
    try:
        from crewai import Agent, Task, Crew, Process, LLM
        from src.server_research_mcp.tools.mcp_tools import get_researcher_tools
        
        print('🧪 TESTING RESEARCHER AGENT WITH ZOTERO TOOLS')
        print('=' * 50)
        
        # Get Zotero tools
        print('📊 Loading researcher tools...')
        tools = get_researcher_tools()
        zotero_tools = [t for t in tools if 'zotero' in t.name.lower()]
        
        print(f'✅ Found {len(zotero_tools)} Zotero tools:')
        for tool in zotero_tools:
            print(f'  • {tool.name}')
        
        if not zotero_tools:
            print('❌ No Zotero tools available for testing')
            return
        
        # Create LLM and agent
        print('\n🤖 Creating researcher agent...')
        llm = LLM(model='openai/gpt-4o-mini', api_key=os.getenv('OPENAI_API_KEY'))
        
        researcher = Agent(
            role='Research Specialist',
            goal='Search Zotero library for research papers efficiently',
            backstory='Expert at finding academic papers using Zotero search tools with concise reporting',
            tools=zotero_tools,
            verbose=True,
            llm=llm,
            max_iter=5,  # Limited to 5 iterations
            max_execution_time=120  # 2 minute timeout
        )
        
        # Create focused task
        task = Task(
            description="""Search for papers about "machine learning" using the zotero_search_items tool. 
            
            Instructions:
            1. Use zotero_search_items with query "machine learning"
            2. Summarize the found papers (titles, authors, types)
            3. Do NOT try to get full text or detailed metadata
            4. Provide a concise summary and finish
            
            Be efficient - complete this in 2-3 tool calls maximum.""",
            expected_output='Concise summary of machine learning papers found in the Zotero library',
            agent=researcher
        )
        
        # Create crew
        crew = Crew(
            agents=[researcher],
            tasks=[task],
            process=Process.sequential,
            verbose=True,
            memory=False,
            planning=False
        )
        
        print('\n🚀 Executing researcher crew (max 5 iterations)...')
        try:
            result = crew.kickoff()
            print('\n✅ CREW EXECUTION COMPLETED!')
            print('=' * 40)
            print('📋 Final Result:')
            print(result.raw if hasattr(result, 'raw') else str(result))
            
        except Exception as e:
            print(f'\n❌ CREW EXECUTION FAILED: {e}')
            print(f'   Error type: {type(e)}')
            
    except ImportError as e:
        print(f'❌ Import error: {e}')
    except Exception as e:
        print(f'❌ Unexpected error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_researcher_agent() 