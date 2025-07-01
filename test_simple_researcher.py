#!/usr/bin/env python3
"""Simplified researcher agent test to isolate CrewAI integration issue."""

import os
import logging

# Enable debug logging for our tools only
logging.getLogger('src.server_research_mcp.tools.mcp_tools').setLevel(logging.DEBUG)
logging.getLogger('src.server_research_mcp.utils.mcpadapt').setLevel(logging.DEBUG)

def test_simple_researcher():
    """Test researcher with minimal configuration."""
    
    if not os.getenv('OPENAI_API_KEY'):
        print('⚠️ Skipping - OPENAI_API_KEY not set')
        return
    
    try:
        from crewai import Agent, Task, Crew, Process, LLM
        from src.server_research_mcp.tools.mcp_tools import get_researcher_tools
        
        print('🧪 SIMPLIFIED RESEARCHER AGENT TEST')
        print('=' * 40)
        
        # Get only the search tool
        tools = get_researcher_tools()
        search_tool = None
        for tool in tools:
            if 'search_items' in tool.name:
                search_tool = tool
                break
        
        if not search_tool:
            print('❌ Search tool not found')
            return
        
        print(f'✅ Using tool: {search_tool.name}')
        
        # Create minimal LLM and agent
        llm = LLM(model='openai/gpt-4o-mini', api_key=os.getenv('OPENAI_API_KEY'))
        
        researcher = Agent(
            role='Simple Researcher',
            goal='Use zotero_search_items tool once and report results',
            backstory='I am a simple researcher who uses tools efficiently',
            tools=[search_tool],  # Only one tool
            verbose=True,
            llm=llm,
            max_iter=2,  # Very limited iterations
            allow_delegation=False
        )
        
        # Create very specific task
        task = Task(
            description='''Use the zotero_search_items tool to search for "machine learning".
            
            Call the tool EXACTLY like this:
            zotero_search_items with query="machine learning" and limit=3
            
            Then report what you found and stop.''',
            expected_output='Brief summary of search results',
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
        
        print('\n🚀 Executing simplified researcher...')
        try:
            result = crew.kickoff()
            print('\n✅ SUCCESS! Crew completed.')
            print('📋 Result:')
            print(result.raw if hasattr(result, 'raw') else str(result))
            return True
            
        except Exception as e:
            print(f'\n❌ CREW FAILED: {e}')
            return False
            
    except Exception as e:
        print(f'❌ Setup failed: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_simple_researcher()
    if success:
        print('\n🎉 CrewAI integration working!')
    else:
        print('\n💥 CrewAI integration has issues') 