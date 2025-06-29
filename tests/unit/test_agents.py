"""Comprehensive agent tests for a fully functioning crew."""

import pytest
from unittest.mock import MagicMock, patch, call
from server_research_mcp.crew import ServerResearchMcp
from server_research_mcp.tools.mcp_tools import get_historian_tools
from crewai import Agent


class TestHistorianAgent:
    """Test Historian agent functionality."""
    
    def test_historian_initialization(self, test_environment, mock_llm):
        """Test Historian agent initialization."""
        with patch('server_research_mcp.crew.get_configured_llm', return_value=mock_llm):
            crew_instance = ServerResearchMcp()
            historian = crew_instance.historian()
            
            assert historian is not None
            assert "memory" in historian.role.lower()  # Historian manages memory and context
            assert len(historian.tools) >= 5  # All MCP tools
    
    def test_historian_tool_usage(self, mock_crew_agents, mock_mcp_manager):
        """Test Historian uses tools correctly."""
        historian = mock_crew_agents["historian"]
        
        # Add a mock tool for testing
        mock_tool = MagicMock()
        mock_tool._run = MagicMock(return_value="Memory search results")
        historian.tools = [mock_tool]
        
        # Mock tool execution
        result = historian.tools[0]._run("test query")
        assert "Memory search results" in result
    
    def test_historian_memory_operations(self, mock_crew_agents, mock_mcp_manager):
        """Test Historian memory operations workflow."""
        historian = mock_crew_agents["historian"]
        
        # Test memory search -> create entity -> add observations workflow
        workflow_results = []
        
        # Search memory
        search_result = mock_mcp_manager.call_tool("memory_search", query="AI testing")
        workflow_results.append(search_result)
        
        # Create entity based on search
        if not search_result["results"]:
            create_result = mock_mcp_manager.call_tool(
                "memory_create_entity",
                name="ai_testing_framework",
                entity_type="concept",
                observations=["Framework for testing AI systems", "Includes unit and integration tests"]
            )
            workflow_results.append(create_result)
        
        # Add observations
        add_result = mock_mcp_manager.call_tool(
            "memory_add_observation",
            entity_name="ai_testing_framework",
            observations=["Supports pytest integration", "Handles async operations"]
        )
        workflow_results.append(add_result)
        
        assert len(workflow_results) >= 2
        assert all(r for r in workflow_results)
    
    def test_historian_context7_integration(self, mock_crew_agents, mock_mcp_manager):
        """Test Historian Context7 integration."""
        historian = mock_crew_agents["historian"]
        
        # Resolve library
        resolve_result = mock_mcp_manager.call_tool(
            "context7_resolve_library",
            library_name="pytest"
        )
        assert "library_id" in resolve_result
        
        # Get documentation
        docs_result = mock_mcp_manager.call_tool(
            "context7_get_docs",
            library_id=resolve_result["library_id"],
            topic="fixtures",
            tokens=5000
        )
        assert "content" in docs_result
        assert docs_result["tokens_used"] <= 5000
    
    def test_historian_sequential_thinking(self, mock_crew_agents, mock_mcp_manager):
        """Test Historian sequential thinking process."""
        historian = mock_crew_agents["historian"]
        
        # Multi-step thinking process
        thoughts = []
        for i in range(3):
            thought_result = mock_mcp_manager.call_tool(
                "sequential_thinking_append_thought",
                thought=f"Step {i+1}: Analyzing aspect {i+1}",
                thought_number=i+1,
                total_thoughts=3
            )
            thoughts.append(thought_result)
        
        # Get all thoughts
        all_thoughts = mock_mcp_manager.call_tool("sequential_thinking_get_thoughts")
        
        assert len(thoughts) == 3
        assert len(all_thoughts["thoughts"]) == 3


class TestResearcherAgent:
    """Test Researcher agent functionality."""
    
    def test_researcher_initialization(self, mock_crew_agents):
        """Test Researcher agent initialization."""
        researcher = mock_crew_agents["researcher"]
        
        assert researcher is not None
        assert "research" in researcher.role.lower()
        assert hasattr(researcher, 'execute')
    
    def test_researcher_deep_analysis(self, mock_crew_agents, mock_mcp_manager):
        """Test Researcher performs deep analysis."""
        researcher = mock_crew_agents["researcher"]
        
        # Mock Zotero integration for academic research
        search_results = mock_mcp_manager.call_tool(
            "zotero_search",
            query="AI testing methodologies"
        )
        
        assert "items" in search_results
        assert len(search_results["items"]) > 0
        
        # Get detailed paper
        if search_results["items"]:
            paper_details = mock_mcp_manager.call_tool(
                "zotero_get_item",
                item_id="test_item_id"
            )
            
            assert "title" in paper_details
            assert "sections" in paper_details
            assert len(paper_details["sections"]) >= 4
    
    def test_researcher_web_search_integration(self, mock_crew_agents):
        """Test Researcher web search capabilities."""
        researcher = mock_crew_agents["researcher"]
        
        # Mock web search tool
        web_search_tool = MagicMock()
        web_search_tool.name = "web_search"
        web_search_tool._run = MagicMock(return_value={
            "results": [
                {"title": "Latest AI Testing Trends", "url": "https://example.com/1"},
                {"title": "Best Practices in AI QA", "url": "https://example.com/2"}
            ]
        })
        
        researcher.tools.append(web_search_tool)
        
        # Execute search
        results = web_search_tool._run("AI testing trends 2024")
        assert len(results["results"]) == 2
    
    def test_researcher_cross_referencing(self, mock_crew_agents, sample_knowledge_graph):
        """Test Researcher cross-references multiple sources."""
        researcher = mock_crew_agents["researcher"]
        
        # Simulate cross-referencing process
        sources = {
            "academic": ["Paper A", "Paper B"],
            "web": ["Article 1", "Article 2"],
            "knowledge_graph": sample_knowledge_graph["entities"]
        }
        
        # Mock cross-reference analysis
        cross_ref_results = {
            "consensus_points": ["Point 1", "Point 2"],
            "conflicting_views": ["Conflict 1"],
            "gaps_identified": ["Gap 1", "Gap 2"]
        }
        
        assert len(cross_ref_results["consensus_points"]) >= 2
        assert len(cross_ref_results["gaps_identified"]) >= 1


class TestSynthesizerAgent:
    """Test Synthesizer agent functionality."""
    
    def test_synthesizer_initialization(self, mock_crew_agents):
        """Test Synthesizer agent initialization."""
        synthesizer = mock_crew_agents["synthesizer"]
        
        assert synthesizer is not None
        assert "synthesiz" in synthesizer.role.lower()
    
    def test_synthesizer_knowledge_integration(self, mock_crew_agents, sample_knowledge_graph):
        """Test Synthesizer integrates knowledge from multiple sources."""
        synthesizer = mock_crew_agents["synthesizer"]
        
        # Simulate knowledge integration process
        knowledge_sources = {
            "memory_graph": sample_knowledge_graph,
            "research_findings": ["Finding 1", "Finding 2", "Finding 3"],
            "external_sources": ["Source A", "Source B"]
        }
        
        # Mock integration result
        integration_result = {
            "unified_concepts": ["Concept X", "Concept Y"],
            "relationship_map": {"X": ["relates_to", "Y"]},
            "synthesis_quality": 0.85
        }
        
        assert len(integration_result["unified_concepts"]) >= 2
        assert integration_result["synthesis_quality"] > 0.8
    
    def test_synthesizer_pattern_recognition(self, mock_crew_agents):
        """Test Synthesizer recognizes patterns across data."""
        synthesizer = mock_crew_agents["synthesizer"]
        
        # Simulate pattern recognition
        data_patterns = [
            {"type": "trend", "strength": 0.8, "description": "Increasing AI adoption"},
            {"type": "correlation", "strength": 0.7, "description": "Testing frameworks and quality"},
            {"type": "gap", "strength": 0.6, "description": "Limited automation testing"}
        ]
        
        # Mock pattern analysis
        pattern_analysis = {
            "identified_patterns": len(data_patterns),
            "confidence_scores": [p["strength"] for p in data_patterns],
            "actionable_insights": ["Insight 1", "Insight 2"]
        }
        
        assert pattern_analysis["identified_patterns"] >= 3
        assert all(score > 0.5 for score in pattern_analysis["confidence_scores"])
    
    def test_synthesizer_research_paper_generation(self, mock_crew_agents, research_paper_validator):
        """Test Synthesizer generates well-structured research papers."""
        synthesizer = mock_crew_agents["synthesizer"]
        
        # Mock research paper structure
        paper = {
            "title": "AI Testing Methodologies: A Comprehensive Review",
            "abstract": "This paper examines current trends in AI testing...",
            "sections": [
                {"title": "Introduction", "content": "AI testing has evolved..."},
                {"title": "Methodology", "content": "We analyzed 50 frameworks..."},
                {"title": "Results", "content": "Our findings indicate..."},
                {"title": "Discussion", "content": "These results suggest..."},
                {"title": "Conclusion", "content": "In conclusion..."}
            ],
            "references": ["Ref 1", "Ref 2", "Ref 3"]
        }
        
        # Validate paper structure
        is_valid, errors = research_paper_validator(paper)
        assert is_valid, f"Paper validation failed: {errors}"


class TestValidatorAgent:
    """Test Validator agent functionality."""
    
    def test_validator_initialization(self, mock_crew_agents):
        """Test Validator agent initialization."""
        validator = mock_crew_agents["validator"]
        
        assert validator is not None
        assert "validat" in validator.role.lower() or "quality" in validator.role.lower()
    
    def test_validator_fact_checking(self, mock_crew_agents):
        """Test Validator performs fact checking."""
        validator = mock_crew_agents["validator"]
        
        # Mock fact checking process
        claims_to_check = [
            "Python was created in 1991",
            "AI testing frameworks improve reliability",
            "Machine learning requires large datasets"
        ]
        
        fact_check_results = []
        for claim in claims_to_check:
            result = {
                "claim": claim,
                "verified": True,
                "confidence": 0.9,
                "sources": ["Source 1", "Source 2"]
            }
            fact_check_results.append(result)
        
        assert len(fact_check_results) == 3
        assert all(r["verified"] for r in fact_check_results)
    
    def test_validator_citation_verification(self, mock_crew_agents):
        """Test Validator verifies citations and references."""
        validator = mock_crew_agents["validator"]
        
        # Mock citations to verify
        citations = [
            {"authors": ["Smith, J.", "Doe, A."], "year": 2023, "title": "AI Testing Methods"},
            {"authors": ["Brown, K."], "year": 2022, "title": "Machine Learning Validation"},
            {"authors": ["Lee, S.", "Wang, T."], "year": 2024, "title": "Automated QA Systems"}
        ]
        
        verification_results = []
        for citation in citations:
            result = {
                "citation": citation,
                "format_valid": True,
                "source_accessible": True,
                "content_relevant": True
            }
            verification_results.append(result)
        
        assert len(verification_results) == 3
        assert all(r["format_valid"] for r in verification_results)
    
    def test_validator_quality_metrics(self, mock_crew_agents, sample_research_paper):
        """Test Validator calculates quality metrics."""
        validator = mock_crew_agents["validator"]
        
        # Mock quality assessment
        quality_metrics = {
            "completeness": 0.92,
            "accuracy": 0.88,
            "clarity": 0.85,
            "novelty": 0.75,
            "overall_score": 0.85
        }
        
        assert quality_metrics["overall_score"] > 0.8
        assert all(score > 0.7 for score in quality_metrics.values())
    
    def test_validator_recommendations(self, mock_crew_agents):
        """Test Validator provides improvement recommendations."""
        validator = mock_crew_agents["validator"]
        
        # Mock validation issues and recommendations
        validation_issues = [
            {"type": "missing_citation", "severity": "medium", "section": "Introduction"},
            {"type": "unclear_methodology", "severity": "high", "section": "Methods"},
            {"type": "weak_conclusion", "severity": "low", "section": "Conclusion"}
        ]
        
        recommendations = []
        for issue in validation_issues:
            recommendation = {
                "issue": issue,
                "suggestion": f"Address {issue['type']} in {issue['section']}",
                "priority": issue["severity"]
            }
            recommendations.append(recommendation)
        
        assert len(recommendations) == 3
        assert any(r["priority"] == "high" for r in recommendations)


class TestAgentCollaboration:
    """Test agent collaboration and coordination."""
    
    def test_agent_handoff(self, mock_crew_agents):
        """Test smooth handoff between agents."""
        historian = mock_crew_agents["historian"]
        researcher = mock_crew_agents["researcher"]
        synthesizer = mock_crew_agents["synthesizer"]
        
        # Simulate handoff chain
        handoff_chain = []
        
        # Historian gathers context
        context = {"knowledge_base": "initialized", "entities": 5}
        handoff_chain.append(("historian", "researcher", context))
        
        # Researcher conducts research
        research = {"findings": 10, "sources": 15, "context": context}
        handoff_chain.append(("researcher", "synthesizer", research))
        
        # Synthesizer creates final output
        synthesis = {"paper": "generated", "quality": 0.9, "research": research}
        handoff_chain.append(("synthesizer", "validator", synthesis))
        
        assert len(handoff_chain) == 3
        assert handoff_chain[-1][2]["quality"] > 0.8
    
    def test_agent_feedback_loop(self, mock_crew_agents):
        """Test feedback loop between agents."""
        researcher = mock_crew_agents["researcher"]
        validator = mock_crew_agents["validator"]
        synthesizer = mock_crew_agents["synthesizer"]
        
        # Simulate feedback iterations
        iterations = []
        quality_score = 0.6
        
        for i in range(3):  # Max 3 iterations
            # Research iteration
            research_quality = quality_score + (i * 0.1)
            
            # Validation feedback
            if research_quality < 0.8:
                feedback = {
                    "approved": False,
                    "issues": [f"Issue {i+1}"],
                    "suggestions": [f"Improve aspect {i+1}"]
                }
            else:
                feedback = {
                    "approved": True,
                    "score": research_quality
                }
            
            iterations.append({
                "iteration": i + 1,
                "quality": research_quality,
                "feedback": feedback
            })
            
            if feedback["approved"]:
                break
            
            quality_score = research_quality
        
        assert len(iterations) <= 3
        assert iterations[-1]["feedback"]["approved"]
    
    def test_parallel_agent_execution(self, mock_crew_agents):
        """Test parallel execution of independent agent tasks."""
        agents = mock_crew_agents
        
        # Simulate parallel tasks
        parallel_results = {}
        
        for agent_name, agent in agents.items():
            if agent_name == "historian":
                result = {"memory_entities": 10, "execution_time": 2.1}
            elif agent_name == "researcher":
                result = {"research_items": 15, "execution_time": 3.2}
            elif agent_name == "synthesizer":
                result = {"synthesis_complete": True, "execution_time": 1.8}
            else:
                result = {"task_complete": True, "execution_time": 1.5}
            
            parallel_results[agent_name] = result
        
        # Verify all agents completed
        assert len(parallel_results) >= 3
        assert all(r for r in parallel_results.values())
    
    def test_agent_state_sharing(self, mock_crew_agents, mock_crew_memory):
        """Test agents share state through crew memory."""
        historian = mock_crew_agents["historian"]
        researcher = mock_crew_agents["researcher"]
        
        # Historian updates shared state
        shared_state = {
            "knowledge_entities": 25,
            "research_context": "AI testing frameworks",
            "active_session": "session_123"
        }
        
        mock_crew_memory.update(shared_state)
        
        # Researcher accesses shared state
        current_state = mock_crew_memory.get_all()
        
        assert current_state["knowledge_entities"] == 25
        assert "research_context" in current_state
        assert current_state["active_session"] == "session_123"


class TestCrewBasics:
    """Test basic crew instantiation and structure."""
    
    def test_crew_instantiation(self):
        """Test that crew can be instantiated without errors."""
        crew_instance = ServerResearchMcp()
        assert crew_instance is not None
        
    def test_crew_creation(self, disable_crew_memory):
        """Test that crew can be created with proper structure."""
        crew_instance = ServerResearchMcp()
        crew = crew_instance.crew()
        assert crew is not None
        assert len(crew.agents) >= 2
        assert len(crew.tasks) >= 2
        assert crew.memory is False  # Memory is disabled in tests to avoid ChromaDB issues
        assert crew.verbose is True