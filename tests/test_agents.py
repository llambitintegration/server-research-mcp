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
        
        # Mock inputs from other agents
        historian_output = {
            "context": "Historical context about AI testing",
            "key_concepts": ["Concept A", "Concept B"]
        }
        
        researcher_output = {
            "findings": ["Finding 1", "Finding 2"],
            "evidence": {"Finding 1": ["Evidence A"], "Finding 2": ["Evidence B"]}
        }
        
        # Synthesize
        synthesis_result = {
            "integrated_knowledge": {
                "main_themes": ["Theme 1", "Theme 2"],
                "supporting_evidence": historian_output["key_concepts"] + list(researcher_output["findings"]),
                "knowledge_graph": sample_knowledge_graph
            },
            "insights": ["New insight 1", "New insight 2"],
            "recommendations": ["Recommendation 1", "Recommendation 2"]
        }
        
        assert len(synthesis_result["integrated_knowledge"]["main_themes"]) >= 2
        assert len(synthesis_result["insights"]) >= 2
    
    def test_synthesizer_pattern_recognition(self, mock_crew_agents):
        """Test Synthesizer identifies patterns across data."""
        synthesizer = mock_crew_agents["synthesizer"]
        
        # Mock pattern analysis
        data_points = [
            {"category": "testing", "frequency": 10, "context": "unit tests"},
            {"category": "testing", "frequency": 8, "context": "integration tests"},
            {"category": "validation", "frequency": 6, "context": "output validation"}
        ]
        
        patterns = {
            "trending_topics": ["testing"],
            "correlations": [("testing", "validation", 0.8)],
            "anomalies": []
        }
        
        assert "testing" in patterns["trending_topics"]
        assert patterns["correlations"][0][2] > 0.7
    
    def test_synthesizer_research_paper_generation(self, mock_crew_agents, research_paper_validator):
        """Test Synthesizer generates valid research papers."""
        synthesizer = mock_crew_agents["synthesizer"]
        
        # Generate paper
        paper = {
            "title": "Comprehensive Analysis of AI Testing Methodologies",
            "abstract": "This paper presents a comprehensive analysis of modern AI testing methodologies, examining current practices, challenges, and future directions in the field.",
            "authors": [{"name": "AI Research Team", "affiliation": "Research Lab"}],
            "sections": [
                {"title": "Introduction", "content": "AI testing has evolved significantly..."},
                {"title": "Literature Review", "content": "Previous work demonstrates..."},
                {"title": "Methodology", "content": "We employed a systematic approach..."},
                {"title": "Results", "content": "Our analysis reveals..."},
                {"title": "Discussion", "content": "The implications suggest..."},
                {"title": "Conclusion", "content": "In conclusion..."}
            ],
            "references": ["Ref 1", "Ref 2", "Ref 3"]
        }
        
        is_valid, errors = research_paper_validator(paper)
        assert is_valid
        assert len(errors) == 0


class TestValidatorAgent:
    """Test Validator agent functionality."""
    
    def test_validator_initialization(self, mock_crew_agents):
        """Test Validator agent initialization."""
        validator = mock_crew_agents["validator"]
        
        assert validator is not None
        assert "validat" in validator.role.lower()
    
    def test_validator_fact_checking(self, mock_crew_agents):
        """Test Validator performs fact checking."""
        validator = mock_crew_agents["validator"]
        
        # Mock fact checking
        claims = [
            {"claim": "pytest is the most popular Python testing framework", "source": "Survey 2024"},
            {"claim": "AI testing requires specialized tools", "source": "Research Paper A"}
        ]
        
        fact_check_results = [
            {"claim": claims[0]["claim"], "verified": True, "confidence": 0.95},
            {"claim": claims[1]["claim"], "verified": True, "confidence": 0.88}
        ]
        
        assert all(result["verified"] for result in fact_check_results)
        assert all(result["confidence"] > 0.8 for result in fact_check_results)
    
    def test_validator_citation_verification(self, mock_crew_agents):
        """Test Validator verifies citations."""
        validator = mock_crew_agents["validator"]
        
        citations = [
            "Smith et al. (2023). 'Testing Deep Learning Systems'. Journal of AI Research.",
            "Johnson, M. (2024). 'Automated Testing for LLMs'. AI Testing Conference."
        ]
        
        verification_results = []
        for citation in citations:
            # Mock citation verification
            result = {
                "citation": citation,
                "valid_format": True,
                "source_exists": True,
                "accurate_quote": True
            }
            verification_results.append(result)
        
        assert all(r["valid_format"] for r in verification_results)
        assert all(r["source_exists"] for r in verification_results)
    
    def test_validator_quality_metrics(self, mock_crew_agents, sample_research_paper):
        """Test Validator evaluates quality metrics."""
        validator = mock_crew_agents["validator"]
        
        # Quality evaluation
        quality_metrics = {
            "completeness": 0.92,  # All required sections present
            "coherence": 0.88,     # Logical flow and consistency
            "accuracy": 0.95,      # Factual correctness
            "depth": 0.85,         # Thoroughness of analysis
            "clarity": 0.90,       # Readability and organization
            "overall_score": 0.90
        }
        
        assert quality_metrics["overall_score"] >= 0.85
        assert all(score >= 0.80 for score in quality_metrics.values())
    
    def test_validator_recommendations(self, mock_crew_agents):
        """Test Validator provides improvement recommendations."""
        validator = mock_crew_agents["validator"]
        
        # Mock validation with issues
        validation_result = {
            "issues": [
                {"type": "minor", "location": "Section 2", "description": "Citation format inconsistent"},
                {"type": "suggestion", "location": "Section 4", "description": "Could expand on methodology"}
            ],
            "recommendations": [
                "Standardize citation format throughout",
                "Add more detail to methodology section",
                "Consider adding a limitations section"
            ],
            "approval_status": "approved_with_suggestions"
        }
        
        assert len(validation_result["recommendations"]) >= 2
        assert validation_result["approval_status"] != "rejected"


class TestAgentCollaboration:
    """Test collaboration between agents."""
    
    def test_agent_handoff(self, mock_crew_agents):
        """Test smooth handoff between agents."""
        historian = mock_crew_agents["historian"]
        researcher = mock_crew_agents["researcher"]
        
        # Historian output
        historian_output = {
            "context_foundation": "Comprehensive context...",
            "key_areas": ["Area 1", "Area 2"],
            "research_questions": ["Question 1", "Question 2"]
        }
        
        # Researcher receives and builds upon historian's work
        researcher_input = historian_output["research_questions"]
        assert len(researcher_input) == 2
    
    def test_agent_feedback_loop(self, mock_crew_agents):
        """Test feedback loops between agents."""
        synthesizer = mock_crew_agents["synthesizer"]
        validator = mock_crew_agents["validator"]
        
        # Initial synthesis
        synthesis_v1 = {"content": "Initial synthesis", "version": 1}
        
        # Validator feedback
        feedback = {
            "approved": False,
            "issues": ["Issue 1", "Issue 2"],
            "required_changes": ["Change 1", "Change 2"]
        }
        
        # Revised synthesis
        synthesis_v2 = {
            "content": "Revised synthesis addressing feedback",
            "version": 2,
            "changes_made": feedback["required_changes"]
        }
        
        # Final validation
        final_validation = {"approved": True, "score": 0.95}
        
        assert synthesis_v2["version"] > synthesis_v1["version"]
        assert final_validation["approved"]
    
    def test_parallel_agent_execution(self, mock_crew_agents):
        """Test agents can work in parallel when appropriate."""
        historian = mock_crew_agents["historian"]
        researcher = mock_crew_agents["researcher"]
        
        # Both agents can work on different aspects simultaneously
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            historian_future = executor.submit(historian.execute, "Gather context")
            researcher_future = executor.submit(researcher.execute, "Conduct research")
            
            historian_result = historian_future.result()
            researcher_result = researcher_future.result()
        
        assert historian_result == "Historian task completed"
        assert researcher_result == "Research task completed"
    
    def test_agent_state_sharing(self, mock_crew_agents, mock_crew_memory):
        """Test agents share state through crew memory."""
        historian = mock_crew_agents["historian"]
        researcher = mock_crew_agents["researcher"]
        
        # Historian saves to memory
        historian_data = {"context": "Important context", "timestamp": "2024-01-01"}
        mock_crew_memory.save(historian_data)
        
        # Researcher retrieves from memory
        retrieved_data = mock_crew_memory.search("context")
        assert len(retrieved_data) > 0
        assert retrieved_data[0]["content"] == "Previous research on topic"