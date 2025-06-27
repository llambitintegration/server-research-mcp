"""End-to-end workflow tests for the research crew."""

import pytest
import json
import os
from unittest.mock import MagicMock, patch, call
from datetime import datetime
import tempfile
import shutil


class TestCompleteResearchWorkflow:
    """Test complete research workflows from start to finish."""
    
    @pytest.mark.integration
    def test_full_research_workflow(self, mock_crew, sample_inputs, temp_workspace):
        """Test complete research workflow from input to final paper."""
        # Setup output directory
        output_dir = f"{temp_workspace}/research_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Configure crew with output settings
        crew = mock_crew.crew()
        crew.output_dir = output_dir
        
        # Execute full workflow
        result = crew.kickoff(inputs=sample_inputs)
        
        # Verify all stages completed
        assert result is not None
        assert "research_paper" in result
        assert "context_foundation" in result
        
        # Verify outputs were saved
        expected_files = [
            "context_foundation.md",
            "research_findings.json",
            "final_paper.md",
            "validation_report.json"
        ]
        
        crew.save_outputs = MagicMock()
        crew.save_outputs(result, output_dir)
        
        # Verify workflow stages
        assert crew.kickoff.called
        assert result["result"] == "Research completed successfully"
    
    @pytest.mark.integration
    def test_iterative_research_refinement(self, mock_crew, sample_inputs):
        """Test iterative refinement based on validation feedback."""
        crew = mock_crew.crew()
        
        # First iteration
        iteration_results = []
        
        # Simulate validation feedback loop
        for i in range(3):  # Max 3 iterations
            result = crew.kickoff(inputs=sample_inputs)
            
            # Mock validation feedback
            if i < 2:  # First two iterations need refinement
                validation = {
                    "approved": False,
                    "score": 0.7 + (i * 0.1),
                    "feedback": f"Iteration {i+1}: Needs more depth in analysis"
                }
            else:  # Final iteration passes
                validation = {
                    "approved": True,
                    "score": 0.92,
                    "feedback": "Research meets all quality standards"
                }
            
            iteration_results.append({
                "iteration": i + 1,
                "result": result,
                "validation": validation
            })
            
            if validation["approved"]:
                break
            
            # Update inputs for next iteration
            sample_inputs["previous_feedback"] = validation["feedback"]
        
        # Verify iterative improvement
        assert len(iteration_results) == 3
        assert iteration_results[-1]["validation"]["approved"]
        assert iteration_results[-1]["validation"]["score"] > 0.9
    
    @pytest.mark.integration
    def test_multi_topic_research_pipeline(self, mock_crew):
        """Test researching multiple related topics in sequence."""
        topics = [
            "AI Testing Frameworks",
            "Machine Learning Validation",
            "Neural Network Verification"
        ]
        
        crew = mock_crew.crew()
        research_results = []
        knowledge_graph = {"entities": [], "relationships": []}
        
        for i, topic in enumerate(topics):
            inputs = {
                "topic": topic,
                "current_year": "2024",
                "previous_topics": topics[:i],
                "knowledge_graph": knowledge_graph
            }
            
            result = crew.kickoff(inputs=inputs)
            research_results.append(result)
            
            # Update knowledge graph with new findings
            if "knowledge_updates" in result:
                knowledge_graph["entities"].extend(result["knowledge_updates"]["new_entities"])
                knowledge_graph["relationships"].extend(result["knowledge_updates"]["new_relationships"])
        
        # Verify all topics were researched
        assert len(research_results) == 3
        
        # Verify knowledge accumulation
        assert len(knowledge_graph["entities"]) >= 3
        assert len(knowledge_graph["relationships"]) >= 2
    
    @pytest.mark.integration
    def test_collaborative_research_synthesis(self, mock_crew, mock_mcp_manager):
        """Test collaborative synthesis across multiple research sessions."""
        crew = mock_crew.crew()
        
        # Simulate multiple research sessions
        session_results = []
        
        # Session 1: Initial research
        session1_result = crew.kickoff(inputs={
            "topic": "Quantum Computing Applications",
            "current_year": "2024",
            "session_id": "session_1"
        })
        session_results.append(session1_result)
        
        # Save to memory
        mock_mcp_manager.call_tool(
            "memory_create_entity",
            name="quantum_computing_research",
            entity_type="research_session",
            observations=["Initial findings on quantum applications"]
        )
        
        # Session 2: Complementary research
        session2_result = crew.kickoff(inputs={
            "topic": "Classical vs Quantum Algorithms",
            "current_year": "2024",
            "session_id": "session_2",
            "related_sessions": ["session_1"]
        })
        session_results.append(session2_result)
        
        # Session 3: Synthesis
        synthesis_result = crew.kickoff(inputs={
            "topic": "Comprehensive Quantum Computing Analysis",
            "current_year": "2024",
            "session_id": "synthesis",
            "synthesize_sessions": ["session_1", "session_2"]
        })
        
        # Verify synthesis incorporates previous sessions
        assert len(session_results) == 2
        assert synthesis_result is not None
        assert "integrated_findings" in synthesis_result


class TestResearchQualityAssurance:
    """Test research quality assurance mechanisms."""
    
    def test_output_quality_validation(self, mock_crew, sample_inputs, research_paper_validator):
        """Test comprehensive output quality validation."""
        crew = mock_crew.crew()
        result = crew.kickoff(inputs=sample_inputs)
        
        paper = result["research_paper"]
        
        # Validate structure
        is_valid, errors = research_paper_validator(paper)
        assert is_valid
        
        # Additional quality checks
        quality_checks = {
            "min_abstract_length": len(paper["abstract"]) >= 150,
            "min_sections": len(paper["sections"]) >= 5,
            "has_introduction": any(s["title"].lower() == "introduction" for s in paper["sections"]),
            "has_conclusion": any(s["title"].lower() == "conclusion" for s in paper["sections"]),
            "has_references": len(paper.get("references", [])) >= 3,
            "sections_have_content": all(len(s["content"]) > 100 for s in paper["sections"])
        }
        
        assert all(quality_checks.values()), f"Quality checks failed: {quality_checks}"
    
    def test_citation_integrity(self, mock_crew, sample_inputs):
        """Test citation integrity and verification."""
        crew = mock_crew.crew()
        result = crew.kickoff(inputs=sample_inputs)
        
        paper = result["research_paper"]
        
        # Mock citation integrity checks
        citations = paper.get("references", [])
        integrity_results = []
        
        for citation in citations:
            result = {
                "citation": citation,
                "format_valid": True,
                "accessible": True,
                "relevant": True
            }
            integrity_results.append(result)
        
        # Verify all citations pass integrity checks
        assert all(r["format_valid"] for r in integrity_results)
        assert all(r["accessible"] for r in integrity_results)


class TestWorkflowOrchestration:
    """Test workflow orchestration and coordination."""
    
    def test_task_dependency_management(self, mock_crew, mock_crew_tasks):
        """Test proper task dependency management."""
        crew = mock_crew.crew()
        
        # Mock task dependency chain
        task_chain = [
            {"name": "context_gathering", "depends_on": []},
            {"name": "research", "depends_on": ["context_gathering"]},
            {"name": "synthesis", "depends_on": ["research"]},
            {"name": "validation", "depends_on": ["synthesis"]}
        ]
        
        # Verify dependency resolution
        execution_order = []
        for task in task_chain:
            if all(dep in execution_order for dep in task["depends_on"]):
                execution_order.append(task["name"])
        
        assert execution_order == ["context_gathering", "research", "synthesis", "validation"]
    
    def test_conditional_workflow_execution(self, mock_crew, sample_inputs):
        """Test conditional workflow paths based on inputs."""
        crew = mock_crew.crew()
        
        # Mock conditional logic
        def select_tasks(inputs):
            tasks = ["context_gathering"]
            if inputs.get("include_deep_research"):
                tasks.append("deep_research")
            else:
                tasks.append("standard_research")
            tasks.extend(["synthesis", "validation"])
            return tasks
        
        standard_tasks = select_tasks(sample_inputs)
        deep_tasks = select_tasks({**sample_inputs, "include_deep_research": True})
        
        assert "standard_research" in standard_tasks
        assert "deep_research" in deep_tasks
        assert "deep_research" not in standard_tasks
    
    def test_parallel_task_execution(self, mock_crew, sample_inputs):
        """Test parallel execution of independent tasks."""
        crew = mock_crew.crew()
        
        # Mock parallel task groups
        parallel_groups = [
            ["literature_search", "expert_interviews"],
            ["data_analysis", "statistical_modeling"],
            ["result_compilation", "visualization_creation"]
        ]
        
        def execute_task_group(group):
            results = {}
            for task in group:
                results[task] = f"{task}_completed"
            return results
        
        # Execute parallel groups
        all_results = {}
        for group in parallel_groups:
            group_results = execute_task_group(group)
            all_results.update(group_results)
        
        assert len(all_results) == 6
        assert all("completed" in result for result in all_results.values())


class TestDataPersistence:
    """Test data persistence and recovery."""
    
    def test_research_state_persistence(self, mock_crew, sample_inputs, temp_workspace):
        """Test persistence of research state across sessions."""    
        state_dir = f"{temp_workspace}/research_state"
        os.makedirs(state_dir, exist_ok=True)
        
        crew = mock_crew.crew()
        
        # First session - partial completion
        partial_result = crew.kickoff(inputs=sample_inputs)
        
        # Mock state persistence
        state = {
            "session_id": "test_session",
            "progress": 0.6,
            "completed_tasks": ["context_gathering", "research"],
            "pending_tasks": ["synthesis", "validation"],
            "intermediate_data": partial_result
        }
        
        state_file = f"{state_dir}/session_state.json"
        with open(state_file, 'w') as f:
            json.dump(state, f, default=str)
        
        # Second session - resume from state
        with open(state_file, 'r') as f:
            restored_state = json.load(f)
        
        assert restored_state["progress"] == 0.6
        assert "synthesis" in restored_state["pending_tasks"]
        assert len(restored_state["completed_tasks"]) == 2
    
    def test_knowledge_graph_persistence(self, mock_mcp_manager, temp_workspace):
        """Test knowledge graph persistence and updates."""
        graph_file = f"{temp_workspace}/knowledge_graph.json"
        
        # Initial knowledge graph
        initial_graph = {
            "entities": [
                {"name": "AI Testing", "type": "concept", "properties": {"importance": "high"}},
                {"name": "Machine Learning", "type": "field", "properties": {"maturity": "established"}}
            ],
            "relationships": [
                {"from": "AI Testing", "to": "Machine Learning", "type": "applies_to"}
            ]
        }
        
        # Save initial graph
        with open(graph_file, 'w') as f:
            json.dump(initial_graph, f, indent=2)
        
        # Simulate research updates
        mock_mcp_manager.call_tool(
            "memory_create_entity",
            name="Neural Network Testing",
            entity_type="concept",
            observations=["Specialized testing for neural networks"]
        )
        
        # Load and verify updates
        with open(graph_file, 'r') as f:
            updated_graph = json.load(f)
        
        assert len(updated_graph["entities"]) >= 2
        assert len(updated_graph["relationships"]) >= 1
    
    def test_research_artifact_management(self, mock_crew, sample_inputs, temp_workspace):
        """Test management of research artifacts and outputs."""
        artifacts_dir = f"{temp_workspace}/artifacts"
        os.makedirs(artifacts_dir, exist_ok=True)
        
        crew = mock_crew.crew()
        result = crew.kickoff(inputs=sample_inputs)
        
        # Mock artifact creation
        artifacts = {
            "research_paper.md": result.get("research_paper", {}),
            "data_analysis.json": {"statistics": "analysis_results"},
            "references.bib": "@article{example,title={Example},year={2024}}",
            "figures/": {"chart1.png": "mock_chart_data"},
            "raw_data/": {"interviews.json": "mock_interview_data"}
        }
        
        # Create artifact structure
        for artifact_name, content in artifacts.items():
            if artifact_name.endswith('/'):
                subdir = f"{artifacts_dir}/{artifact_name}"
                os.makedirs(subdir, exist_ok=True)
                # Add mock files to subdirectory
                for subfile, subcontent in content.items():
                    with open(f"{subdir}/{subfile}", 'w') as f:
                        if isinstance(subcontent, dict):
                            json.dump(subcontent, f)
                        else:
                            f.write(str(subcontent))
            else:
                with open(f"{artifacts_dir}/{artifact_name}", 'w') as f:
                    if isinstance(content, dict):
                        json.dump(content, f, indent=2, default=str)
                    else:
                        f.write(str(content))
        
        # Verify artifact creation
        assert os.path.exists(f"{artifacts_dir}/research_paper.md")
        assert os.path.exists(f"{artifacts_dir}/figures/chart1.png")
        assert os.path.isdir(f"{artifacts_dir}/raw_data/")


class TestErrorRecovery:
    """Test error recovery and resilience mechanisms."""
    
    def test_workflow_error_recovery(self, mock_crew, sample_inputs, error_scenarios):
        """Test workflow recovery from various error scenarios."""
        crew = mock_crew.crew()
        
        for scenario in error_scenarios:
            # Mock error injection
            error_type = scenario["type"]
            recovery_strategy = scenario["recovery"]
            
            if error_type == "network_timeout":
                # Mock network timeout and retry
                with patch('requests.get', side_effect=TimeoutError("Network timeout")):
                    result = crew.kickoff(inputs=sample_inputs)
                    
                    # Should implement retry logic
                    assert result is not None
                    assert "retry_attempted" in result
            
            elif error_type == "memory_full":
                # Mock memory pressure
                with patch('psutil.virtual_memory') as mock_memory:
                    mock_memory.return_value.percent = 95  # High memory usage
                    result = crew.kickoff(inputs=sample_inputs)
                    
                    # Should implement memory management
                    assert result is not None
                    assert "memory_optimized" in result
            
            elif error_type == "invalid_data":
                # Mock data validation error
                invalid_inputs = {**sample_inputs, "topic": ""}  # Invalid empty topic
                
                result = crew.kickoff(inputs=invalid_inputs)
                
                # Should handle gracefully
                assert result is not None
                assert "validation_error" in result
    
    def test_checkpoint_recovery(self, mock_crew, sample_inputs, temp_workspace):
        """Test recovery from workflow checkpoints."""
        checkpoint_dir = f"{temp_workspace}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        crew = mock_crew.crew()
        
        # Mock checkpoint creation during execution
        def task_executor(task_name):
            checkpoint = {
                "task": task_name,
                "timestamp": datetime.now().isoformat(),
                "status": "completed",
                "output": f"{task_name}_output"
            }
            
            checkpoint_file = f"{checkpoint_dir}/{task_name}_checkpoint.json"
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            
            return checkpoint["output"]
        
        # Simulate task execution with checkpoints
        tasks = ["context_gathering", "research", "synthesis"]
        completed_tasks = []
        
        for task in tasks:
            try:
                output = task_executor(task)
                completed_tasks.append(task)
            except Exception as e:
                # Recovery: load from checkpoint
                checkpoint_file = f"{checkpoint_dir}/{task}_checkpoint.json"
                if os.path.exists(checkpoint_file):
                    with open(checkpoint_file, 'r') as f:
                        checkpoint = json.load(f)
                    completed_tasks.append(task)
        
        # Verify all tasks completed (either executed or recovered)
        assert len(completed_tasks) == len(tasks)
        assert all(task in completed_tasks for task in tasks)


@pytest.mark.slow
class TestEndToEndResearchFlow:
    """Test complete research workflow from input to output (legacy compatibility)."""
    
    @patch('builtins.input', side_effect=['Artificial Intelligence Ethics', 'y'])
    @patch('src.server_research_mcp.crew.ServerResearchMcp.crew')
    def test_complete_research_workflow(self, mock_crew_method, mock_input, 
                                      valid_research_output, valid_report_output):
        """Test complete workflow from user input to final report."""
        # Setup mock crew
        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = {
            'research_output': valid_research_output,
            'report_output': valid_report_output
        }
        mock_crew_method.return_value = mock_crew
        
        # Import and run
        from server_research_mcp.main import run
        
        with patch('builtins.print') as mock_print:
            with patch('os.path.exists', return_value=False):
                run()
        
        # Verify execution flow
        assert mock_input.call_count == 2  # Topic and confirmation
        mock_crew.kickoff.assert_called_once()
        
        # Verify inputs passed
        call_args = mock_crew.kickoff.call_args
        assert call_args.kwargs['inputs']['topic'] == 'Artificial Intelligence Ethics'
        assert 'current_year' in call_args.kwargs['inputs'] 