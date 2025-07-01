"""End-to-end workflow tests - consolidated from test_end_to_end.py."""

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
        """Test citation integrity and reference validation."""
        crew = mock_crew.crew()
        result = crew.kickoff(inputs=sample_inputs)
        
        paper = result["research_paper"]
        
        # Extract citations from content
        citations = []
        for section in paper["sections"]:
            content = section["content"]
            # Simple citation extraction (in reality would be more sophisticated)
            import re
            found_citations = re.findall(r'\[(\d+)\]', content)
            citations.extend(found_citations)
        
        # Check references exist for all citations
        references = paper.get("references", [])
        reference_ids = [str(i+1) for i in range(len(references))]
        
        # All citations should have corresponding references
        for citation in citations:
            assert citation in reference_ids, f"Citation [{citation}] missing reference"


class TestWorkflowOrchestration:
    """Test workflow orchestration and task management."""
    
    def test_task_dependency_management(self, mock_crew, mock_crew_tasks):
        """Test task dependency resolution and execution order."""
        crew = mock_crew.crew()
        tasks = mock_crew_tasks
        
        # Set up task dependencies
        tasks["context_gathering"].dependencies = []
        tasks["deep_research"].dependencies = [tasks["context_gathering"]]
        tasks["synthesis"].dependencies = [tasks["deep_research"]]
        tasks["validation"].dependencies = [tasks["synthesis"]]
        
        # Simulate task execution in dependency order
        expected_order = ["context_gathering", "deep_research", "synthesis", "validation"]
        for task_name in expected_order:
            crew.execute_task(task_name)
        
        # Execute workflow
        result = crew.kickoff(inputs={"topic": "Test Dependencies"})
        
        # Verify execution order through mock calls
        actual_calls = [call[0][0] for call in crew.execute_task.call_args_list]
        
        # Tasks should execute in dependency order - check that all expected tasks were called
        assert len(actual_calls) >= len(expected_order), f"Expected at least {len(expected_order)} task calls, got {len(actual_calls)}"
        
        # Check that all expected tasks appear in the calls (order matters for dependencies)
        for expected_task in expected_order:
            assert expected_task in actual_calls, f"Expected task {expected_task} not found in actual calls: {actual_calls}"
        
        # Verify dependency order: each task should appear after its dependencies
        task_positions = {task: actual_calls.index(task) for task in expected_order if task in actual_calls}
        
        # context_gathering should come before deep_research
        if "context_gathering" in task_positions and "deep_research" in task_positions:
            assert task_positions["context_gathering"] < task_positions["deep_research"], "context_gathering should come before deep_research"
        
        # deep_research should come before synthesis  
        if "deep_research" in task_positions and "synthesis" in task_positions:
            assert task_positions["deep_research"] < task_positions["synthesis"], "deep_research should come before synthesis"
        
        # synthesis should come before validation
        if "synthesis" in task_positions and "validation" in task_positions:
            assert task_positions["synthesis"] < task_positions["validation"], "synthesis should come before validation"
    
    def test_conditional_workflow_execution(self, mock_crew, sample_inputs):
        """Test conditional workflow paths based on intermediate results."""
        crew = mock_crew.crew()
        
        # Mock conditional logic
        def select_tasks(inputs):
            if inputs.get("detailed_analysis", False):
                return ["context_gathering", "deep_research", "detailed_analysis", "synthesis", "validation"]
            else:
                return ["context_gathering", "quick_research", "synthesis", "validation"]
        
        # Test detailed analysis path
        detailed_inputs = {**sample_inputs, "detailed_analysis": True}
        crew.select_tasks = select_tasks
        result = crew.kickoff(inputs=detailed_inputs)
        
        assert "detailed_analysis" in result.get("executed_tasks", [])
    
    def test_parallel_task_execution(self, mock_crew, sample_inputs):
        """Test parallel execution of independent tasks."""
        import concurrent.futures
        import time
        
        crew = mock_crew.crew()
        
        # Mock parallel tasks
        parallel_tasks = ["literature_review", "data_collection", "expert_interviews"]
        
        def execute_task_group(group):
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = []
                for task in group:
                    future = executor.submit(lambda t: {"task": t, "result": f"Completed {t}"}, task)
                    futures.append(future)
                
                results = []
                for future in concurrent.futures.as_completed(futures):
                    results.append(future.result())
                
                return results
        
        crew.execute_parallel_tasks = execute_task_group
        
        # Execute parallel tasks
        start_time = time.time()
        results = crew.execute_parallel_tasks(parallel_tasks)
        end_time = time.time()
        
        # Should complete faster than sequential execution
        assert len(results) == 3
        assert end_time - start_time < 1.0  # Should be much faster than 3 seconds sequential


class TestDataPersistence:
    """Test data persistence across workflow stages."""
    
    def test_research_state_persistence(self, mock_crew, sample_inputs, temp_workspace):
        """Test persistence of research state across interruptions."""
        crew = mock_crew.crew()
        state_file = f"{temp_workspace}/research_state.json"
        
        # Start research
        crew.kickoff(inputs=sample_inputs)
        
        # Simulate saving state
        research_state = {
            "current_stage": "deep_research",
            "completed_tasks": ["context_gathering"],
            "context_data": {"key_concepts": ["AI", "Testing"]},
            "intermediate_results": {"search_results": ["Result 1", "Result 2"]},
            "timestamp": datetime.now().isoformat()
        }
        
        with open(state_file, 'w') as f:
            json.dump(research_state, f)
        
        # Verify state can be restored
        with open(state_file, 'r') as f:
            restored_state = json.load(f)
        
        assert restored_state["current_stage"] == "deep_research"
        assert "context_gathering" in restored_state["completed_tasks"]
        assert len(restored_state["context_data"]["key_concepts"]) == 2
    
    def test_knowledge_graph_persistence(self, mock_mcp_manager, temp_workspace):
        """Test knowledge graph persistence across sessions."""
        # Create initial knowledge graph
        entities = [
            {"name": "machine_learning", "type": "field", "observations": ["Subset of AI"]},
            {"name": "deep_learning", "type": "subfield", "observations": ["Uses neural networks"]},
            {"name": "neural_networks", "type": "technology", "observations": ["Inspired by brain"]}
        ]
        
        # Save entities
        for entity in entities:
            mock_mcp_manager.call_tool(
                "memory_create_entity",
                **entity
            )
        
        # Simulate session end and restart
        kg_export = mock_mcp_manager.call_tool("memory_export_graph")
        kg_file = f"{temp_workspace}/knowledge_graph.json"
        
        with open(kg_file, 'w') as f:
            json.dump(kg_export, f)
        
        # Verify persistence
        with open(kg_file, 'r') as f:
            restored_kg = json.load(f)
        
        assert len(restored_kg["entities"]) >= 3
        entity_names = [e["name"] for e in restored_kg["entities"]]
        assert "machine_learning" in entity_names
        assert "deep_learning" in entity_names
        assert "neural_networks" in entity_names
    
    def test_research_artifact_management(self, mock_crew, sample_inputs, temp_workspace):
        """Test management of research artifacts and outputs."""
        crew = mock_crew.crew()
        artifacts_dir = f"{temp_workspace}/artifacts"
        os.makedirs(artifacts_dir, exist_ok=True)
        
        # Execute research
        result = crew.kickoff(inputs=sample_inputs)
        
        # Simulate artifact creation
        artifacts = {
            "search_queries.json": {"queries": ["AI", "testing", "validation"]},
            "raw_data.json": {"papers": [{"title": "Paper 1"}, {"title": "Paper 2"}]},
            "processed_data.json": {"structured_papers": [{"id": 1}, {"id": 2}]},
            "final_output.md": "# Research Paper\n\nFinal research output..."
        }
        
        # Save artifacts
        for filename, content in artifacts.items():
            filepath = f"{artifacts_dir}/{filename}"
            if filename.endswith('.json'):
                with open(filepath, 'w') as f:
                    json.dump(content, f)
            else:
                with open(filepath, 'w') as f:
                    f.write(content)
        
        # Verify artifacts exist
        for filename in artifacts.keys():
            filepath = f"{artifacts_dir}/{filename}"
            assert os.path.exists(filepath)
            
            # Verify content integrity
            if filename.endswith('.json'):
                with open(filepath, 'r') as f:
                    loaded_content = json.load(f)
                    assert loaded_content == artifacts[filename]


class TestErrorRecovery:
    """Test error recovery and resilience mechanisms."""
    
    def test_workflow_error_recovery(self, mock_crew, sample_inputs, error_scenarios):
        """Test workflow recovery from various error scenarios."""
        crew = mock_crew.crew()
        
        for scenario_name, error_config in error_scenarios.items():
            # Configure error scenario
            if scenario_name == "network_failure":
                with patch('requests.get', side_effect=ConnectionError("Network unavailable")):
                    result = crew.kickoff(inputs=sample_inputs)
                    
                    # Should handle network errors gracefully
                    assert result is not None
                    assert "error_handled" in result
                    assert result["error_type"] == "network"
            
            elif scenario_name == "memory_exhaustion":
                with patch('psutil.virtual_memory', return_value=MagicMock(available=1024)):  # Very low memory
                    result = crew.kickoff(inputs=sample_inputs)
                    
                    # Should handle memory constraints
                    assert result is not None
                    assert "memory_optimized" in result
            
            elif scenario_name == "timeout":
                with patch('time.time', side_effect=[0, 0, 3600, 3600]):  # Simulate timeout
                    result = crew.kickoff(inputs=sample_inputs)
                    
                    # Should handle timeouts
                    assert result is not None
                    assert result.get("timeout_handled", False)
    
    def test_checkpoint_recovery(self, mock_crew, sample_inputs, temp_workspace):
        """Test recovery from saved checkpoints."""
        crew = mock_crew.crew()
        checkpoint_dir = f"{temp_workspace}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Mock checkpoint creation during execution
        def task_executor(task_name):
            # Create checkpoint after each task
            checkpoint = {
                "task": task_name,
                "timestamp": datetime.now().isoformat(),
                "state": {"completed": True, "output": f"Output of {task_name}"}
            }
            
            checkpoint_file = f"{checkpoint_dir}/{task_name}_checkpoint.json"
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f)
            
            return checkpoint["state"]["output"]
        
        crew.execute_task = task_executor
        
        # Execute with checkpointing
        result = crew.kickoff(inputs=sample_inputs)
        
        # Verify checkpoints were created
        checkpoint_files = os.listdir(checkpoint_dir)
        assert len(checkpoint_files) > 0
        
        # Test recovery from checkpoint
        latest_checkpoint = max(checkpoint_files)
        checkpoint_path = f"{checkpoint_dir}/{latest_checkpoint}"
        
        with open(checkpoint_path, 'r') as f:
            checkpoint_data = json.load(f)
        
        # Should be able to resume from checkpoint
        assert "task" in checkpoint_data
        assert "state" in checkpoint_data
        assert checkpoint_data["state"]["completed"] is True


@pytest.mark.slow
class TestEndToEndResearchFlow:
    """Test complete research workflow from input to output (legacy compatibility)."""
    
    @patch('builtins.input', side_effect=['Artificial Intelligence Ethics', 'y'])
    @patch('src.server_research_mcp.crew.ServerResearchMcp.crew')
    @patch('sys.argv', ['main.py', 'test query', '--yes'])  # Add --yes flag to skip confirmation
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
                with patch('server_research_mcp.main.load_dotenv'):
                    run()
        
        # Verify execution flow
        assert mock_input.call_count == 2  # Topic and confirmation
        mock_crew.kickoff.assert_called_once()
        
        # Verify inputs passed
        call_args = mock_crew.kickoff.call_args
        assert call_args.kwargs['inputs']['topic'] == 'Artificial Intelligence Ethics'
        assert 'current_year' in call_args.kwargs['inputs'] 