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
    
    def test_fact_checking_pipeline(self, mock_crew, mock_mcp_manager):
        """Test fact-checking pipeline for research outputs."""
        crew = mock_crew.crew()
        
        # Research with claims
        result = crew.kickoff(inputs={
            "topic": "AI Safety Measures",
            "current_year": "2024",
            "require_fact_checking": True
        })
        
        # Extract claims from research
        claims = [
            "GPT-4 was released in March 2023",
            "Transformer architecture was introduced in 2017",
            "RLHF improves model alignment"
        ]
        
        # Fact check each claim
        fact_check_results = []
        for claim in claims:
            # Search for supporting evidence
            evidence = mock_mcp_manager.call_tool(
                "memory_search",
                query=claim
            )
            
            fact_check_results.append({
                "claim": claim,
                "evidence_found": len(evidence["results"]) > 0,
                "confidence": 0.85 if evidence["results"] else 0.3
            })
        
        # Verify fact checking
        verified_claims = [r for r in fact_check_results if r["confidence"] > 0.8]
        assert len(verified_claims) >= 2
    
    def test_citation_integrity(self, mock_crew, sample_inputs):
        """Test citation integrity and traceability."""
        crew = mock_crew.crew()
        result = crew.kickoff(inputs=sample_inputs)
        
        paper = result["research_paper"]
        
        # Extract all citations from content
        citations_in_text = []
        for section in paper["sections"]:
            # Simple citation extraction (would be more sophisticated in practice)
            import re
            matches = re.findall(r'\[(\d+)\]', section["content"])
            citations_in_text.extend(matches)
        
        # Verify citation integrity
        citation_checks = {
            "all_citations_have_references": all(
                int(c) <= len(paper.get("references", [])) 
                for c in citations_in_text if c.isdigit()
            ),
            "all_references_are_cited": len(set(citations_in_text)) >= len(paper.get("references", [])) * 0.8,
            "no_duplicate_references": len(paper.get("references", [])) == len(set(paper.get("references", [])))
        }
        
        assert all(citation_checks.values()), f"Citation checks failed: {citation_checks}"


class TestWorkflowOrchestration:
    """Test workflow orchestration and task dependencies."""
    
    def test_task_dependency_management(self, mock_crew, mock_crew_tasks):
        """Test proper task dependency execution."""
        crew = mock_crew.crew()
        
        # Define task dependencies
        task_graph = {
            "context_gathering": [],  # No dependencies
            "deep_research": ["context_gathering"],
            "synthesis": ["context_gathering", "deep_research"],
            "validation": ["synthesis"]
        }
        
        # Track execution order
        execution_order = []
        
        for task_name, task in mock_crew_tasks.items():
            task.execute = MagicMock(side_effect=lambda tn=task_name: execution_order.append(tn))
        
        # Execute workflow
        crew.kickoff(inputs={"topic": "Test", "current_year": "2024"})
        
        # Verify dependency order
        for task, deps in task_graph.items():
            if task in execution_order:
                task_idx = execution_order.index(task)
                for dep in deps:
                    if dep in execution_order:
                        dep_idx = execution_order.index(dep)
                        assert dep_idx < task_idx, f"{dep} should execute before {task}"
    
    def test_conditional_workflow_execution(self, mock_crew, sample_inputs):
        """Test conditional workflow paths based on intermediate results."""
        crew = mock_crew.crew()
        
        # Add conditional logic
        sample_inputs["research_depth"] = "exploratory"  # vs "comprehensive"
        
        # Mock conditional task selection
        def select_tasks(inputs):
            if inputs.get("research_depth") == "exploratory":
                return ["context_gathering", "synthesis"]  # Skip deep research
            else:
                return ["context_gathering", "deep_research", "synthesis", "validation"]
        
        crew.select_tasks = MagicMock(side_effect=select_tasks)
        selected_tasks = crew.select_tasks(sample_inputs)
        
        # Verify conditional execution
        assert len(selected_tasks) == 2
        assert "deep_research" not in selected_tasks
    
    def test_parallel_task_execution(self, mock_crew, sample_inputs):
        """Test parallel execution of independent tasks."""
        crew = mock_crew.crew()
        
        # Define parallel task groups
        parallel_groups = [
            ["web_search", "academic_search"],  # Can run in parallel
            ["initial_synthesis"],  # Depends on searches
            ["peer_review_1", "peer_review_2", "peer_review_3"],  # Parallel reviews
            ["final_synthesis"]  # Depends on reviews
        ]
        
        # Mock parallel execution tracking
        import concurrent.futures
        execution_times = {}
        
        def execute_task_group(group):
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(group)) as executor:
                start_time = datetime.now()
                futures = {executor.submit(lambda t=task: t): task for task in group}
                results = [f.result() for f in concurrent.futures.as_completed(futures)]
                end_time = datetime.now()
                return results, (end_time - start_time).total_seconds()
        
        # Execute groups
        for group in parallel_groups:
            results, duration = execute_task_group(group)
            execution_times[str(group)] = duration
            assert len(results) == len(group)
        
        # Verify parallel execution (groups with multiple tasks should complete faster)
        assert execution_times[str(parallel_groups[0])] < len(parallel_groups[0]) * 0.5
        assert execution_times[str(parallel_groups[2])] < len(parallel_groups[2]) * 0.5


class TestDataPersistence:
    """Test data persistence and recovery mechanisms."""
    
    def test_research_state_persistence(self, mock_crew, sample_inputs, temp_workspace):
        """Test saving and loading research state."""
        crew = mock_crew.crew()
        state_file = f"{temp_workspace}/research_state.json"
        
        # Execute partial research
        crew.enable_checkpoints = True
        result = crew.kickoff(inputs=sample_inputs)
        
        # Save state
        state = {
            "timestamp": datetime.now().isoformat(),
            "inputs": sample_inputs,
            "completed_tasks": ["context_gathering", "deep_research"],
            "pending_tasks": ["synthesis", "validation"],
            "intermediate_results": {
                "context": "Gathered context...",
                "research_findings": ["Finding 1", "Finding 2"]
            }
        }
        
        with open(state_file, 'w') as f:
            json.dump(state, f)
        
        # Load and verify state
        with open(state_file, 'r') as f:
            loaded_state = json.load(f)
        
        assert loaded_state["inputs"] == sample_inputs
        assert len(loaded_state["completed_tasks"]) == 2
        assert len(loaded_state["pending_tasks"]) == 2
    
    def test_knowledge_graph_persistence(self, mock_mcp_manager, temp_workspace):
        """Test knowledge graph persistence across sessions."""
        kg_file = f"{temp_workspace}/knowledge_graph.json"
        
        # Build knowledge graph
        entities = []
        for i in range(5):
            result = mock_mcp_manager.call_tool(
                "memory_create_entity",
                name=f"concept_{i}",
                entity_type="research_concept",
                observations=[f"Observation about concept {i}"]
            )
            entities.append({
                "id": result["entity_id"],
                "name": f"concept_{i}",
                "type": "research_concept"
            })
        
        # Add relationships
        relationships = []
        for i in range(len(entities) - 1):
            relationships.append({
                "source": entities[i]["id"],
                "target": entities[i + 1]["id"],
                "type": "related_to"
            })
        
        # Save knowledge graph
        kg_data = {
            "entities": entities,
            "relationships": relationships,
            "metadata": {
                "created": datetime.now().isoformat(),
                "version": "1.0"
            }
        }
        
        with open(kg_file, 'w') as f:
            json.dump(kg_data, f)
        
        # Load and verify
        with open(kg_file, 'r') as f:
            loaded_kg = json.load(f)
        
        assert len(loaded_kg["entities"]) == 5
        assert len(loaded_kg["relationships"]) == 4
    
    def test_research_artifact_management(self, mock_crew, sample_inputs, temp_workspace):
        """Test management of research artifacts and outputs."""
        crew = mock_crew.crew()
        artifacts_dir = f"{temp_workspace}/artifacts"
        os.makedirs(artifacts_dir, exist_ok=True)
        
        # Execute research
        result = crew.kickoff(inputs=sample_inputs)
        
        # Save various artifacts
        artifacts = {
            "paper.md": result["research_paper"],
            "context.md": result["context_foundation"],
            "metadata.json": {
                "topic": sample_inputs["topic"],
                "timestamp": datetime.now().isoformat(),
                "version": "1.0",
                "quality_score": 0.92
            },
            "visualizations/knowledge_graph.json": {
                "nodes": [],
                "edges": []
            }
        }
        
        # Save artifacts
        for path, content in artifacts.items():
            full_path = os.path.join(artifacts_dir, path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            with open(full_path, 'w') as f:
                if isinstance(content, dict):
                    json.dump(content, f)
                else:
                    f.write(str(content))
        
        # Verify artifacts
        assert os.path.exists(f"{artifacts_dir}/paper.md")
        assert os.path.exists(f"{artifacts_dir}/metadata.json")
        assert os.path.exists(f"{artifacts_dir}/visualizations/knowledge_graph.json")


class TestErrorRecovery:
    """Test error recovery in end-to-end workflows."""
    
    def test_workflow_error_recovery(self, mock_crew, sample_inputs, error_scenarios):
        """Test recovery from various workflow errors."""
        crew = mock_crew.crew()
        recovery_attempts = []
        
        for error_type, error in error_scenarios.items():
            # Inject error at different stages
            if error_type == "llm_rate_limit":
                crew.agents[1].execute = MagicMock(side_effect=error)
            elif error_type == "mcp_connection_error":
                crew.agents[0].tools[0]._run = MagicMock(side_effect=error)
            
            # Attempt recovery
            try:
                result = crew.kickoff(inputs=sample_inputs)
                recovery_attempts.append({
                    "error_type": error_type,
                    "recovered": True,
                    "result": result
                })
            except Exception as e:
                # Implement recovery strategy
                if error_type == "llm_rate_limit":
                    # Wait and retry
                    import time
                    time.sleep(0.1)  # Simulated wait
                    crew.agents[1].execute = MagicMock(return_value="Recovered result")
                    result = crew.kickoff(inputs=sample_inputs)
                    recovery_attempts.append({
                        "error_type": error_type,
                        "recovered": True,
                        "result": result
                    })
                else:
                    recovery_attempts.append({
                        "error_type": error_type,
                        "recovered": False,
                        "error": str(e)
                    })
        
        # Verify recovery success rate
        recovered = [r for r in recovery_attempts if r["recovered"]]
        assert len(recovered) >= len(error_scenarios) * 0.6  # 60% recovery rate
    
    def test_checkpoint_recovery(self, mock_crew, sample_inputs, temp_workspace):
        """Test recovery from checkpoints after failure."""
        crew = mock_crew.crew()
        checkpoint_dir = f"{temp_workspace}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Enable checkpointing
        crew.checkpoint_dir = checkpoint_dir
        crew.checkpoint_interval = 1  # After each task
        
        # Simulate failure after second task
        executed_tasks = []
        
        def task_executor(task_name):
            executed_tasks.append(task_name)
            if len(executed_tasks) == 3:  # Fail on third task
                raise Exception("Simulated task failure")
            return f"{task_name} completed"
        
        # Mock task execution with failure
        for i, task in enumerate(crew.tasks):
            task.execute = MagicMock(side_effect=lambda tn=f"task_{i}": task_executor(tn))
        
        # First attempt - should fail
        with pytest.raises(Exception):
            crew.kickoff(inputs=sample_inputs)
        
        # Verify checkpoint was created
        assert len(executed_tasks) == 3
        
        # Reset and recover from checkpoint
        executed_tasks.clear()
        crew.restore_from_checkpoint = MagicMock(return_value={
            "completed_tasks": ["task_0", "task_1"],
            "pending_tasks": ["task_2", "task_3"]
        })
        
        # Fix the failing task
        crew.tasks[2].execute = MagicMock(return_value="task_2 completed")
        
        # Resume from checkpoint
        result = crew.kickoff(inputs=sample_inputs, resume_from_checkpoint=True)
        
        # Verify successful completion
        assert result is not None
        assert crew.restore_from_checkpoint.called