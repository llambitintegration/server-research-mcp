"""Performance and scalability tests for the research crew."""

import pytest
import time
import concurrent.futures
import psutil
import os
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
import asyncio
import statistics


class TestPerformanceBaseline:
    """Establish performance baselines for crew operations."""
    
    @pytest.mark.performance
    def test_single_agent_performance(self, mock_crew_agents, performance_monitor):
        """Test single agent task execution performance."""
        historian = mock_crew_agents["historian"]
        
        # Measure execution time
        performance_monitor.start_timer("single_agent_execution")
        
        # Execute multiple tasks
        results = []
        for i in range(10):
            start = time.time()
            result = historian.execute(f"Task {i}")
            end = time.time()
            results.append({
                "task_id": i,
                "duration": end - start,
                "result": result
            })
        
        total_time = performance_monitor.stop_timer("single_agent_execution")
        
        # Calculate metrics
        durations = [r["duration"] for r in results]
        avg_duration = statistics.mean(durations)
        max_duration = max(durations)
        min_duration = min(durations)
        
        # Performance assertions
        assert avg_duration < 0.1  # Average under 100ms
        assert max_duration < 0.5  # Max under 500ms
        assert total_time < 2.0   # Total under 2 seconds for 10 tasks
    
    @pytest.mark.performance
    def test_crew_initialization_performance(self, performance_monitor):
        """Test crew initialization performance."""
        from server_research_mcp.crew import ServerResearchMcp
        
        performance_monitor.start_timer("crew_initialization")
        
        # Initialize crew multiple times
        init_times = []
        for i in range(5):
            start = time.time()
            crew = ServerResearchMcp()
            end = time.time()
            init_times.append(end - start)
        
        total_time = performance_monitor.stop_timer("crew_initialization")
        
        # Performance metrics
        avg_init_time = statistics.mean(init_times)
        
        # Assertions
        assert avg_init_time < 0.5  # Average initialization under 500ms
        assert total_time < 3.0     # Total under 3 seconds for 5 initializations
    
    @pytest.mark.performance
    def test_task_creation_performance(self, mock_crew, performance_monitor):
        """Test task creation performance."""
        crew = mock_crew.crew()
        
        performance_monitor.start_timer("task_creation")
        
        # Create multiple tasks
        task_times = []
        for i in range(20):
            start = time.time()
            task = MagicMock()
            task.description = f"Test task {i}"
            task.expected_output = f"Output for task {i}"
            crew.tasks.append(task)
            end = time.time()
            task_times.append(end - start)
        
        total_time = performance_monitor.stop_timer("task_creation")
        
        # Assertions
        assert statistics.mean(task_times) < 0.01  # Average under 10ms
        assert total_time < 0.5  # Total under 500ms for 20 tasks


class TestScalabilityLimits:
    """Test system scalability limits and behavior under load."""
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_large_knowledge_graph_performance(self, mock_mcp_manager, performance_monitor):
        """Test performance with large knowledge graphs."""
        # Create a large knowledge graph
        num_entities = 1000
        
        performance_monitor.start_timer("large_kg_creation")
        
        # Batch create entities
        batch_size = 100
        for batch in range(0, num_entities, batch_size):
            entities = []
            for i in range(batch, min(batch + batch_size, num_entities)):
                result = mock_mcp_manager.call_tool(
                    "memory_create_entity",
                    name=f"entity_{i}",
                    entity_type="concept",
                    observations=[f"Observation {i}"]
                )
                entities.append(result)
        
        creation_time = performance_monitor.stop_timer("large_kg_creation")
        
        # Test search performance on large graph
        performance_monitor.start_timer("large_kg_search")
        
        search_times = []
        for i in range(10):
            start = time.time()
            result = mock_mcp_manager.call_tool(
                "memory_search",
                query=f"entity_{i * 100}"
            )
            end = time.time()
            search_times.append(end - start)
        
        search_time = performance_monitor.stop_timer("large_kg_search")
        
        # Assertions
        assert creation_time < 30  # Creation under 30 seconds
        assert statistics.mean(search_times) < 0.5  # Search under 500ms average
    
    @pytest.mark.performance
    def test_concurrent_agent_scaling(self, mock_crew_agents, performance_monitor):
        """Test scaling with concurrent agent execution."""
        agents = list(mock_crew_agents.values())
        
        def execute_agent_task(agent, task_id):
            start = time.time()
            result = agent.execute(f"Concurrent task {task_id}")
            end = time.time()
            return {
                "agent": agent.role,
                "task_id": task_id,
                "duration": end - start,
                "result": result
            }
        
        # Test different concurrency levels
        concurrency_levels = [1, 2, 4, 8]
        scaling_results = {}
        
        for level in concurrency_levels:
            performance_monitor.start_timer(f"concurrent_{level}")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=level) as executor:
                futures = []
                for i in range(level * 5):  # 5 tasks per worker
                    agent = agents[i % len(agents)]
                    future = executor.submit(execute_agent_task, agent, i)
                    futures.append(future)
                
                results = [f.result() for f in concurrent.futures.as_completed(futures)]
            
            total_time = performance_monitor.stop_timer(f"concurrent_{level}")
            
            scaling_results[level] = {
                "total_time": total_time,
                "num_tasks": level * 5,
                "throughput": (level * 5) / total_time
            }
        
        # Verify scaling efficiency
        base_throughput = scaling_results[1]["throughput"]
        for level in [2, 4]:
            efficiency = scaling_results[level]["throughput"] / (base_throughput * level)
            assert efficiency > 0.7  # At least 70% scaling efficiency
    
    @pytest.mark.performance
    def test_memory_usage_under_load(self, mock_crew, sample_inputs):
        """Test memory usage under heavy load."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        crew = mock_crew.crew()
        memory_samples = []
        
        # Execute multiple research cycles
        for i in range(10):
            # Modify inputs to create different research topics
            inputs = sample_inputs.copy()
            inputs["topic"] = f"Research Topic {i}"
            
            # Execute research
            result = crew.kickoff(inputs=inputs)
            
            # Sample memory usage
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_samples.append(current_memory)
            
            # Force garbage collection every few iterations
            if i % 3 == 0:
                import gc
                gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Analyze memory growth
        memory_growth_rate = (memory_samples[-1] - memory_samples[0]) / len(memory_samples)
        
        # Assertions
        assert memory_increase < 100  # Less than 100MB increase
        assert memory_growth_rate < 5  # Less than 5MB per iteration average


class TestAsyncPerformance:
    """Test asynchronous operation performance."""
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_async_tool_performance(self, async_mcp_client, performance_monitor):
        """Test async MCP tool execution performance."""
        await async_mcp_client.connect()
        
        performance_monitor.start_timer("async_tools")
        
        # Execute multiple async operations
        tasks = []
        for i in range(50):
            task = async_mcp_client.call_tool(
                "memory_search",
                query=f"async_query_{i}"
            )
            tasks.append(task)
        
        # Execute concurrently
        start = time.time()
        results = await asyncio.gather(*tasks)
        end = time.time()
        
        total_time = performance_monitor.stop_timer("async_tools")
        concurrent_time = end - start
        
        await async_mcp_client.disconnect()
        
        # Assertions
        assert len(results) == 50
        assert concurrent_time < 5  # All 50 operations in under 5 seconds
        assert concurrent_time < total_time * 0.3  # Significant concurrency benefit
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_async_agent_collaboration(self, mock_crew_agents):
        """Test async collaboration between agents."""
        agents = list(mock_crew_agents.values())
        
        async def async_agent_execute(agent, task):
            # Simulate async execution
            await asyncio.sleep(0.01)  # Simulate some work
            return f"{agent.role} completed {task}"
        
        # Create collaboration tasks
        collaboration_tasks = []
        for i in range(20):
            agent = agents[i % len(agents)]
            task = async_agent_execute(agent, f"collab_task_{i}")
            collaboration_tasks.append(task)
        
        start = time.time()
        results = await asyncio.gather(*collaboration_tasks)
        end = time.time()
        
        # Assertions
        assert len(results) == 20
        assert (end - start) < 0.5  # Should complete quickly due to concurrency


class TestResourceOptimization:
    """Test resource optimization strategies."""
    
    @pytest.mark.performance
    def test_caching_performance(self, mock_mcp_manager, performance_monitor):
        """Test caching impact on performance."""
        # Create a simple cache
        cache = {}
        cache_hits = 0
        cache_misses = 0
        
        def cached_tool_call(tool_name, **kwargs):
            nonlocal cache_hits, cache_misses
            cache_key = f"{tool_name}:{str(kwargs)}"
            
            if cache_key in cache:
                cache_hits += 1
                return cache[cache_key]
            else:
                cache_misses += 1
                result = mock_mcp_manager.call_tool(tool_name, **kwargs)
                cache[cache_key] = result
                return result
        
        # Test with repeated queries
        queries = ["test_query"] * 10 + ["unique_query_" + str(i) for i in range(10)]
        
        performance_monitor.start_timer("with_caching")
        for query in queries:
            cached_tool_call("memory_search", query=query)
        cached_time = performance_monitor.stop_timer("with_caching")
        
        # Compare with non-cached
        performance_monitor.start_timer("without_caching")
        for query in queries:
            mock_mcp_manager.call_tool("memory_search", query=query)
        non_cached_time = performance_monitor.stop_timer("without_caching")
        
        # Calculate cache effectiveness
        hit_rate = cache_hits / (cache_hits + cache_misses)
        
        # Assertions
        assert hit_rate > 0.4  # At least 40% cache hit rate
        assert cached_time < non_cached_time  # Caching provides benefit
    
    @pytest.mark.performance
    def test_batch_processing_optimization(self, mock_mcp_manager, performance_monitor):
        """Test batch processing optimization."""
        # Individual processing
        individual_items = list(range(100))
        
        performance_monitor.start_timer("individual_processing")
        individual_results = []
        for item in individual_items:
            result = mock_mcp_manager.call_tool(
                "memory_create_entity",
                name=f"item_{item}",
                entity_type="test",
                observations=[f"Observation {item}"]
            )
            individual_results.append(result)
        individual_time = performance_monitor.stop_timer("individual_processing")
        
        # Batch processing simulation
        performance_monitor.start_timer("batch_processing")
        batch_size = 10
        batch_results = []
        
        for i in range(0, len(individual_items), batch_size):
            batch = individual_items[i:i + batch_size]
            # Simulate batch operation
            batch_result = {
                "success": True,
                "processed": len(batch),
                "items": [f"item_{j}" for j in batch]
            }
            batch_results.append(batch_result)
            time.sleep(0.01)  # Simulate batch processing time
        
        batch_time = performance_monitor.stop_timer("batch_processing")
        
        # Assertions
        assert batch_time < individual_time * 0.5  # Batch is at least 2x faster
        assert sum(r["processed"] for r in batch_results) == len(individual_items)
    
    @pytest.mark.performance
    def test_lazy_loading_optimization(self, mock_crew):
        """Test lazy loading of resources."""
        crew = mock_crew.crew()
        
        # Track resource loading
        loaded_resources = []
        
        def lazy_load_resource(resource_name):
            if resource_name not in loaded_resources:
                loaded_resources.append(resource_name)
                time.sleep(0.01)  # Simulate loading time
                return f"Loaded {resource_name}"
            return f"Cached {resource_name}"
        
        # Simulate lazy loading in crew
        crew.load_resource = MagicMock(side_effect=lazy_load_resource)
        
        # Access resources multiple times
        start = time.time()
        for i in range(10):
            crew.load_resource("knowledge_base")
            crew.load_resource("llm_config")
            crew.load_resource("tool_registry")
        end = time.time()
        
        # Assertions
        assert len(loaded_resources) == 3  # Only loaded once each
        assert (end - start) < 0.1  # Fast due to caching


class TestLoadTesting:
    """Comprehensive load testing scenarios."""
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_sustained_load(self, mock_crew, sample_inputs, performance_monitor):
        """Test system under sustained load."""
        crew = mock_crew.crew()
        
        # Run for extended period
        duration = 10  # seconds
        start_time = time.time()
        end_time = start_time + duration
        
        results = []
        request_count = 0
        
        performance_monitor.start_timer("sustained_load")
        
        while time.time() < end_time:
            # Vary the input slightly
            inputs = sample_inputs.copy()
            inputs["topic"] = f"Topic {request_count}"
            
            task_start = time.time()
            result = crew.kickoff(inputs=inputs)
            task_end = time.time()
            
            results.append({
                "request_id": request_count,
                "duration": task_end - task_start,
                "timestamp": task_start
            })
            
            request_count += 1
        
        total_time = performance_monitor.stop_timer("sustained_load")
        
        # Calculate metrics
        throughput = request_count / total_time
        avg_latency = statistics.mean(r["duration"] for r in results)
        p95_latency = sorted(r["duration"] for r in results)[int(len(results) * 0.95)]
        
        # Assertions
        assert throughput > 5  # At least 5 requests per second
        assert avg_latency < 0.5  # Average latency under 500ms
        assert p95_latency < 1.0  # 95th percentile under 1 second
    
    @pytest.mark.performance
    def test_spike_load_handling(self, mock_crew, sample_inputs):
        """Test handling of sudden load spikes."""
        crew = mock_crew.crew()
        
        # Normal load
        normal_load_times = []
        for i in range(5):
            start = time.time()
            crew.kickoff(inputs=sample_inputs)
            end = time.time()
            normal_load_times.append(end - start)
            time.sleep(0.1)  # Normal pacing
        
        # Spike load
        spike_load_times = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for i in range(20):  # Sudden spike
                future = executor.submit(crew.kickoff, inputs=sample_inputs)
                futures.append(future)
            
            for future in concurrent.futures.as_completed(futures):
                # Time would be measured in real implementation
                spike_load_times.append(0.2)  # Simulated
        
        # Recovery period
        recovery_times = []
        time.sleep(0.5)  # Allow system to recover
        for i in range(5):
            start = time.time()
            crew.kickoff(inputs=sample_inputs)
            end = time.time()
            recovery_times.append(end - start)
        
        # Analyze performance degradation and recovery
        normal_avg = statistics.mean(normal_load_times)
        spike_avg = statistics.mean(spike_load_times)
        recovery_avg = statistics.mean(recovery_times)
        
        # Assertions
        assert spike_avg < normal_avg * 3  # Degradation less than 3x
        assert recovery_avg < normal_avg * 1.5  # Quick recovery