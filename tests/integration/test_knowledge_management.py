"""Knowledge management and RAG system integration tests - consolidated from root."""

import pytest
import json
import os
from unittest.mock import MagicMock, patch, call
from datetime import datetime
import numpy as np


class TestKnowledgeGraphOperations:
    """Test knowledge graph creation and manipulation."""
    
    def test_entity_creation_and_retrieval(self, mock_mcp_manager):
        """Test creating and retrieving entities in knowledge graph."""
        # Create multiple entity types
        entity_types = [
            ("concept", "machine_learning", ["ML is a subset of AI", "Uses data to improve"]),
            ("framework", "pytorch", ["Deep learning framework", "Developed by Meta"]),
            ("researcher", "yann_lecun", ["Pioneer in deep learning", "Turing Award winner"]),
            ("paper", "attention_is_all_you_need", ["Introduced transformers", "Published in 2017"])
        ]
        
        created_entities = []
        for entity_type, name, observations in entity_types:
            result = mock_mcp_manager.call_tool(
                "memory_create_entity",
                name=name,
                entity_type=entity_type,
                observations=observations
            )
            created_entities.append(result)
        
        # Verify all entities created
        assert len(created_entities) == 4
        assert all(e["success"] for e in created_entities)
        
        # Search for entities
        search_result = mock_mcp_manager.call_tool(
            "memory_search",
            query="deep learning"
        )
        
        assert len(search_result["results"]) >= 2  # Should find pytorch and yann_lecun
    
    def test_relationship_management(self, mock_mcp_manager):
        """Test creating and managing relationships between entities."""
        # Create entities
        entities = {}
        for name in ["neural_networks", "backpropagation", "gradient_descent"]:
            result = mock_mcp_manager.call_tool(
                "memory_create_entity",
                name=name,
                entity_type="concept",
                observations=[f"{name} is a key ML concept"]
            )
            entities[name] = result["entity_id"]
        
        # Create relationships through observations
        relationships = [
            ("neural_networks", "uses", "backpropagation"),
            ("backpropagation", "relies_on", "gradient_descent")
        ]
        
        for source, rel_type, target in relationships:
            mock_mcp_manager.call_tool(
                "memory_add_observation",
                entity_name=source,
                observations=[f"{rel_type}: {target}"]
            )
        
        # Query relationships
        search_result = mock_mcp_manager.call_tool(
            "memory_search",
            query="relies_on gradient_descent"
        )
        
        assert len(search_result["results"]) >= 1
        assert any("backpropagation" in str(r) for r in search_result["results"])
    
    def test_knowledge_graph_traversal(self, mock_mcp_manager):
        """Test traversing knowledge graph for connected information."""
        # Build a connected graph
        graph_data = {
            "ai": ["parent of machine_learning", "parent of nlp"],
            "machine_learning": ["child of ai", "sibling of nlp", "uses statistics"],
            "nlp": ["child of ai", "sibling of machine_learning", "uses linguistics"],
            "statistics": ["used by machine_learning", "foundation of data_science"],
            "linguistics": ["used by nlp", "studies language"]
        }
        
        # Create entities with relationships
        for entity, observations in graph_data.items():
            mock_mcp_manager.call_tool(
                "memory_create_entity",
                name=entity,
                entity_type="field",
                observations=observations
            )
        
        # Traverse from AI to find all connected fields
        def traverse_graph(start_entity, max_depth=3):
            visited = set()
            to_visit = [(start_entity, 0)]
            connections = []
            
            while to_visit and len(visited) < 20:  # Prevent infinite loops
                current, depth = to_visit.pop(0)
                if current in visited or depth > max_depth:
                    continue
                
                visited.add(current)
                result = mock_mcp_manager.call_tool(
                    "memory_search",
                    query=current
                )
                
                connections.append({
                    "entity": current,
                    "depth": depth,
                    "results": result["results"]
                })
                
                # Extract related entities (simplified)
                for r in result["results"]:
                    if "entity" in r:
                        related = r["entity"]
                        if related not in visited:
                            to_visit.append((related, depth + 1))
            
            return connections
        
        # Traverse from AI
        connections = traverse_graph("ai")
        assert len(connections) >= 3  # Should find ai, machine_learning, nlp at minimum
    
    def test_knowledge_evolution(self, mock_mcp_manager):
        """Test how knowledge evolves over time with new observations."""
        entity_name = "transformer_architecture"
        
        # Initial creation
        initial_result = mock_mcp_manager.call_tool(
            "memory_create_entity",
            name=entity_name,
            entity_type="architecture",
            observations=[
                "Introduced in 2017",
                "Uses self-attention mechanism"
            ]
        )
        
        # Evolution over time
        evolution_timeline = [
            ("2018", ["BERT uses transformer", "Bidirectional pretraining"]),
            ("2019", ["GPT-2 released", "Larger models show emergence"]),
            ("2020", ["GPT-3 demonstrates few-shot learning", "175B parameters"]),
            ("2023", ["GPT-4 multimodal capabilities", "ChatGPT mainstream adoption"]),
            ("2024", ["Focus on efficiency", "Small language models trend"])
        ]
        
        for year, observations in evolution_timeline:
            mock_mcp_manager.call_tool(
                "memory_add_observation",
                entity_name=entity_name,
                observations=[f"{year}: {obs}" for obs in observations]
            )
        
        # Verify knowledge evolution
        search_result = mock_mcp_manager.call_tool(
            "memory_search",
            query=f"{entity_name} evolution"
        )
        
        assert search_result["results"]
        # Would contain all historical observations


class TestRAGIntegration:
    """Test Retrieval-Augmented Generation integration."""
    
    def test_document_chunking_and_embedding(self, mock_chromadb, temp_workspace):
        """Test document chunking and embedding for RAG."""
        # Create test document
        document = """
        # Machine Learning Fundamentals
        
        ## Introduction
        Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.
        
        ## Types of Machine Learning
        1. Supervised Learning: Learning from labeled data
        2. Unsupervised Learning: Finding patterns in unlabeled data
        3. Reinforcement Learning: Learning through interaction with environment
        
        ## Key Algorithms
        - Linear Regression
        - Decision Trees
        - Neural Networks
        - Support Vector Machines
        """
        
        # Chunk document
        def chunk_document(text, chunk_size=200, overlap=50):
            words = text.split()
            chunks = []
            for i in range(0, len(words), chunk_size - overlap):
                chunk = ' '.join(words[i:i + chunk_size])
                chunks.append(chunk)
            return chunks
        
        chunks = chunk_document(document)
        assert len(chunks) >= 2
        
        # Generate embeddings (simulated)
        def generate_embeddings(texts):
            # Simulate embeddings (in practice, use actual embedding model)
            return [np.random.rand(384).tolist() for _ in texts]
        
        embeddings = generate_embeddings(chunks)
        assert len(embeddings) == len(chunks)
        
        # Store in ChromaDB
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            mock_chromadb.add(
                documents=[chunk],
                embeddings=[embedding],
                ids=[f"chunk_{i}"]
            )
        
        # Verify storage
        assert mock_chromadb.count() == len(chunks)
    
    def test_semantic_search_retrieval(self, mock_chromadb):
        """Test semantic search and retrieval from vector database."""
        # Query for relevant chunks
        query = "What are the types of machine learning?"
        query_embedding = np.random.rand(384).tolist()
        
        # Search in ChromaDB
        results = mock_chromadb.query(
            query_embeddings=[query_embedding],
            n_results=3
        )
        
        assert "documents" in results
        assert len(results["documents"][0]) <= 3
    
    def test_context_augmentation(self, mock_chromadb, mock_llm):
        """Test augmenting context with retrieved information."""
        query = "Explain supervised learning"
        
        # Retrieve relevant context
        query_embedding = np.random.rand(384).tolist()
        retrieved_docs = mock_chromadb.query(
            query_embeddings=[query_embedding],
            n_results=3
        )
        
        # Augment prompt with context
        context = "\n".join(retrieved_docs["documents"][0])
        augmented_prompt = f"""
        Context: {context}
        
        Question: {query}
        
        Please provide a comprehensive answer based on the context provided.
        """
        
        # Generate response
        response = mock_llm.generate(augmented_prompt)
        
        assert response is not None
        assert len(response) > 0
    
    def test_hybrid_search(self, mock_chromadb, mock_mcp_manager):
        """Test hybrid search combining vector similarity and knowledge graph."""
        query = "neural network optimization"
        
        # Vector search
        query_embedding = np.random.rand(384).tolist()
        vector_results = mock_chromadb.query(
            query_embeddings=[query_embedding],
            n_results=5
        )
        
        # Knowledge graph search
        kg_results = mock_mcp_manager.call_tool(
            "memory_search",
            query=query
        )
        
        # Combine results
        combined_results = {
            "vector_results": vector_results["documents"][0],
            "knowledge_graph": kg_results["results"],
            "query": query
        }
        
        assert len(combined_results["vector_results"]) > 0
        assert len(combined_results["knowledge_graph"]) > 0


class TestKnowledgeOrganization:
    """Test knowledge organization and structuring."""
    
    def test_hierarchical_knowledge_structure(self, mock_mcp_manager):
        """Test creating hierarchical knowledge structures."""
        # Create hierarchy: AI -> ML -> Deep Learning -> Transformers
        hierarchy = [
            ("ai", "field", "root", []),
            ("machine_learning", "subfield", "ai", ["Subset of AI"]),
            ("deep_learning", "subfield", "machine_learning", ["Uses neural networks"]),
            ("transformers", "architecture", "deep_learning", ["Attention-based models"])
        ]
        
        created_entities = {}
        for name, entity_type, parent, observations in hierarchy:
            # Create entity
            result = mock_mcp_manager.call_tool(
                "memory_create_entity",
                name=name,
                entity_type=entity_type,
                observations=observations + ([f"Child of {parent}"] if parent else [])
            )
            created_entities[name] = result
            
            # Link to parent
            if parent and parent in created_entities:
                mock_mcp_manager.call_tool(
                    "memory_add_observation",
                    entity_name=parent,
                    observations=[f"Parent of {name}"]
                )
        
        # Test traversal
        def get_descendants(entity):
            search_result = mock_mcp_manager.call_tool(
                "memory_search",
                query=f"Parent of {entity}"
            )
            return [r for r in search_result["results"] if "Parent of" in str(r)]
        
        ai_descendants = get_descendants("ai")
        assert len(ai_descendants) >= 1  # Should have machine_learning as descendant
    
    def test_knowledge_categorization(self, mock_mcp_manager):
        """Test automatic knowledge categorization."""
        # Sample knowledge items to categorize
        knowledge_items = [
            ("python", ["programming language", "interpreted", "object-oriented"]),
            ("tensorflow", ["machine learning framework", "Google", "deep learning"]),
            ("attention_mechanism", ["neural network component", "transformers", "sequence modeling"]),
            ("supervised_learning", ["ML paradigm", "labeled data", "prediction tasks"]),
            ("bert", ["language model", "bidirectional", "pre-trained"])
        ]
        
        # Create entities and auto-categorize
        categories = {
            "programming": [],
            "frameworks": [],
            "concepts": [],
            "models": []
        }
        
        for name, observations in knowledge_items:
            # Simple categorization logic
            if "programming language" in observations:
                category = "programming"
            elif "framework" in ' '.join(observations):
                category = "frameworks"
            elif "model" in ' '.join(observations):
                category = "models"
            else:
                category = "concepts"
            
            # Create entity with category
            result = mock_mcp_manager.call_tool(
                "memory_create_entity",
                name=name,
                entity_type=category,
                observations=observations + [f"Category: {category}"]
            )
            categories[category].append(result)
        
        # Verify categorization
        assert len(categories["programming"]) >= 1
        assert len(categories["frameworks"]) >= 1
        assert len(categories["concepts"]) >= 1
        assert len(categories["models"]) >= 1
    
    def test_knowledge_deduplication(self, mock_mcp_manager):
        """Test deduplication of similar knowledge entries."""
        # Create similar entities
        similar_entities = [
            ("neural_networks", ["artificial neurons", "deep learning", "backpropagation"]),
            ("neural_nets", ["artificial neurons", "machine learning", "gradient descent"]),
            ("artificial_neural_networks", ["biological inspiration", "deep learning", "weights"])
        ]
        
        created_entities = []
        for name, observations in similar_entities:
            result = mock_mcp_manager.call_tool(
                "memory_create_entity",
                name=name,
                entity_type="concept",
                observations=observations
            )
            created_entities.append((name, result, observations))
        
        # Deduplication logic
        def calculate_similarity(obs1, obs2):
            # Simple word overlap similarity
            words1 = set(' '.join(obs1).lower().split())
            words2 = set(' '.join(obs2).lower().split())
            overlap = len(words1.intersection(words2))
            total = len(words1.union(words2))
            return overlap / total if total > 0 else 0
        
        # Find duplicates
        duplicates = []
        for i, (name1, result1, obs1) in enumerate(created_entities):
            for j, (name2, result2, obs2) in enumerate(created_entities[i+1:], i+1):
                similarity = calculate_similarity(obs1, obs2)
                if similarity > 0.5:  # Threshold for similarity
                    duplicates.append((name1, name2, similarity))
        
        # Should detect similarity between neural network variants
        assert len(duplicates) >= 1
        assert any(dup[2] > 0.5 for dup in duplicates)


class TestKnowledgeQualityControl:
    """Test knowledge quality control mechanisms."""
    
    def test_knowledge_validation(self, mock_mcp_manager):
        """Test validation of knowledge entries for accuracy and completeness."""
        # Create entities with varying quality
        entities_data = [
            ("good_entity", ["Complete description", "Multiple reliable sources", "Recent information"]),
            ("incomplete_entity", ["Brief description"]),  # Too brief
            ("outdated_entity", ["Information from 1990", "No recent updates"]),  # Outdated
            ("conflicting_entity", ["Python is compiled", "Python is interpreted"])  # Conflicting
        ]
        
        validation_results = []
        for name, observations in entities_data:
            # Create entity
            result = mock_mcp_manager.call_tool(
                "memory_create_entity",
                name=name,
                entity_type="test_entity",
                observations=observations
            )
            
            # Validate entity
            validation = {
                "name": name,
                "completeness": len(observations) >= 3,
                "recency": not any("1990" in obs for obs in observations),
                "consistency": not self._has_contradictions(observations),
                "quality_score": 0
            }
            
            # Calculate quality score
            validation["quality_score"] = sum([
                validation["completeness"],
                validation["recency"],
                validation["consistency"]
            ]) / 3
            
            validation_results.append(validation)
        
        # Check validation results
        good_entities = [v for v in validation_results if v["quality_score"] > 0.7]
        poor_entities = [v for v in validation_results if v["quality_score"] < 0.5]
        
        assert len(good_entities) >= 1
        assert len(poor_entities) >= 2
    
    def _has_contradictions(self, observations):
        """Simple contradiction detection."""
        # Very basic contradiction detection
        if any("compiled" in obs.lower() for obs in observations) and \
           any("interpreted" in obs.lower() for obs in observations):
            return True
        return False
    
    def test_knowledge_consistency_checking(self, mock_mcp_manager):
        """Test consistency checking across related knowledge entries."""
        # Create related entities with potential inconsistencies
        python_info = [
            ("python_language", ["Python is interpreted", "Created by Guido van Rossum", "Version 3.x current"]),
            ("python_version", ["Python 2.7 is current", "Released in 2010"]),  # Inconsistent with above
            ("python_features", ["Object-oriented", "Dynamic typing", "Interpreted execution"])
        ]
        
        for name, observations in python_info:
            mock_mcp_manager.call_tool(
                "memory_create_entity",
                name=name,
                entity_type="language_info",
                observations=observations
            )
        
        # Check for version consistency
        def check_python_version_consistency(observations):
            version_claims = []
            for obs in observations:
                if "python" in obs.lower() and ("version" in obs.lower() or "current" in obs.lower()):
                    version_claims.append(obs)
            
            # Check for contradictory version claims
            has_v2_current = any("2.7" in claim and "current" in claim for claim in version_claims)
            has_v3_current = any("3.x" in claim and "current" in claim for claim in version_claims)
            
            return not (has_v2_current and has_v3_current)
        
        # Search for Python-related information
        search_result = mock_mcp_manager.call_tool(
            "memory_search",
            query="python version"
        )
        
        all_observations = []
        for result in search_result["results"]:
            if "observations" in result:
                all_observations.extend(result["observations"])
        
        # Check consistency
        is_consistent = check_python_version_consistency(all_observations)
        assert not is_consistent  # Should detect the inconsistency
    
    def test_knowledge_completeness_assessment(self, mock_mcp_manager, sample_knowledge_graph):
        """Test assessment of knowledge completeness for a topic."""
        topic = "machine_learning_pipeline"
        
        # Expected components of a complete ML pipeline
        expected_components = [
            "data_collection", "data_preprocessing", "feature_engineering",
            "model_selection", "training", "validation", "deployment", "monitoring"
        ]
        
        # Create partial knowledge
        partial_components = ["data_collection", "training", "deployment"]
        for component in partial_components:
            mock_mcp_manager.call_tool(
                "memory_create_entity",
                name=f"{topic}_{component}",
                entity_type="pipeline_component",
                observations=[f"Component of {topic}"]
            )
        
        # Assess completeness
        def assess_coverage(topic, expected):
            found_components = []
            for component in expected:
                search_result = mock_mcp_manager.call_tool(
                    "memory_search",
                    query=f"{topic} {component}"
                )
                if search_result["results"]:
                    found_components.append(component)
            
            coverage = len(found_components) / len(expected)
            missing = set(expected) - set(found_components)
            
            return {
                "coverage": coverage,
                "found": found_components,
                "missing": list(missing)
            }
        
        assessment = assess_coverage(topic, expected_components)
        
        # Should identify missing components
        assert assessment["coverage"] < 1.0
        assert len(assessment["missing"]) > 0
        assert "feature_engineering" in assessment["missing"]
        assert "validation" in assessment["missing"] 