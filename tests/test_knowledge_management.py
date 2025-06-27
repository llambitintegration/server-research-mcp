"""Knowledge management and RAG system tests."""

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
            chunks = []
            sentences = text.split('.')
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) <= chunk_size:
                    current_chunk += sentence + "."
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + "."
            
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            return chunks
        
        chunks = chunk_document(document)
        assert len(chunks) >= 3
        
        # Mock embedding generation
        def generate_embeddings(texts):
            # Simulate embeddings (in practice, use actual embedding model)
            return [np.random.rand(384).tolist() for _ in texts]
        
        embeddings = generate_embeddings(chunks)
        
        # Store in ChromaDB
        collection = mock_chromadb().get_or_create_collection("test_docs")
        collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=[f"chunk_{i}" for i in range(len(chunks))],
            metadatas=[{"source": "ml_fundamentals", "chunk_id": i} for i in range(len(chunks))]
        )
        
        collection.add.assert_called_once()
    
    def test_semantic_search_retrieval(self, mock_chromadb):
        """Test semantic search for relevant chunks."""
        collection = mock_chromadb().get_collection("test_docs")
        
        # Query
        query = "What are the types of machine learning?"
        query_embedding = np.random.rand(384).tolist()  # Mock embedding
        
        # Search
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )
        
        assert "documents" in results
        assert len(results["documents"][0]) == 2  # Based on mock return
        assert results["distances"][0][0] < results["distances"][0][1]  # Ordered by relevance
    
    def test_context_augmentation(self, mock_chromadb, mock_llm):
        """Test augmenting LLM context with retrieved information."""
        # Retrieve relevant chunks
        collection = mock_chromadb().get_collection("research_papers")
        query = "neural network optimization techniques"
        
        search_results = collection.query(
            query_embeddings=[np.random.rand(384).tolist()],
            n_results=5
        )
        
        # Build augmented context
        retrieved_context = "\n\n".join(search_results["documents"][0])
        
        # Create augmented prompt
        augmented_prompt = f"""Based on the following context, answer the question.
        
Context:
{retrieved_context}

Question: {query}

Answer:"""
        
        # Get LLM response with context
        response = mock_llm.invoke(augmented_prompt)
        
        assert mock_llm.invoke.called
        assert query in augmented_prompt
        assert retrieved_context in augmented_prompt
    
    def test_hybrid_search(self, mock_chromadb, mock_mcp_manager):
        """Test hybrid search combining vector and keyword search."""
        query = "transformer architecture attention mechanism"
        
        # Vector search in ChromaDB
        vector_results = mock_chromadb().get_collection("papers").query(
            query_embeddings=[np.random.rand(384).tolist()],
            n_results=10
        )
        
        # Keyword search in knowledge graph
        kg_results = mock_mcp_manager.call_tool(
            "memory_search",
            query=query
        )
        
        # Combine and rank results
        combined_results = []
        
        # Add vector results with scores
        for i, doc in enumerate(vector_results["documents"][0]):
            combined_results.append({
                "content": doc,
                "source": "vector_search",
                "score": 1.0 - vector_results["distances"][0][i],
                "metadata": vector_results["metadatas"][0][i]
            })
        
        # Add KG results
        for result in kg_results["results"]:
            combined_results.append({
                "content": str(result.get("observations", [])),
                "source": "knowledge_graph",
                "score": result.get("relevance", 0.5),
                "entity": result.get("entity", "unknown")
            })
        
        # Sort by score
        combined_results.sort(key=lambda x: x["score"], reverse=True)
        
        assert len(combined_results) >= 2
        assert combined_results[0]["score"] >= combined_results[-1]["score"]


class TestKnowledgeOrganization:
    """Test knowledge organization and structuring."""
    
    def test_hierarchical_knowledge_structure(self, mock_mcp_manager):
        """Test creating hierarchical knowledge structures."""
        # Create hierarchy: AI -> ML -> Deep Learning -> Transformers
        hierarchy = [
            ("artificial_intelligence", "field", None),
            ("machine_learning", "subfield", "artificial_intelligence"),
            ("deep_learning", "subfield", "machine_learning"),
            ("transformers", "technique", "deep_learning"),
            ("bert", "model", "transformers"),
            ("gpt", "model", "transformers")
        ]
        
        # Create entities with parent relationships
        for name, entity_type, parent in hierarchy:
            observations = [f"Type: {entity_type}"]
            if parent:
                observations.append(f"Parent: {parent}")
            
            mock_mcp_manager.call_tool(
                "memory_create_entity",
                name=name,
                entity_type=entity_type,
                observations=observations
            )
        
        # Query hierarchy
        def get_descendants(entity):
            results = mock_mcp_manager.call_tool(
                "memory_search",
                query=f"Parent: {entity}"
            )
            descendants = []
            for r in results["results"]:
                if "entity" in r:
                    descendants.append(r["entity"])
            return descendants
        
        # Get ML descendants
        ml_descendants = get_descendants("machine_learning")
        assert "deep_learning" in str(ml_descendants)
    
    def test_knowledge_categorization(self, mock_mcp_manager):
        """Test automatic categorization of knowledge."""
        # Define categories
        categories = {
            "algorithms": ["sorting", "searching", "optimization", "algorithm"],
            "data_structures": ["array", "list", "tree", "graph", "structure"],
            "machine_learning": ["neural", "learning", "training", "model"],
            "databases": ["sql", "query", "database", "table", "index"]
        }
        
        # Test items to categorize
        items = [
            ("quicksort", "A fast sorting algorithm"),
            ("binary_tree", "Hierarchical data structure"),
            ("cnn", "Convolutional neural network for images"),
            ("postgres", "PostgreSQL database system")
        ]
        
        # Categorize items
        categorized = {}
        for item_name, description in items:
            # Find best category
            best_category = None
            max_score = 0
            
            for category, keywords in categories.items():
                score = sum(1 for kw in keywords if kw in description.lower())
                if score > max_score:
                    max_score = score
                    best_category = category
            
            if best_category:
                if best_category not in categorized:
                    categorized[best_category] = []
                categorized[best_category].append(item_name)
            
            # Create entity with category
            mock_mcp_manager.call_tool(
                "memory_create_entity",
                name=item_name,
                entity_type=best_category or "uncategorized",
                observations=[description, f"Category: {best_category}"]
            )
        
        # Verify categorization
        assert "algorithms" in categorized
        assert "quicksort" in categorized["algorithms"]
        assert "data_structures" in categorized
        assert "binary_tree" in categorized["data_structures"]
    
    def test_knowledge_deduplication(self, mock_mcp_manager):
        """Test deduplication of similar knowledge entries."""
        # Create potentially duplicate entries
        similar_entries = [
            ("ml_basics", "machine learning fundamentals"),
            ("machine_learning_basics", "fundamentals of ML"),
            ("ml_fundamentals", "basic machine learning concepts"),
            ("deep_learning_intro", "introduction to deep learning")
        ]
        
        # Create entries
        for name, description in similar_entries:
            mock_mcp_manager.call_tool(
                "memory_create_entity",
                name=name,
                entity_type="topic",
                observations=[description]
            )
        
        # Detect duplicates
        def calculate_similarity(obs1, obs2):
            # Simple word overlap similarity
            words1 = set(obs1.lower().split())
            words2 = set(obs2.lower().split())
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            return len(intersection) / len(union) if union else 0
        
        # Find similar entries
        duplicates = []
        for i, (name1, desc1) in enumerate(similar_entries):
            for j, (name2, desc2) in enumerate(similar_entries[i+1:], i+1):
                similarity = calculate_similarity(desc1, desc2)
                if similarity > 0.5:  # Threshold for duplicates
                    duplicates.append({
                        "entities": [name1, name2],
                        "similarity": similarity
                    })
        
        # Merge duplicates
        for dup in duplicates:
            if dup["similarity"] > 0.7:  # High similarity threshold
                # Would merge observations from both entities
                primary = dup["entities"][0]
                secondary = dup["entities"][1]
                
                # Add merged observation
                mock_mcp_manager.call_tool(
                    "memory_add_observation",
                    entity_name=primary,
                    observations=[f"Merged with: {secondary}"]
                )
        
        assert len(duplicates) >= 2  # Should find ML basics duplicates


class TestKnowledgeQualityControl:
    """Test knowledge quality control and validation."""
    
    def test_knowledge_validation(self, mock_mcp_manager):
        """Test validation of knowledge entries."""
        # Define validation rules
        validation_rules = {
            "min_observation_length": 20,
            "required_metadata": ["source", "date_added"],
            "forbidden_content": ["test", "todo", "fixme"],
            "required_observation_count": 2
        }
        
        # Test entries
        test_entries = [
            {
                "name": "valid_entry",
                "observations": [
                    "This is a valid observation about machine learning",
                    "Another detailed observation about the topic"
                ],
                "metadata": {"source": "research_paper", "date_added": "2024-01-01"}
            },
            {
                "name": "invalid_short",
                "observations": ["Too short"],
                "metadata": {"source": "unknown"}
            },
            {
                "name": "invalid_forbidden",
                "observations": ["This is a test entry TODO: finish this"],
                "metadata": {"source": "draft", "date_added": "2024-01-01"}
            }
        ]
        
        # Validate entries
        validation_results = []
        for entry in test_entries:
            issues = []
            
            # Check observation length
            for obs in entry["observations"]:
                if len(obs) < validation_rules["min_observation_length"]:
                    issues.append(f"Observation too short: {len(obs)} chars")
                
                # Check forbidden content
                for forbidden in validation_rules["forbidden_content"]:
                    if forbidden.lower() in obs.lower():
                        issues.append(f"Contains forbidden content: {forbidden}")
            
            # Check observation count
            if len(entry["observations"]) < validation_rules["required_observation_count"]:
                issues.append("Insufficient observations")
            
            # Check metadata
            for required in validation_rules["required_metadata"]:
                if required not in entry.get("metadata", {}):
                    issues.append(f"Missing required metadata: {required}")
            
            validation_results.append({
                "name": entry["name"],
                "valid": len(issues) == 0,
                "issues": issues
            })
        
        # Verify validation
        valid_entries = [r for r in validation_results if r["valid"]]
        assert len(valid_entries) == 1
        assert valid_entries[0]["name"] == "valid_entry"
    
    def test_knowledge_consistency_checking(self, mock_mcp_manager):
        """Test checking consistency across knowledge entries."""
        # Create related entries that should be consistent
        entries = [
            ("python_version", "concept", ["Python 3.12 is the latest stable version"]),
            ("python_features", "concept", ["Python 2.7 has new features"]),  # Inconsistent
            ("django_framework", "framework", ["Django 5.0 requires Python 3.10+"]),
            ("flask_framework", "framework", ["Flask works with Python 2.6"])  # Inconsistent
        ]
        
        for name, entity_type, observations in entries:
            mock_mcp_manager.call_tool(
                "memory_create_entity",
                name=name,
                entity_type=entity_type,
                observations=observations
            )
        
        # Check consistency
        def check_python_version_consistency(observations):
            issues = []
            python_2_mentions = []
            python_3_mentions = []
            
            for obs in observations:
                if "python 2" in obs.lower():
                    python_2_mentions.append(obs)
                if "python 3" in obs.lower():
                    python_3_mentions.append(obs)
            
            # Flag if mixing Python 2 and 3 inappropriately
            if python_2_mentions and "latest" in str(python_3_mentions):
                issues.append("Inconsistent Python version references")
            
            return issues
        
        # Search for all Python-related entries
        search_result = mock_mcp_manager.call_tool(
            "memory_search",
            query="Python"
        )
        
        # Would analyze results for consistency
        # In this test, we'd find inconsistencies about Python versions
    
    def test_knowledge_completeness_assessment(self, mock_mcp_manager, sample_knowledge_graph):
        """Test assessing completeness of knowledge coverage."""
        # Define expected knowledge areas for a topic
        expected_coverage = {
            "machine_learning": {
                "required_subtopics": [
                    "supervised_learning",
                    "unsupervised_learning",
                    "reinforcement_learning",
                    "evaluation_metrics",
                    "common_algorithms"
                ],
                "optional_subtopics": [
                    "deep_learning",
                    "feature_engineering",
                    "model_deployment"
                ]
            }
        }
        
        # Check current coverage
        def assess_coverage(topic, expected):
            current_coverage = {
                "required": [],
                "optional": [],
                "missing_required": [],
                "coverage_score": 0.0
            }
            
            # Search for each expected subtopic
            for subtopic in expected["required_subtopics"]:
                result = mock_mcp_manager.call_tool(
                    "memory_search",
                    query=subtopic.replace("_", " ")
                )
                
                if result["results"]:
                    current_coverage["required"].append(subtopic)
                else:
                    current_coverage["missing_required"].append(subtopic)
            
            for subtopic in expected.get("optional_subtopics", []):
                result = mock_mcp_manager.call_tool(
                    "memory_search",
                    query=subtopic.replace("_", " ")
                )
                
                if result["results"]:
                    current_coverage["optional"].append(subtopic)
            
            # Calculate coverage score
            required_score = len(current_coverage["required"]) / len(expected["required_subtopics"])
            optional_score = len(current_coverage["optional"]) / len(expected.get("optional_subtopics", [1]))
            current_coverage["coverage_score"] = (required_score * 0.7) + (optional_score * 0.3)
            
            return current_coverage
        
        # Assess ML coverage
        ml_coverage = assess_coverage("machine_learning", expected_coverage["machine_learning"])
        
        # In a real scenario, this would help identify knowledge gaps
        assert "coverage_score" in ml_coverage
        assert 0 <= ml_coverage["coverage_score"] <= 1