#!/usr/bin/env python3
"""
Demonstration of Schema-Guided Output in CrewAI

This script demonstrates how to use Pydantic schemas to guide AI output
while maintaining flexibility and providing helpful feedback.
"""

import json
import logging
from typing import Dict, Any
from datetime import datetime

from src.server_research_mcp.utils.schema_guidance import (
    create_schema_guided_task,
    create_schema_guardrail,
    create_flexible_guardrail,
    validate_output_against_schema,
    enhance_llm_prompt_for_schema,
    SCHEMA_REGISTRY
)
from src.server_research_mcp.schemas.research_paper import (
    EnrichedQuery,
    RawPaperData,
    ResearchPaperSchema,
    Author,
    PaperMetadata,
    PaperSection
)
from src.server_research_mcp.schemas.obsidian_meta import (
    ObsidianDocument,
    ObsidianFrontmatter
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demonstrate_schema_validation():
    """Demonstrate schema validation with various input scenarios."""
    
    print("=" * 70)
    print("🔬 SCHEMA VALIDATION DEMONSTRATION")
    print("=" * 70)
    
    # Test 1: Valid EnrichedQuery
    print("\n📋 Test 1: Valid EnrichedQuery")
    print("-" * 40)
    
    valid_enriched_query = {
        "original_query": "Machine learning in healthcare",
        "expanded_terms": ["machine learning", "healthcare", "medical AI", "clinical decision support"],
        "related_papers": [
            {"title": "Deep Learning for Medical Diagnosis", "relevance": "high"}
        ],
        "known_authors": [
            {"name": "Dr. Smith", "affiliation": "Medical AI Lab"}
        ],
        "topic_context": {"domain": "healthcare", "subdomain": "diagnostics"},
        "search_strategy": "Focus on recent papers from top medical journals",
        "memory_entities": ["healthcare_ai", "medical_diagnosis"]
    }
    
    json_output = json.dumps(valid_enriched_query, indent=2)
    success, result = validate_output_against_schema(json_output, "EnrichedQuery")
    
    print(f"✅ Validation Success: {success}")
    if success:
        print(f"📊 Validated fields: {list(result.model_dump().keys())}")
    else:
        print(f"❌ Error: {result}")
    
    # Test 2: Invalid EnrichedQuery (missing required fields)
    print("\n📋 Test 2: Invalid EnrichedQuery (Missing Fields)")
    print("-" * 50)
    
    invalid_enriched_query = {
        "original_query": "Machine learning in healthcare",
        # Missing required fields
    }
    
    json_output = json.dumps(invalid_enriched_query, indent=2)
    success, result = validate_output_against_schema(json_output, "EnrichedQuery")
    
    print(f"✅ Validation Success: {success}")
    print(f"❌ Validation Error: {result}")
    
    # Test 3: Partial RawPaperData
    print("\n📋 Test 3: RawPaperData with Flexible Validation")
    print("-" * 50)
    
    partial_paper_data = {
        "metadata": {
            "title": "AI in Medical Diagnosis",
            "authors": ["Dr. Jane Smith", "Dr. John Doe"],
            "year": 2024,
            "journal": "Medical AI Journal"
        },
        "full_text": "Abstract: This paper presents a novel approach...",
        "sections": {
            "abstract": "This paper presents a novel approach to medical diagnosis using AI...",
            "introduction": "Healthcare is rapidly adopting AI technologies..."
        },
        "extraction_method": "automatic",
        "extraction_quality": 0.85
    }
    
    json_output = json.dumps(partial_paper_data, indent=2)
    success, result = validate_output_against_schema(json_output, "RawPaperData")
    
    print(f"✅ Validation Success: {success}")
    if success:
        print(f"📊 Paper Title: {result.metadata['title']}")
        print(f"📊 Authors: {', '.join(result.metadata['authors'])}")
        print(f"📊 Quality Score: {result.extraction_quality}")
    else:
        print(f"❌ Error: {result}")


def demonstrate_guardrail_functions():
    """Demonstrate different levels of guardrail validation."""
    
    print("\n" + "=" * 70)
    print("🛡️ GUARDRAIL FUNCTIONS DEMONSTRATION")
    print("=" * 70)
    
    # Create different guardrails
    strict_guardrail = create_schema_guardrail(EnrichedQuery, strict=True)
    medium_guardrail = create_schema_guardrail(EnrichedQuery, strict=False)
    flexible_guardrail = create_flexible_guardrail(EnrichedQuery)
    
    # Test with invalid JSON
    class MockTaskOutput:
        def __init__(self, content):
            self.raw = content
    
    invalid_json = "This is not JSON at all, just plain text about AI research."
    
    print("\n🔍 Testing with Non-JSON Content:")
    print(f"Input: '{invalid_json[:50]}...'")
    
    # Test strict guardrail
    success, message = strict_guardrail(MockTaskOutput(invalid_json))
    print(f"\n🚫 Strict Guardrail - Success: {success}")
    print(f"   Message: {message}")
    
    # Test medium guardrail
    success, message = medium_guardrail(MockTaskOutput(invalid_json))
    print(f"\n⚖️ Medium Guardrail - Success: {success}")
    print(f"   Message: {message}")
    
    # Test flexible guardrail
    success, message = flexible_guardrail(MockTaskOutput(invalid_json))
    print(f"\n🤝 Flexible Guardrail - Success: {success}")
    print(f"   Message: {message}")


def demonstrate_llm_prompt_enhancement():
    """Demonstrate how schemas enhance LLM prompts."""
    
    print("\n" + "=" * 70)
    print("🎯 LLM PROMPT ENHANCEMENT DEMONSTRATION")
    print("=" * 70)
    
    base_prompt = "Analyze the research topic 'quantum machine learning' and provide context."
    
    print("\n📝 Original Prompt:")
    print("-" * 20)
    print(base_prompt)
    
    enhanced_prompt = enhance_llm_prompt_for_schema(base_prompt, EnrichedQuery)
    
    print("\n🎯 Enhanced Prompt with Schema Guidance:")
    print("-" * 45)
    print(enhanced_prompt[:500] + "..." if len(enhanced_prompt) > 500 else enhanced_prompt)
    
    print(f"\n📊 Enhancement adds {len(enhanced_prompt) - len(base_prompt)} characters of guidance")


def demonstrate_obsidian_schema():
    """Demonstrate Obsidian document schema validation."""
    
    print("\n" + "=" * 70)
    print("📝 OBSIDIAN DOCUMENT SCHEMA DEMONSTRATION")
    print("=" * 70)
    
    # Create a sample Obsidian document
    obsidian_doc = {
        "frontmatter": {
            "title": "Quantum Machine Learning: A Survey",
            "authors": ["Dr. Alice Johnson", "Dr. Bob Wilson"],
            "year": 2024,
            "created": "2024-01-15 10:30:00",
            "modified": "2024-01-15 14:45:00",
            "tags": ["quantum-computing", "machine-learning", "survey"],
            "aliases": ["QML Survey", "Quantum ML"],
            "journal": "Quantum Computing Review",
            "doi": "10.1234/qcr.2024.001",
            "keywords": ["quantum", "machine learning", "algorithms"],
            "related_papers": ["[[Quantum Algorithms]]", "[[ML Fundamentals]]"],
            "quality_score": 9.2,
            "read_status": "read"
        },
        "content": "# Quantum Machine Learning: A Survey\n\n## Abstract\n\nThis paper surveys...",
        "vault_path": "Papers/Quantum Computing/QML_Survey_2024.md"
    }
    
    json_output = json.dumps(obsidian_doc, indent=2)
    success, result = validate_output_against_schema(json_output, "ObsidianDocument")
    
    print(f"✅ Validation Success: {success}")
    if success:
        print(f"📄 Document Title: {result.frontmatter.title}")
        print(f"📅 Created: {result.frontmatter.created}")
        print(f"🏷️  Tags: {', '.join(result.frontmatter.tags)}")
        print(f"📍 Vault Path: {result.vault_path}")
        
        # Demonstrate markdown generation
        markdown_content = result.to_markdown()
        print(f"\n📝 Generated Markdown (first 300 chars):")
        print("-" * 45)
        print(markdown_content[:300] + "..." if len(markdown_content) > 300 else markdown_content)
    else:
        print(f"❌ Error: {result}")


def demonstrate_practical_workflow():
    """Demonstrate a practical workflow using schema guidance."""
    
    print("\n" + "=" * 70)
    print("🔄 PRACTICAL WORKFLOW DEMONSTRATION")
    print("=" * 70)
    
    # Simulate a research workflow
    workflow_steps = [
        {
            "step": "Context Gathering",
            "schema": "EnrichedQuery",
            "description": "Historian agent enriches the research query"
        },
        {
            "step": "Paper Extraction", 
            "schema": "RawPaperData",
            "description": "Researcher agent extracts paper data"
        },
        {
            "step": "Analysis & Structuring",
            "schema": "ResearchPaperSchema", 
            "description": "Archivist agent structures the data"
        },
        {
            "step": "Publishing",
            "schema": "ObsidianDocument",
            "description": "Publisher agent creates final document"
        }
    ]
    
    print("\n🔍 Workflow Steps with Schema Validation:")
    print("-" * 45)
    
    for i, step in enumerate(workflow_steps, 1):
        schema_model = SCHEMA_REGISTRY.get(step["schema"])
        if schema_model:
            schema_info = schema_model.model_json_schema()
            required_fields = schema_info.get('required', [])
            
            print(f"\n{i}. {step['step']}")
            print(f"   📋 Schema: {step['schema']}")
            print(f"   📝 Description: {step['description']}")
            print(f"   ✅ Required Fields: {', '.join(required_fields[:3])}{'...' if len(required_fields) > 3 else ''}")
            print(f"   📊 Total Fields: {len(schema_info.get('properties', {}))}")


def demonstrate_error_guidance():
    """Demonstrate how schema validation provides helpful error guidance."""
    
    print("\n" + "=" * 70)
    print("🔧 ERROR GUIDANCE DEMONSTRATION")
    print("=" * 70)
    
    # Test with common errors
    test_cases = [
        {
            "name": "Missing Required Field",
            "data": {
                "expanded_terms": ["AI", "healthcare"],
                "search_strategy": "comprehensive search"
                # Missing "original_query"
            }
        },
        {
            "name": "Wrong Data Type",
            "data": {
                "original_query": "AI in healthcare",
                "expanded_terms": "AI healthcare",  # Should be list
                "search_strategy": "comprehensive search",
                "memory_entities": []
            }
        },
        {
            "name": "Invalid Structure",
            "data": {
                "original_query": "AI in healthcare",
                "expanded_terms": ["AI", "healthcare"],
                "search_strategy": "comprehensive search",
                "memory_entities": [],
                "related_papers": "some papers"  # Should be list of objects
            }
        }
    ]
    
    flexible_guardrail = create_flexible_guardrail(EnrichedQuery)
    
    for case in test_cases:
        print(f"\n🔍 Test Case: {case['name']}")
        print("-" * 30)
        
        json_output = json.dumps(case['data'], indent=2)
        mock_output = type('MockOutput', (), {'raw': json_output})()
        
        success, guidance = flexible_guardrail(mock_output)
        print(f"✅ Success: {success}")
        print(f"💡 Guidance:\n{guidance}")


def main():
    """Run all demonstrations."""
    
    print("🚀 SCHEMA-GUIDED OUTPUT DEMONSTRATION")
    print("=====================================")
    print("Demonstrating how Pydantic schemas guide AI output in CrewAI")
    print(f"Available schemas: {', '.join(SCHEMA_REGISTRY.keys())}")
    
    try:
        demonstrate_schema_validation()
        demonstrate_guardrail_functions()
        demonstrate_llm_prompt_enhancement()
        demonstrate_obsidian_schema()
        demonstrate_practical_workflow()
        demonstrate_error_guidance()
        
        print("\n" + "=" * 70)
        print("✅ DEMONSTRATION COMPLETE")
        print("=" * 70)
        print("\n💡 Key Takeaways:")
        print("  • Schemas provide structure without being overly restrictive")
        print("  • Multiple validation levels (strict, medium, flexible)")
        print("  • Helpful error messages guide AI toward correct output")
        print("  • Enhanced prompts include schema information automatically")
        print("  • Validation chains support complex workflows")
        print("  • Integration with CrewAI's built-in structured output features")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    main() 