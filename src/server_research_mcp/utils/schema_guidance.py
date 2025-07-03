"""
Schema-Guided Output Utilities for CrewAI Integration

This module provides utilities for integrating Pydantic schemas with CrewAI's
structured output capabilities, enabling reliable and validated AI outputs.
"""

from typing import Type, Tuple, Any, Union, Dict, List, Optional
from crewai import Task, TaskOutput
from crewai.llm import LLM
from pydantic import BaseModel, ValidationError
import json
import logging

from ..schemas.research_paper import (
    EnrichedQuery, 
    RawPaperData, 
    ResearchPaperSchema,
    PaperMetadata,
    Author
)
from ..schemas.obsidian_meta import (
    ObsidianDocument,
    ObsidianFrontmatter,
    ObsidianLink
)

logger = logging.getLogger(__name__)

# =============================================================================
# Schema Validation Guardrails
# =============================================================================

def create_schema_guardrail(schema_model: Type[BaseModel], strict: bool = False) -> callable:
    """
    Create a guardrail function for validating task output against a Pydantic schema.
    
    Args:
        schema_model: The Pydantic model to validate against
        strict: If True, fails on any validation error. If False, provides guidance.
    
    Returns:
        A guardrail function suitable for CrewAI Task.guardrail parameter
    """
    def validate_schema(result: TaskOutput) -> Tuple[bool, Any]:
        """Validate task output against the schema."""
        try:
            # Extract content based on result type
            if hasattr(result, 'raw'):
                content = result.raw
            elif isinstance(result, str):
                content = result
            else:
                content = str(result)
            
            # Try to parse as JSON first
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                if strict:
                    return (False, f"Output must be valid JSON for {schema_model.__name__}")
                else:
                    # Try to guide the AI to produce JSON
                    return (False, f"Please format your response as valid JSON matching the {schema_model.__name__} schema")
            
            # Validate against schema
            validated_instance = schema_model.model_validate(data)
            
            # Return the validated instance as JSON
            return (True, validated_instance.model_dump_json(indent=2))
            
        except ValidationError as e:
            error_details = []
            for error in e.errors():
                field = " -> ".join(str(loc) for loc in error['loc'])
                error_details.append(f"Field '{field}': {error['msg']}")
            
            if strict:
                return (False, f"Schema validation failed:\n" + "\n".join(error_details))
            else:
                # Provide helpful guidance
                guidance = f"Schema validation guidance for {schema_model.__name__}:\n"
                guidance += "\n".join(error_details)
                guidance += f"\n\nPlease ensure your JSON output includes all required fields and matches the expected format."
                return (False, guidance)
                
        except Exception as e:
            return (False, f"Unexpected validation error: {str(e)}")
    
    return validate_schema


def create_flexible_guardrail(schema_model: Type[BaseModel]) -> callable:
    """
    Create a flexible guardrail that provides guidance rather than strict enforcement.
    This allows the AI to iterate and improve while maintaining structure.
    """
    def flexible_validate(result: TaskOutput) -> Tuple[bool, Any]:
        """Flexible validation with improvement suggestions."""
        try:
            # Extract content
            if hasattr(result, 'raw'):
                content = result.raw
            elif isinstance(result, str):
                content = result
            else:
                content = str(result)
            
            # Try JSON parsing
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                # For non-JSON content, check if it contains required information
                schema_info = schema_model.model_json_schema()
                required_fields = schema_info.get('required', [])
                
                suggestions = [
                    f"Output should be formatted as JSON",
                    f"Required fields: {', '.join(required_fields)}",
                    f"Consider structuring your response to match {schema_model.__name__} format"
                ]
                
                return (False, "Format guidance:\n" + "\n".join(suggestions))
            
            # Attempt validation
            try:
                validated_instance = schema_model.model_validate(data)
                return (True, validated_instance.model_dump_json(indent=2))
            except ValidationError as e:
                # Extract partial data and provide improvement suggestions
                partial_data = {}
                suggestions = []
                
                for error in e.errors():
                    field_path = " -> ".join(str(loc) for loc in error['loc'])
                    if error['type'] == 'missing':
                        suggestions.append(f"Missing required field: {field_path}")
                    else:
                        suggestions.append(f"Field '{field_path}': {error['msg']}")
                
                # Try to salvage what we can
                schema_fields = schema_model.model_json_schema().get('properties', {})
                for field_name in schema_fields:
                    if field_name in data:
                        partial_data[field_name] = data[field_name]
                
                guidance = f"Partial success! Found: {list(partial_data.keys())}\n"
                guidance += "Improvements needed:\n" + "\n".join(suggestions)
                
                return (False, guidance)
                
        except Exception as e:
            return (False, f"Processing error: {str(e)}")
    
    return flexible_validate


# =============================================================================
# Enhanced Task Creation
# =============================================================================

def create_schema_guided_task(
    description: str,
    expected_output: str,
    agent,
    schema_model: Type[BaseModel],
    validation_level: str = "medium",
    max_retries: int = 2,
    **kwargs
) -> Task:
    """
    Create a CrewAI task with schema-guided output.
    
    Args:
        description: Task description
        expected_output: Description of expected output
        agent: The agent to execute the task
        schema_model: Pydantic model for validation
        validation_level: "strict", "medium", or "flexible"
        max_retries: Number of retry attempts
        **kwargs: Additional Task parameters
    
    Returns:
        Configured CrewAI Task with schema validation
    """
    # Choose guardrail based on validation level
    if validation_level == "strict":
        guardrail = create_schema_guardrail(schema_model, strict=True)
    elif validation_level == "flexible":
        guardrail = create_flexible_guardrail(schema_model)
    else:  # medium
        guardrail = create_schema_guardrail(schema_model, strict=False)
    
    # Enhance description with schema guidance
    schema_info = schema_model.model_json_schema()
    enhanced_description = f"{description}\n\n"
    enhanced_description += f"OUTPUT FORMAT: Structure your response as JSON matching the {schema_model.__name__} schema.\n"
    
    if 'required' in schema_info:
        enhanced_description += f"Required fields: {', '.join(schema_info['required'])}\n"
    
    enhanced_description += f"Ensure your response is valid JSON that can be parsed and validated."
    
    # Enhanced expected output
    enhanced_expected_output = f"{expected_output}\n\n"
    enhanced_expected_output += f"Format: Valid JSON conforming to {schema_model.__name__} schema.\n"
    enhanced_expected_output += "Include all required fields with appropriate data types."
    
    return Task(
        description=enhanced_description,
        expected_output=enhanced_expected_output,
        agent=agent,
        output_pydantic=schema_model,  # Use CrewAI's built-in Pydantic support
        guardrail=guardrail,
        max_retries=max_retries,
        **kwargs
    )


# =============================================================================
# Schema Registry and Utilities
# =============================================================================

SCHEMA_REGISTRY = {
    "EnrichedQuery": EnrichedQuery,
    "RawPaperData": RawPaperData,
    "ResearchPaperSchema": ResearchPaperSchema,
    "ObsidianDocument": ObsidianDocument,
    "ObsidianFrontmatter": ObsidianFrontmatter,
    "PaperMetadata": PaperMetadata,
    "Author": Author,
}

def get_schema_model(schema_name: str) -> Type[BaseModel]:
    """Get a schema model by name from the registry."""
    if schema_name not in SCHEMA_REGISTRY:
        raise ValueError(f"Unknown schema: {schema_name}. Available: {list(SCHEMA_REGISTRY.keys())}")
    return SCHEMA_REGISTRY[schema_name]


def validate_output_against_schema(output: str, schema_name: str) -> Tuple[bool, Union[BaseModel, str]]:
    """
    Validate output against a named schema.
    
    Returns:
        Tuple of (success, validated_model_or_error_message)
    """
    try:
        schema_model = get_schema_model(schema_name)
        data = json.loads(output)
        validated = schema_model.model_validate(data)
        return (True, validated)
    except (json.JSONDecodeError, ValidationError, ValueError) as e:
        return (False, str(e))


def enhance_llm_prompt_for_schema(base_prompt: str, schema_model: Type[BaseModel]) -> str:
    """
    Enhance an LLM prompt with schema-specific instructions.
    
    This can be used for direct LLM calls within flows or agents.
    """
    schema_info = schema_model.model_json_schema()
    
    enhanced_prompt = f"{base_prompt}\n\n"
    enhanced_prompt += f"IMPORTANT: Your response must be valid JSON matching the {schema_model.__name__} schema.\n\n"
    enhanced_prompt += f"Schema structure:\n```json\n{json.dumps(schema_info, indent=2)}\n```\n\n"
    enhanced_prompt += "Ensure your entire response is ONLY the valid JSON object, without any introductory text, explanations, or concluding remarks."
    
    return enhanced_prompt


# =============================================================================
# Validation Chain Utilities
# =============================================================================

def chain_schema_validations(*schema_models: Type[BaseModel]) -> callable:
    """
    Create a guardrail that validates against multiple schemas in sequence.
    Useful for complex workflows with multiple validation stages.
    """
    def multi_schema_validate(result: TaskOutput) -> Tuple[bool, Any]:
        """Validate against multiple schemas."""
        content = result.raw if hasattr(result, 'raw') else str(result)
        
        for i, schema_model in enumerate(schema_models):
            try:
                data = json.loads(content)
                validated = schema_model.model_validate(data)
                content = validated.model_dump_json()
            except (json.JSONDecodeError, ValidationError) as e:
                return (False, f"Validation failed at stage {i+1} ({schema_model.__name__}): {str(e)}")
        
        return (True, content)
    
    return multi_schema_validate


# =============================================================================
# Integration Helpers
# =============================================================================

def create_schema_guided_llm_call(
    llm: LLM,
    prompt: str,
    schema_model: Type[BaseModel],
    max_retries: int = 2
) -> BaseModel:
    """
    Make a schema-guided LLM call with automatic retries and validation.
    Useful for direct LLM usage in flows.
    """
    enhanced_prompt = enhance_llm_prompt_for_schema(prompt, schema_model)
    
    for attempt in range(max_retries + 1):
        try:
            response = llm.call(enhanced_prompt)
            data = json.loads(response)
            return schema_model.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            if attempt < max_retries:
                enhanced_prompt += f"\n\nPrevious attempt failed: {str(e)}. Please correct and provide valid JSON."
            else:
                raise ValueError(f"Failed to get valid schema output after {max_retries + 1} attempts: {str(e)}")


def apply_schema_guidance_to_existing_task(task: Task, schema_name: str, validation_level: str = "medium") -> Task:
    """
    Apply schema guidance to an existing task by updating its configuration.
    """
    schema_model = get_schema_model(schema_name)
    
    # Update task properties
    task.output_pydantic = schema_model
    
    if validation_level == "strict":
        task.guardrail = create_schema_guardrail(schema_model, strict=True)
    elif validation_level == "flexible":
        task.guardrail = create_flexible_guardrail(schema_model)
    else:
        task.guardrail = create_schema_guardrail(schema_model, strict=False)
    
    return task


# =============================================================================
# Example Usage and Testing
# =============================================================================

def example_schema_guided_workflow():
    """Example of how to use schema-guided output in a CrewAI workflow."""
    
    # Example 1: Create a schema-guided task
    enriched_query_task = create_schema_guided_task(
        description="Gather context for research query about AI safety",
        expected_output="Enriched query with expanded terms and context",
        agent=None,  # Would be your actual agent
        schema_model=EnrichedQuery,
        validation_level="medium",
        max_retries=2
    )
    
    # Example 2: Validate existing output
    sample_output = '{"original_query": "AI safety", "expanded_terms": ["AI", "safety"], "search_strategy": "comprehensive"}'
    success, result = validate_output_against_schema(sample_output, "EnrichedQuery")
    
    if success:
        logger.info(f"Validation successful: {result}")
    else:
        logger.error(f"Validation failed: {result}")
    
    # Example 3: Enhanced LLM prompt
    base_prompt = "Analyze the research topic 'machine learning ethics'"
    enhanced_prompt = enhance_llm_prompt_for_schema(base_prompt, ResearchPaperSchema)
    
    return {
        "task": enriched_query_task,
        "validation_result": (success, result),
        "enhanced_prompt": enhanced_prompt
    }


if __name__ == "__main__":
    # Run example
    examples = example_schema_guided_workflow()
    print("Schema guidance utilities loaded successfully!")
    print(f"Available schemas: {list(SCHEMA_REGISTRY.keys())}") 