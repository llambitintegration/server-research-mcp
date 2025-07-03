# Schema-Guided Output in CrewAI

## Overview

This document explains how to use Pydantic schemas to guide AI agent output in your CrewAI research workflow. The approach provides structure without being overly restrictive, allowing agents to learn and improve while maintaining data consistency.

## Key Components

### 1. Schema Guidance Utilities (`src/utils/schema_guidance.py`)

The core utilities provide:
- **Flexible validation**: Guide rather than block AI output
- **Guardrail functions**: Multiple validation levels (strict, medium, flexible)
- **Enhanced prompts**: Automatic schema information injection
- **Task creators**: Schema-aware CrewAI Task factory functions

### 2. Existing Schemas

Your project includes well-designed schemas:

- **`EnrichedQuery`**: Historian agent output for context gathering
- **`RawPaperData`**: Researcher agent output for paper extraction  
- **`ResearchPaperSchema`**: Archivist agent output for structured papers
- **`ObsidianDocument`**: Publisher agent output for final documents

## Implementation Strategy

### Agent Configuration

```python
@agent
def historian(self) -> Agent:
    return Agent(
        config=self.agents_config['historian'],
        tools=get_historian_tools(),
        llm=self.llm_config.get_llm(),
        output_pydantic=EnrichedQuery,  # Schema-guided output
        max_retry_limit=2
    )
```

### Task Configuration

```python
@task
def context_gathering_task(self) -> Task:
    return create_schema_guided_task(
        description="""
        Gather and enrich context for research query: {topic}.
        Search memory, expand terms, develop strategy.
        """,
        expected_output="Structured enriched query with context",
        agent=self.historian(),
        schema_model=EnrichedQuery,
        validation_level="medium",  # Recommended
        max_retries=2
    )
```

## Validation Levels

### 1. Flexible (Recommended for Learning)
- Provides helpful guidance on errors
- Extracts partial data when possible
- Suggests improvements without blocking
- Best for agent learning and iteration

### 2. Medium (Recommended for Production)
- Validates strictly but provides clear error messages
- Allows retries with improved guidance
- Good balance of structure and flexibility
- Suitable for most use cases

### 3. Strict (Use Sparingly)
- Blocks on any validation error
- Minimal guidance provided
- Use only when data integrity is critical
- Can hinder AI learning and creativity

## Benefits

### ✅ Structure Without Rigidity
- Agents know what's expected but aren't overly constrained
- Natural language guidance helps AI learn the schema
- Multiple validation attempts allow iteration and improvement

### ✅ Automatic Type Safety
- Validated Pydantic models in downstream processing
- Consistent data structures across your workflow
- Reduced debugging time from malformed data

### ✅ Enhanced AI Performance  
- Clear expectations improve output quality
- Schema descriptions guide content generation
- Error feedback helps agents learn correct formats

### ✅ Workflow Integration
- Seamless integration with existing CrewAI patterns
- Uses CrewAI's built-in `output_pydantic` feature
- Compatible with task chaining and context passing

## Example Workflows

### Research Pipeline

1. **Historian** → `EnrichedQuery`
   - Searches memory for related content
   - Expands search terms
   - Develops search strategy

2. **Researcher** → `RawPaperData`  
   - Uses enriched context for targeted search
   - Extracts comprehensive paper data
   - Maintains extraction quality metrics

3. **Archivist** → `ResearchPaperSchema`
   - Analyzes and structures raw data
   - Creates academic paper format
   - Generates quality assessments

4. **Publisher** → `ObsidianDocument`
   - Transforms to Obsidian format
   - Creates knowledge graph connections
   - Writes to vault with proper metadata

### Error Handling Example

When an agent produces incorrect output:

```
Schema validation guidance for EnrichedQuery:
  • Field 'search_strategy': Field required
  • Field 'expanded_terms': Input should be a valid list

Required fields: original_query, expanded_terms, search_strategy

Please ensure your JSON output includes all required fields.
```

This guidance helps the AI understand what's missing and how to fix it.

## Publisher Dual MCP Integration

The publisher agent now has access to both:

### Filesystem MCP
- File operations (read, write, create, delete)
- Directory management
- Path: `C:\0_repos\mcp\Obsidian`

### Obsidian MCP Tools  
- Note creation and management
- Knowledge graph operations
- Vault-specific functionality

This dual integration allows the publisher to:
- Write files directly to the Obsidian vault
- Create proper knowledge graph connections
- Maintain vault metadata and organization
- Generate cross-references and backlinks

## Best Practices

### 1. Start with Flexible Validation
Begin with flexible validation to allow agents to learn, then gradually increase strictness as performance improves.

### 2. Use Descriptive Field Names
Your schemas already do this well - clear field names help AI understand expectations.

### 3. Provide Examples in Descriptions
Include example values in field descriptions when helpful.

### 4. Chain Validation for Complex Workflows
Use schema validation at each step to catch issues early in multi-step processes.

### 5. Monitor and Iterate
Use validation reports to identify common failure patterns and improve schemas.

## Customization

### Adding New Schemas

1. Create Pydantic model in appropriate schema file
2. Add to `SCHEMA_REGISTRY` in `schema_guidance.py`
3. Configure agent with `output_pydantic=YourSchema`
4. Use in task creation with appropriate validation level

### Custom Validation Logic

```python
def custom_guardrail(result: TaskOutput) -> Tuple[bool, Any]:
    # Your custom validation logic
    # Return (success, validated_data_or_error_message)
    pass
```

## Testing and Validation

Run the demonstration script to see the system in action:

```bash
python demo_schema_integration.py
```

This demonstrates:
- Schema validation with various input scenarios
- Error guidance examples
- Prompt enhancement techniques
- Integration patterns with CrewAI

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure schema files are properly imported
2. **Validation Failures**: Start with flexible validation to debug
3. **Performance Issues**: Use medium validation for best balance
4. **Agent Confusion**: Enhance prompts with more specific schema guidance

### Debug Tools

- Use `validate_output_against_schema()` to test outputs manually
- Check validation reports for common failure patterns
- Review enhanced prompts to ensure clarity
- Test with different validation levels to find optimal settings

## Conclusion

Schema-guided output provides the perfect balance between structure and flexibility for your CrewAI research workflow. By using your existing well-designed schemas with the validation utilities, you can ensure consistent, high-quality output while allowing agents to learn and improve over time.

The key is to guide rather than constrain - provide clear expectations and helpful feedback that leads to better AI performance and more reliable results. 