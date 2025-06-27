# Research Paper Parser - CrewAI + MCP Integration

A sophisticated academic research tool that extracts, structures, and formats research papers from Zotero into Obsidian-compatible markdown documents using a four-agent CrewAI system with MCP tool integration.

## 🌟 Features

- **Four-Agent Sequential Pipeline**: Historian → Researcher → Archivist → Publisher
- **MCP Tool Integration**: Memory management, sequential thinking, Zotero access, and Obsidian integration
- **Rich Metadata Extraction**: Authors, citations, keywords, and more
- **Knowledge Graph Building**: Automatic wiki-links and connections between papers
- **Intelligent Summarization**: AI-powered section summaries preserving key insights
- **Schema Validation**: Ensures data quality and consistency
- **Obsidian-Ready Output**: YAML frontmatter, tags, and wiki-links

## 📋 Prerequisites

- Python 3.10+
- Zotero account with API access (optional for testing)
- Obsidian vault
- Anthropic or OpenAI API key

## 🚀 Quick Start

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd server-research-mcp
   ```

2. **Install dependencies**:
   ```bash
   pip install -e .
   ```

3. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and paths
   ```

4. **Run the parser**:
   ```bash
   python main.py "machine learning paper title" --topic "AI"
   ```

## 🏗️ Architecture

### Four-Agent System

1. **Historian Agent** 🧠
   - Manages memory and context
   - Retrieves historical knowledge about related papers
   - Implements bidirectional thinking
   - Creates enriched queries for downstream agents

2. **Researcher Agent** 🔍
   - Interfaces with Zotero to locate papers
   - Extracts full content (PDF, metadata, annotations)
   - Performs structure analysis
   - Handles various paper formats

3. **Archivist Agent** 📚
   - Transforms raw data into validated JSON schema
   - Generates intelligent summaries
   - Ensures Obsidian compatibility
   - Validates data completeness

4. **Publisher Agent** 📝
   - Converts JSON to Obsidian markdown
   - Generates YAML frontmatter
   - Creates wiki-links and knowledge connections
   - Saves to Obsidian vault

### MCP Tools

Each agent has access to specialized MCP tools:

- **Memory Tools**: Knowledge graph operations
- **Sequential Thinking**: Multi-step reasoning
- **Zotero Integration**: Paper search and extraction
- **Obsidian Tools**: Note creation and linking
- **File System**: Reading/writing operations

## 📁 Project Structure

```
server-research-mcp/
├── src/
│   └── server_research_mcp/
│       ├── agents/           # Agent implementations
│       ├── config/           # YAML configurations
│       ├── schemas/          # Pydantic data models
│       ├── tools/            # MCP tool implementations
│       ├── crew.py           # Main crew orchestration
│       └── main.py           # CLI entry point
├── tests/                    # Test suite
├── .env.example              # Environment template
└── README.md                 # This file
```

## ⚙️ Configuration

### Environment Variables

Key environment variables in `.env`:

```bash
# LLM Configuration
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your_key_here

# Zotero Configuration
ZOTERO_API_KEY=your_key_here
ZOTERO_LIBRARY_ID=your_library_id

# Obsidian Configuration
OBSIDIAN_VAULT_PATH=/path/to/vault
```

### Agent Configuration

Agents are configured in `src/server_research_mcp/config/agents.yaml`

### Task Configuration

Tasks are defined in `src/server_research_mcp/config/tasks.yaml`

## 🔧 Usage

### Basic Usage

```bash
# Parse a paper by title
python main.py "Attention is All You Need"

# Parse by DOI
python main.py "10.1234/example.doi" --topic "transformers"

# Parse by Zotero key
python main.py "ABC123XYZ" --year 2023
```

### Advanced Options

```bash
# Dry run (no files created)
python main.py "paper title" --dry-run

# Specify output directory
python main.py "paper title" --output-dir ./results

# Verbose output
python main.py "paper title" --verbose
```

### Python API

```python
from server_research_mcp.crew import ServerResearchMcp

# Initialize crew
crew = ServerResearchMcp()

# Prepare inputs
inputs = {
    "paper_query": "Your paper search query",
    "topic": "research topic",
    "current_year": 2024
}

# Run the crew
result = crew.crew().kickoff(inputs=inputs)
```

## 📊 Output

The system generates multiple outputs:

1. **Enriched Query** (`outputs/enriched_query.json`): Context and search strategy
2. **Raw Paper Data** (`outputs/raw_paper_data.json`): Extracted content
3. **Structured Paper** (`outputs/structured_paper.json`): Validated schema
4. **Obsidian Note** (`vault/Papers/paper-title.md`): Final markdown

### Example Obsidian Output

```markdown
---
title: Attention Is All You Need
authors:
  - Ashish Vaswani
  - Noam Shazeer
  - Niki Parmar
year: 2017
journal: NeurIPS
tags:
  - transformer
  - attention-mechanism
  - deep-learning
related_papers:
  - [[BERT- Pre-training of Deep Bidirectional Transformers]]
  - [[GPT-3- Language Models are Few-Shot Learners]]
---

# Attention Is All You Need

## Summary

This groundbreaking paper introduces the Transformer architecture...

## Key Findings

- Self-attention mechanisms can replace recurrent layers
- Parallel computation significantly improves training speed
- ...
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_research_paper_parser.py -v

# Run with coverage
pytest --cov=server_research_mcp
```

## 🔍 Troubleshooting

### Common Issues

1. **LLM API Key Error**: Ensure your API key is set in `.env`
2. **Zotero Connection**: Check API key and library ID
3. **Obsidian Path**: Use absolute path to vault
4. **Memory Errors**: Ensure MCP servers are accessible

### Debug Mode

Enable debug logging:

```bash
export DEBUG=true
python main.py "paper title" --verbose
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

[Your License Here]

## 🙏 Acknowledgments

- CrewAI framework for agent orchestration
- MCP (Model Context Protocol) for tool integration
- Zotero for reference management
- Obsidian for knowledge management
