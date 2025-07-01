# Paper Template for Research Documentation

This template provides the structure for documenting research papers in Obsidian, following the established schema.

## Frontmatter Template
```yaml
---
title: "{{paper_title}}"
authors:
  - {{author_name_1}}
  - {{author_name_2}}
year: {{publication_year}}
doi: {{doi}}
arxiv_id: {{arxiv_id}}  # if applicable
journal: {{journal_name}}
tags:
  - "#{{research_domain}}"
  - "#{{methodology}}"
  - "#{{application_area}}"
aliases:
  - {{short_name}}
  - {{acronym}}
cited_dois:
  - {{referenced_doi_1}}
  - {{referenced_doi_2}}
---
```

## Document Structure Template

```markdown
# {{paper_title}}

**Authors**: {{formatted_author_list}}
**Published**: {{publication_date}} in {{venue}}
**DOI**: [[{{doi}}]]
**arXiv**: {{arxiv_id}} (if applicable)

## Abstract
{{abstract_content}}

## Key Contributions
- {{contribution_1}}
- {{contribution_2}}
- {{contribution_3}}

## Content

### {{section_number}}. {{section_title}}
{{section_content}}

#### {{subsection_number}} {{subsection_title}}
{{subsection_content}}

**Figure {{figure_number}}**: {{figure_caption}}
{{figure_description}}

**Table {{table_number}}**: {{table_caption}}
| {{header_1}} | {{header_2}} | {{header_3}} |
|--------------|--------------|--------------|
| {{data_1_1}} | {{data_1_2}} | {{data_1_3}} |
| {{data_2_1}} | {{data_2_2}} | {{data_2_3}} |

**Equation {{eq_number}}**: 
$${{latex_equation}}$$
{{equation_description}}

### {{next_section_number}}. {{next_section_title}}
{{next_section_content}}

## References
- [[{{cited_doi_1}}]] {{reference_1_full_citation}}
- [[{{cited_doi_2}}]] {{reference_2_full_citation}}

## Graph Connections
- **Cites**: [[{{cited_doi_1}}]], [[{{cited_doi_2}}]], [[{{cited_doi_3}}]]
- **Related Papers**: 
  - [[{{related_paper_1_title}}|{{related_paper_1_alias}}]]
  - [[{{related_paper_2_title}}|{{related_paper_2_alias}}]]
```

## Schema Requirements

### Required Fields:
1. **metadata**:
   - title (string)
   - authors (array of objects with name, affiliation)
   - publication_date (YYYY-MM-DD format)
   - doi (format: 10.xxxx/xxxxx)
   - paper_type (journal_article, conference_paper, preprint, technical_report, thesis)

2. **content**:
   - abstract (string)
   - sections (array with section_number, title, content)
   - key_contributions (array of strings)

3. **references**:
   - cited_papers (array with DOIs, titles, authors, year)
   - citation_count (integer)

4. **obsidian_config**:
   - filename (without .md extension)
   - folder_path (e.g., "01-Documentation/Research")
   - tags (array starting with #)
   - aliases (array of alternative names)

### Optional Fields:
- arxiv_id (format: xxxx.xxxxx)
- journal, volume, issue, pages
- keywords (array)
- funding information
- figures, tables, equations with captions
- methodology details
- results and performance metrics
- limitations and future work

## DOI-based Clustering
- Use DOI as primary identifier for graph connections
- Extract all referenced DOIs for citation network
- Create bidirectional links between related papers
- Use wikilinks format: [[DOI]] or [[Paper Title|Alias]]

## Example Usage
See: [[A Multi-Agent Framework for Extensible Structured Text Generation in PLCs]] for a complete implementation following this template. 