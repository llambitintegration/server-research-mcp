#!/usr/bin/env python3
"""Utility to migrate emoji usage to cross-platform system."""

import re
from pathlib import Path
from typing import Dict, List, Tuple

# Import the symbol mapping from logging_config
from .logging_config import SYMBOLS, get_symbol


# Extended mapping for migrating existing emojis to symbol names
EMOJI_TO_SYMBOL = {
    '✅': 'success',
    '❌': 'error', 
    '⚠️': 'warning',
    '🚀': 'rocket',
    '💥': 'boom',
    '🎉': 'party',
    '👋': 'wave',
    'ℹ️': 'info',
    '🔧': 'debug',
    '🧪': 'test',
    '📝': 'docs',
    '🔍': 'search',
    '📊': 'stats',
    '📋': 'task',
    '🎯': 'target',
    '🔄': 'cycle',
    '⭐': 'star',
    '📱': 'mobile',
    '💻': 'computer',
    '🌟': 'star2',
    '🚧': 'construction',
    '🎨': 'art',
    '⚖️': 'balance',
    '🛡️': 'shield',
    '🏷️': 'tag',
    '💡': 'bulb',
    
    # Additional emojis found in codebase
    '🐛': 'debug',  # bug -> debug
    '📱': 'mobile',
    '💻': 'computer',
    '🌍': 'info',   # world -> info
    '💩': 'boom',   # poop -> boom (error)
}


def find_emoji_usage(file_path: Path) -> List[Tuple[int, str, List[str]]]:
    """Find all emoji usage in a file.
    
    Returns list of (line_number, line_content, emojis_found)
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
    except Exception:
        return []
    
    results = []
    for i, line in enumerate(lines, 1):
        emojis_found = []
        for emoji in EMOJI_TO_SYMBOL.keys():
            if emoji in line:
                emojis_found.append(emoji)
        
        if emojis_found:
            results.append((i, line.rstrip(), emojis_found))
    
    return results


def suggest_migration(line: str) -> str:
    """Suggest how to migrate a line with emojis."""
    migrated = line
    
    # Look for print statements and f-strings
    for emoji, symbol_name in EMOJI_TO_SYMBOL.items():
        if emoji in migrated:
            # For print statements, suggest using get_symbol
            if 'print(' in migrated and 'f"' in migrated:
                migrated = migrated.replace(emoji, f'{{get_symbol("{symbol_name}")}}')
            elif 'print(' in migrated:
                migrated = migrated.replace(f'"{emoji}', f'"{{get_symbol("{symbol_name}")}}')
                migrated = migrated.replace(f"'{emoji}", f"'{{get_symbol('{symbol_name}')}}")
            else:
                # For other contexts, just replace with the ASCII version
                migrated = migrated.replace(emoji, SYMBOLS[symbol_name])
    
    return migrated


def generate_migration_report(root_path: Path = None) -> Dict:
    """Generate a report of all emoji usage in the codebase."""
    if root_path is None:
        root_path = Path.cwd()
    
    # File patterns to check
    patterns = ['*.py', '*.md', '*.txt', '*.yaml', '*.yml']
    
    report = {
        'total_files_checked': 0,
        'files_with_emojis': 0,
        'total_emoji_instances': 0,
        'emoji_counts': {},
        'files': {}
    }
    
    for pattern in patterns:
        for file_path in root_path.rglob(pattern):
            # Skip certain directories
            if any(skip in str(file_path) for skip in ['.git', '__pycache__', '.venv', 'node_modules']):
                continue
                
            report['total_files_checked'] += 1
            
            emoji_usage = find_emoji_usage(file_path)
            if emoji_usage:
                report['files_with_emojis'] += 1
                relative_path = file_path.relative_to(root_path)
                
                file_info = {
                    'path': str(relative_path),
                    'emoji_lines': []
                }
                
                for line_num, line_content, emojis in emoji_usage:
                    file_info['emoji_lines'].append({
                        'line': line_num,
                        'content': line_content,
                        'emojis': emojis,
                        'suggested': suggest_migration(line_content)
                    })
                    
                    report['total_emoji_instances'] += len(emojis)
                    
                    for emoji in emojis:
                        report['emoji_counts'][emoji] = report['emoji_counts'].get(emoji, 0) + 1
                
                report['files'][str(relative_path)] = file_info
    
    return report


def print_migration_report(report: Dict):
    """Print a formatted migration report."""
    from .logging_config import get_symbol
    
    print(f"\n{get_symbol('search')} Emoji Migration Analysis")
    print("=" * 50)
    
    print(f"\n{get_symbol('stats')} Summary:")
    print(f"  Files checked: {report['total_files_checked']}")
    print(f"  Files with emojis: {report['files_with_emojis']}")
    print(f"  Total emoji instances: {report['total_emoji_instances']}")
    
    if report['emoji_counts']:
        print(f"\n{get_symbol('task')} Emoji frequency:")
        for emoji, count in sorted(report['emoji_counts'].items(), key=lambda x: x[1], reverse=True):
            symbol_name = EMOJI_TO_SYMBOL.get(emoji, 'unknown')
            replacement = SYMBOLS.get(symbol_name, f'[{symbol_name.upper()}]')
            print(f"  {emoji} -> {replacement} ({count} instances)")
    
    print(f"\n{get_symbol('docs')} Files requiring migration:")
    for file_path, file_info in report['files'].items():
        print(f"\n  {get_symbol('docs')} {file_path}:")
        for line_info in file_info['emoji_lines'][:3]:  # Show first 3 lines
            print(f"    Line {line_info['line']}: {line_info['emojis']}")
            print(f"      Original: {line_info['content'][:80]}...")
            print(f"      Suggested: {line_info['suggested'][:80]}...")
        
        if len(file_info['emoji_lines']) > 3:
            print(f"    ... and {len(file_info['emoji_lines']) - 3} more lines")


def create_migration_script(report: Dict, output_file: str = "emoji_migration.py"):
    """Create a script to automatically apply migrations."""
    script_content = f'''#!/usr/bin/env python3
"""Auto-generated script to migrate emojis to cross-platform system."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from server_research_mcp.utils.logging_config import get_symbol

def apply_migrations():
    """Apply all emoji migrations."""
    print(f"{{get_symbol('rocket')}} Starting emoji migration...")
    
    migrations = {{
'''
    
    for file_path, file_info in report['files'].items():
        if file_info['emoji_lines']:
            script_content += f'''
        "{file_path}": [
'''
            for line_info in file_info['emoji_lines']:
                original = line_info['content'].replace('\\', '\\\\').replace('"', '\\"')
                suggested = line_info['suggested'].replace('\\', '\\\\').replace('"', '\\"')
                script_content += f'''            ({line_info['line']}, "{original}", "{suggested}"),
'''
            script_content += '''        ],
'''
    
    script_content += '''    }
    
    files_updated = 0
    for file_path, changes in migrations.items():
        if update_file(file_path, changes):
            files_updated += 1
    
    print(f"{get_symbol('success')} Migration completed: {files_updated} files updated")

def update_file(file_path: str, changes: list) -> bool:
    """Update a single file with the given changes."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Apply changes in reverse order to maintain line numbers
        for line_num, original, suggested in reversed(changes):
            if line_num <= len(lines) and lines[line_num - 1].rstrip() == original:
                lines[line_num - 1] = suggested + '\\n'
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        print(f"  {get_symbol('success')} Updated {file_path}")
        return True
        
    except Exception as e:
        print(f"  {get_symbol('error')} Failed to update {file_path}: {e}")
        return False

if __name__ == "__main__":
    apply_migrations()
'''
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print(f"\n{get_symbol('success')} Migration script created: {output_file}")


if __name__ == "__main__":
    print(f"{get_symbol('rocket')} Starting emoji migration analysis...")
    
    report = generate_migration_report()
    print_migration_report(report)
    
    if report['files_with_emojis'] > 0:
        create_migration_script(report)
        print(f"\n{get_symbol('info')} To apply migrations, run: python emoji_migration.py")
    else:
        print(f"\n{get_symbol('success')} No emoji migrations needed!") 