#!/usr/bin/env python3
"""Simple script to check Zotero credentials and environment setup."""

import os
import subprocess
import sys

def check_environment():
    """Check environment variables."""
    print("🔍 ZOTERO ENVIRONMENT CHECK")
    print("=" * 40)
    
    # Check environment variables
    api_key = os.getenv("ZOTERO_API_KEY")
    library_id = os.getenv("ZOTERO_LIBRARY_ID")
    
    print(f"ZOTERO_API_KEY: {'SET' if api_key else 'MISSING'}")
    if api_key:
        print(f"  Length: {len(api_key)} characters")
        print(f"  Format: {'Valid' if len(api_key) >= 20 and api_key.isalnum() else 'Invalid'}")
    
    print(f"ZOTERO_LIBRARY_ID: {'SET' if library_id else 'MISSING'}")
    if library_id:
        print(f"  Value: {library_id}")
        print(f"  Format: {'Valid' if library_id.isdigit() else 'Invalid'}")
    
    print()
    
    # Check uvx availability
    print("🔧 UVX AVAILABILITY CHECK")
    print("=" * 30)
    try:
        result = subprocess.run(["uvx", "--version"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ uvx available: {result.stdout.strip()}")
        else:
            print(f"❌ uvx error: {result.stderr.strip()}")
    except FileNotFoundError:
        print("❌ uvx not found - install uv package manager")
    except Exception as e:
        print(f"❌ uvx check failed: {e}")
    
    print()
    
    # Check zotero-mcp package
    print("📦 ZOTERO-MCP PACKAGE CHECK")
    print("=" * 35)
    try:
        result = subprocess.run(["uvx", "zotero-mcp", "--help"], capture_output=True, text=True, timeout=15)
        if result.returncode == 0 or "zotero" in result.stderr.lower():
            print("✅ zotero-mcp package available")
        else:
            print("⚠️ zotero-mcp package may need installation")
        print(f"Output preview: {(result.stdout or result.stderr)[:200]}...")
    except Exception as e:
        print(f"❌ zotero-mcp check failed: {e}")
    
    print()
    
    # Recommendations
    print("💡 RECOMMENDATIONS")
    print("=" * 20)
    
    if not api_key:
        print("• Set ZOTERO_API_KEY environment variable")
    elif not (len(api_key) >= 20 and api_key.isalnum()):
        print("• Check ZOTERO_API_KEY format (should be 40-character alphanumeric)")
    
    if not library_id:
        print("• Set ZOTERO_LIBRARY_ID environment variable")
    elif not library_id.isdigit():
        print("• Check ZOTERO_LIBRARY_ID format (should be numeric)")
    
    if not api_key or not library_id:
        print("• Visit https://www.zotero.org/settings/keys to create API credentials")
        print("• Get your library ID from the URL when viewing your library")

if __name__ == "__main__":
    check_environment() 