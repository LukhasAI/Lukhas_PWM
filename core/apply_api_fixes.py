#!/usr/bin/env python3
"""
Automated API Fix Application Script
Generated by API Diff Analyzer
"""

import re
import json
from pathlib import Path


def apply_fixes():
    """Apply all API fixes to test files"""
    # Load fixes
    with open('api_fixes.json', 'r') as f:
        fixes_data = json.load(f)

    print(f"🔧 Applying {len(fixes_data['fixes'])} fixes...")

    # Group fixes by file
    fixes_by_file = {}
    for fix in fixes_data['fixes']:
        if fix['confidence'] > 0.7:  # Only apply high-confidence fixes
            file_path = fix['test_file']
            if file_path not in fixes_by_file:
                fixes_by_file[file_path] = []
            fixes_by_file[file_path].append(fix)

    # Apply fixes to each file
    for file_path, file_fixes in fixes_by_file.items():
        path = Path(file_path)
        if path.exists():
            print(f"📝 Fixing {path.name}...")

            content = path.read_text()
            original_content = content

            for fix in file_fixes:
                old_pattern = rf"{fix['old_method']}"
                new_method = fix['new_method']

                # Count replacements
                count = len(re.findall(old_pattern, content))
                if count > 0:
                    content = re.sub(old_pattern, new_method, content)
                    print(f"  ✅ Replaced {count} instances of {fix['old_method']} with {new_method}")

            # Write back if changed
            if content != original_content:
                # Backup original
                backup_path = path.with_suffix('.bak')
                path.rename(backup_path)

                # Write fixed content
                path.write_text(content)
                print(f"  💾 Saved fixed file (backup: {backup_path.name})")

    print("✅ All fixes applied!")


if __name__ == "__main__":
    apply_fixes()
