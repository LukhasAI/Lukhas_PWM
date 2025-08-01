#!/usr/bin/env python3
"""
Extract and enumerate all git-tracked README files into a single flat directory.
"""

import os
import shutil
import subprocess
from pathlib import Path


def extract_enumerated_readmes():
    """Extract all README files with enumerated names in flat structure."""

    # Create output directory
    output_dir = Path("lukhas_readmes_enumerated")
    output_dir.mkdir(exist_ok=True)

    # Get all git-tracked README files
    result = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, cwd="."
    )

    if result.returncode != 0:
        print(f"Git command failed: {result.stderr}")
        return

    # Filter for README files and sort
    readme_files = []
    for line in result.stdout.strip().split("\n"):
        if line and "readme.md" in line.lower():
            readme_files.append(line)

    readme_files.sort()

    print(f"Found {len(readme_files)} README files")

    # Copy files with enumerated names
    for i, file_path in enumerate(readme_files, 1):
        source_path = Path(file_path)
        if source_path.exists():
            # Create descriptive enumerated name
            path_parts = str(source_path).replace("/", "_").replace(".", "_")
            enumerated_name = f"{i:03d}_{path_parts}"

            dest_path = output_dir / enumerated_name
            shutil.copy2(source_path, dest_path)
            print(f"Copied: {file_path} -> {enumerated_name}")

    # Create index file
    index_path = output_dir / "000_INDEX.txt"
    with open(index_path, "w") as f:
        f.write("LUKHAS README Files - Enumerated Collection\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total files: {len(readme_files)}\n")
        f.write(f"Generated: {os.popen('date').read().strip()}\n\n")

        for i, file_path in enumerate(readme_files, 1):
            path_parts = str(Path(file_path)).replace("/", "_").replace(".", "_")
            enumerated_name = f"{i:03d}_{path_parts}"
            f.write(f"{enumerated_name} <- {file_path}\n")

    print(f"\nCompleted! All files in: {output_dir.absolute()}")
    print(f"Index file created: {index_path}")


if __name__ == "__main__":
    extract_enumerated_readmes()
