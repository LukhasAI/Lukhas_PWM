# 📋 Full List of Unused Files

**Total Unused Files:** 1,157  
**Report Location:** `/analysis-tools/unused_files_report.json`

## How to Access the Full List

### 1. **View the JSON file directly:**
```bash
cat analysis-tools/unused_files_report.json
```

### 2. **Pretty print with jq:**
```bash
jq '.unused_files[] | .path' analysis-tools/unused_files_report.json
```

### 3. **Extract just the file paths:**
```bash
jq -r '.unused_files[].path' analysis-tools/unused_files_report.json > unused_files_list.txt
```

### 4. **View with size information:**
```bash
jq -r '.unused_files[] | "\(.size_human)\t\(.path)"' analysis-tools/unused_files_report.json
```

### 5. **Filter by directory:**
```bash
# Show unused files in core directory
jq -r '.unused_files[] | select(.directory | startswith("core")) | .path' analysis-tools/unused_files_report.json

# Show unused files in DAST
jq -r '.unused_files[] | select(.path | contains("dast")) | .path' analysis-tools/unused_files_report.json
```

### 6. **Sort by size (largest first):**
```bash
jq -r '.unused_files | sort_by(.size_bytes) | reverse[] | "\(.size_human)\t\(.path)"' analysis-tools/unused_files_report.json
```

## File Structure

The `unused_files_report.json` contains:
```json
{
  "total_files": 2342,
  "unused_files": [
    {
      "path": "path/to/file.py",
      "size_bytes": 12345,
      "size_human": "12.1 KB",
      "directory": "path/to"
    },
    ...
  ]
}
```

## Quick Summary Script

Create a summary of unused files by directory:
```bash
jq -r '.unused_files[].directory' analysis-tools/unused_files_report.json | sort | uniq -c | sort -nr
```

---

**Note:** The unused_files_report.json file is 273KB and contains all 1,157 unused files with their paths, sizes, and directory information.