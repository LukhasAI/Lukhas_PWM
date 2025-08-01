#!/bin/bash

# Root Directory Organization Script
# Organizes cluttered root directory into logical structure

set -e

echo "🗂️ Starting Root Directory Organization..."
echo "📁 Working directory: $(pwd)"

# Create new directory structure
echo "📁 Creating directory structure..."
mkdir -p docs/{guides,reports,planning,archives}
mkdir -p scripts/{analysis,organization,setup,testing}
mkdir -p data/{analysis,archives,temp}

echo "✅ Directory structure created"

# Phase 1: Move Documentation Files
echo "📝 Moving documentation files..."

# Move specialized docs
if [ -f "AGENT.md" ]; then
    git mv AGENT.md docs/guides/ && echo "  ✅ Moved AGENT.md"
fi

if [ -f "WORKSPACE_INDEX.md" ]; then
    git mv WORKSPACE_INDEX.md docs/guides/ && echo "  ✅ Moved WORKSPACE_INDEX.md"
fi

if [ -f "CHANGELOG.md" ]; then
    git mv CHANGELOG.md docs/archives/ && echo "  ✅ Moved CHANGELOG.md"
fi

# Move reports
if [ -f "REPORT_REORGANIZATION_PLAN.md" ]; then
    git mv REPORT_REORGANIZATION_PLAN.md docs/reports/ && echo "  ✅ Moved REPORT_REORGANIZATION_PLAN.md"
fi

if [ -f "TASK_STATUS_AND_ROOT_ORGANIZATION.md" ]; then
    git mv TASK_STATUS_AND_ROOT_ORGANIZATION.md docs/reports/ && echo "  ✅ Moved TASK_STATUS_AND_ROOT_ORGANIZATION.md"
fi

if [ -f ".secrets_scan_report.md" ]; then
    git mv .secrets_scan_report.md docs/reports/ && echo "  ✅ Moved .secrets_scan_report.md"
fi

# Phase 2: Move Python Scripts
echo "🐍 Moving Python scripts..."

if [ -f "analyze_core_overlap.py" ]; then
    git mv analyze_core_overlap.py scripts/analysis/ && echo "  ✅ Moved analyze_core_overlap.py"
fi

if [ -f "test_reorganized_modules.py" ]; then
    git mv test_reorganized_modules.py scripts/testing/ && echo "  ✅ Moved test_reorganized_modules.py"
fi

# Phase 3: Move Data Files
echo "📊 Moving data files..."

if [ -f "core_analysis_data.json" ]; then
    git mv core_analysis_data.json data/analysis/ && echo "  ✅ Moved core_analysis_data.json"
fi

if [ -f "github_todo_issues.json" ]; then
    git mv github_todo_issues.json data/analysis/ && echo "  ✅ Moved github_todo_issues.json"
fi

if [ -f "github_todo_issues_comprehensive.json" ]; then
    git mv github_todo_issues_comprehensive.json data/analysis/ && echo "  ✅ Moved github_todo_issues_comprehensive.json"
fi

if [ -f "requirements.txt.backup" ]; then
    git mv requirements.txt.backup data/archives/ && echo "  ✅ Moved requirements.txt.backup"
fi

# Move archive files
if [ -f "LUKHAS_AGI_Consolidation_Clean_20250723_054620.zip" ]; then
    mv "LUKHAS_AGI_Consolidation_Clean_20250723_054620.zip" data/archives/ && echo "  ✅ Moved clean archive"
fi

# Phase 4: Move Shell Scripts
echo "🔧 Moving shell scripts..."

if [ -f "reorganize_remaining_reports.sh" ]; then
    git mv reorganize_remaining_reports.sh scripts/organization/ && echo "  ✅ Moved reorganize_remaining_reports.sh"
fi

if [ -f "reorganize_reports.sh" ]; then
    git mv reorganize_reports.sh scripts/organization/ && echo "  ✅ Moved reorganize_reports.sh"
fi

if [ -f "setup_security_hooks.sh" ]; then
    git mv setup_security_hooks.sh scripts/setup/ && echo "  ✅ Moved setup_security_hooks.sh"
fi

# Phase 5: Handle Database
echo "🗄️ Moving database files..."

if [ -f "lukhas_agi.db" ]; then
    # Ensure lukhas_db directory exists
    mkdir -p lukhas_db/production/
    mv lukhas_agi.db lukhas_db/production/ && echo "  ✅ Moved lukhas_agi.db to lukhas_db/production/"
fi

# Create index files for new directories
echo "📋 Creating directory index files..."

cat > docs/README.md << 'EOF'
# 📚 Documentation

This directory contains all project documentation organized by category.

## Structure

- `guides/` - User and developer guides
- `reports/` - Analysis and status reports  
- `planning/` - Planning and strategy documents
- `archives/` - Historical documentation

## Quick Access

- [Agent Guide](guides/AGENT.md) - Agent system documentation
- [Workspace Guide](guides/WORKSPACE_INDEX.md) - Workspace navigation
- [Reports](reports/) - Current analysis reports
EOF

cat > scripts/README.md << 'EOF'
# 🔧 Scripts

This directory contains all utility scripts organized by function.

## Structure

- `analysis/` - Analysis and data processing utilities
- `organization/` - File and project organization scripts
- `setup/` - Setup and installation scripts
- `testing/` - Testing utilities and validators

## Usage

All scripts should be run from the project root directory.
EOF

cat > data/README.md << 'EOF'
# 📊 Data

This directory contains all data files and analysis results.

## Structure

- `analysis/` - Analysis results and processed data
- `archives/` - Historical data and backups
- `temp/` - Temporary data files (not committed)

## Note

Temporary files in `temp/` are excluded from version control.
EOF

# Add temp directory to .gitignore
if ! grep -q "data/temp/" .gitignore 2>/dev/null; then
    echo "data/temp/" >> .gitignore
    echo "  ✅ Added data/temp/ to .gitignore"
fi

# Display final root structure
echo ""
echo "🎉 Root directory organization complete!"
echo ""
echo "📁 New root structure:"
ls -la | grep "^d" | grep -v "^drw.*\.$" | sort

echo ""
echo "📄 Remaining root files:"
find . -maxdepth 1 -type f -not -path "./.git*" -not -name ".DS_Store" | sort

echo ""
echo "✨ Organization summary:"
echo "  📁 Created: docs/, scripts/, data/ directories"
echo "  📝 Moved: Documentation files to docs/"
echo "  🐍 Moved: Python scripts to scripts/"
echo "  📊 Moved: Data files to data/"
echo "  🔧 Moved: Shell scripts to scripts/"
echo "  🗄️ Moved: Database to lukhas_db/production/"
echo ""
echo "🎯 Root directory is now organized and professional!"
