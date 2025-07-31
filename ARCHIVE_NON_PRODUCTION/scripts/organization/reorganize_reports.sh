#!/bin/bash

# ═══════════════════════════════════════════════════════════════════════════
# LUKHAS AGI Root Directory Reorganization Script
# 
# Purpose: Move report and audit files from root to organized subdirectories
# Date: 2025-07-23
# 
# This script will:
# 1. Create necessary directory structure
# 2. Move audit reports to audit/reports/
# 3. Move security reports to audit/security/
# 4. Move Claude reports to agents/claude/reports/
# 5. Move analysis docs to audit/analysis/
# 6. Move cleanup reports to audit/cleanup/
# 7. Create index files for easy navigation
# ═══════════════════════════════════════════════════════════════════════════

set -e  # Exit on error

echo "🔄 Starting LUKHAS AGI Root Directory Reorganization..."
echo "=================================================="

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p audit/reports
mkdir -p audit/security
mkdir -p audit/analysis
mkdir -p audit/cleanup
mkdir -p agents/claude/reports

# Move general audit reports to audit/reports/
echo ""
echo "📋 Moving general audit reports to audit/reports/..."
if [ -f "LINT_HYGIENE_AUDIT_REPORT.md" ]; then
    mv -v LINT_HYGIENE_AUDIT_REPORT.md audit/reports/
fi
if [ -f "COMPREHENSIVE_LINT_HYGIENE_AUDIT.md" ]; then
    mv -v COMPREHENSIVE_LINT_HYGIENE_AUDIT.md audit/reports/
fi
if [ -f "PHASE1_AUDIT_CONSOLIDATED_REPORT.md" ]; then
    mv -v PHASE1_AUDIT_CONSOLIDATED_REPORT.md audit/reports/
fi
if [ -f "TODO_CLASSIFICATION_REPORT_COMPREHENSIVE.md" ]; then
    mv -v TODO_CLASSIFICATION_REPORT_COMPREHENSIVE.md audit/reports/
fi

# Move security reports to audit/security/
echo ""
echo "🔐 Moving security reports to audit/security/..."
if [ -f "COMMAND_INJECTION_VULNERABILITY_REPORT.md" ]; then
    mv -v COMMAND_INJECTION_VULNERABILITY_REPORT.md audit/security/
fi
if [ -f "CRITICAL_SECURITY_VULNERABILITY_REPORT.md" ]; then
    mv -v CRITICAL_SECURITY_VULNERABILITY_REPORT.md audit/security/
fi
if [ -f "HARDCODED_SECRETS_SCAN_REPORT.md" ]; then
    mv -v HARDCODED_SECRETS_SCAN_REPORT.md audit/security/
fi
if [ -f "SECURITY_REMEDIATION_COMPREHENSIVE_REPORT.md" ]; then
    mv -v SECURITY_REMEDIATION_COMPREHENSIVE_REPORT.md audit/security/
fi
if [ -f "SECURITY_REMEDIATION_FINAL_REPORT.md" ]; then
    mv -v SECURITY_REMEDIATION_FINAL_REPORT.md audit/security/
fi
if [ -f "SECURITY_REMEDIATION_REPORT.md" ]; then
    mv -v SECURITY_REMEDIATION_REPORT.md audit/security/
fi
if [ -f "SECURITY_REVERIFICATION_REPORT.md" ]; then
    mv -v SECURITY_REVERIFICATION_REPORT.md audit/security/
fi
if [ -f "SECURITY_VULNERABILITY_SCAN_REPORT.md" ]; then
    mv -v SECURITY_VULNERABILITY_SCAN_REPORT.md audit/security/
fi
if [ -f "SECURE_ENV_SETUP_GUIDE.md" ]; then
    mv -v SECURE_ENV_SETUP_GUIDE.md audit/security/
fi
if [ -f "SECURITY_FIXES_SUMMARY.md" ]; then
    mv -v SECURITY_FIXES_SUMMARY.md audit/security/
fi

# Move Claude-specific reports to agents/claude/reports/
echo ""
echo "🤖 Moving Claude reports to agents/claude/reports/..."
if [ -f "CLAUDE_02_PRINT_AUDIT_RECONCILIATION.md" ]; then
    mv -v CLAUDE_02_PRINT_AUDIT_RECONCILIATION.md agents/claude/reports/
fi
if [ -f "CLAUDE_LINT_CLEANUP_PROMPTS.md" ]; then
    mv -v CLAUDE_LINT_CLEANUP_PROMPTS.md agents/claude/reports/
fi

# Move analysis and planning docs to audit/analysis/
echo ""
echo "📊 Moving analysis documents to audit/analysis/..."
if [ -f "MODULARIZATION_PLAN_BY_CLAUDE.md" ]; then
    mv -v MODULARIZATION_PLAN_BY_CLAUDE.md audit/analysis/
fi
if [ -f "CORE_TO_ROOT_CONSOLIDATION_PLAN.md" ]; then
    mv -v CORE_TO_ROOT_CONSOLIDATION_PLAN.md audit/analysis/
fi

# Move cleanup reports to audit/cleanup/
echo ""
echo "🧹 Moving cleanup reports to audit/cleanup/..."
if [ -f "MEMORY_FOLD_CLEANUP_REPORT.md" ]; then
    mv -v MEMORY_FOLD_CLEANUP_REPORT.md audit/cleanup/
fi
if [ -f "PRINT_TO_LOGGING_REFACTOR_REPORT.md" ]; then
    mv -v PRINT_TO_LOGGING_REFACTOR_REPORT.md audit/cleanup/
fi
if [ -f "REORGANIZATION_COMPLETE_REPORT.md" ]; then
    mv -v REORGANIZATION_COMPLETE_REPORT.md audit/cleanup/
fi
if [ -f "WORKSPACE_CLEANUP_REPORT.md" ]; then
    mv -v WORKSPACE_CLEANUP_REPORT.md audit/cleanup/
fi

# Create index files for each directory
echo ""
echo "📝 Creating index files..."

# Create audit/reports/INDEX.md
cat > audit/reports/INDEX.md << 'EOF'
# 📋 Audit Reports Index

This directory contains general audit reports for the LUKHAS AGI system.

## Files:
- `LINT_HYGIENE_AUDIT_REPORT.md` - Initial code hygiene audit
- `COMPREHENSIVE_LINT_HYGIENE_AUDIT.md` - Complete system-wide hygiene audit
- `PHASE1_AUDIT_CONSOLIDATED_REPORT.md` - Consolidated Phase 1 audit findings
- `TODO_CLASSIFICATION_REPORT_COMPREHENSIVE.md` - Complete TODO analysis (785 items)

## Navigation:
- [Security Reports](../security/INDEX.md)
- [Analysis Documents](../analysis/INDEX.md)
- [Cleanup Reports](../cleanup/INDEX.md)
EOF

# Create audit/security/INDEX.md
cat > audit/security/INDEX.md << 'EOF'
# 🔐 Security Reports Index

This directory contains all security-related reports and remediation documentation.

## Chronological Order:
1. `HARDCODED_SECRETS_SCAN_REPORT.md` - Initial security scan
2. `SECURITY_VULNERABILITY_SCAN_REPORT.md` - Comprehensive vulnerability assessment
3. `COMMAND_INJECTION_VULNERABILITY_REPORT.md` - Specific injection vulnerabilities
4. `CRITICAL_SECURITY_VULNERABILITY_REPORT.md` - Critical issues requiring immediate attention
5. `SECURITY_REMEDIATION_REPORT.md` - First remediation attempt
6. `SECURITY_REMEDIATION_COMPREHENSIVE_REPORT.md` - Comprehensive remediation plan
7. `SECURITY_REMEDIATION_FINAL_REPORT.md` - Final remediation status
8. `SECURITY_REVERIFICATION_REPORT.md` - Post-remediation verification
9. `SECURITY_FIXES_SUMMARY.md` - Summary of all security fixes applied
10. `SECURE_ENV_SETUP_GUIDE.md` - Guide for secure environment configuration

## Key Findings:
- Initial scan found 916 potential security issues
- After excluding virtual environments: 33 actual vulnerabilities
- 4 critical issues fixed immediately
- Comprehensive .env.example created for secure configuration
EOF

# Create audit/analysis/INDEX.md
cat > audit/analysis/INDEX.md << 'EOF'
# 📊 Analysis Documents Index

This directory contains architectural analysis and planning documents.

## Files:
- `MODULARIZATION_PLAN_BY_CLAUDE.md` - Comprehensive plan for core/ directory modularization
- `CORE_TO_ROOT_CONSOLIDATION_PLAN.md` - Plan for consolidating core/ modules into root directories

## Key Insights:
- Discovered 11+ hidden gems including Bio-Quantum AGI paradigm
- Identified 383 files in core/ requiring reorganization
- Proposed consolidation of 250 modules to appropriate domain directories
EOF

# Create audit/cleanup/INDEX.md
cat > audit/cleanup/INDEX.md << 'EOF'
# 🧹 Cleanup Reports Index

This directory contains reports from various cleanup and refactoring efforts.

## Files:
- `MEMORY_FOLD_CLEANUP_REPORT.md` - Memory fold system cleanup
- `PRINT_TO_LOGGING_REFACTOR_REPORT.md` - Conversion of print statements to logging
- `WORKSPACE_CLEANUP_REPORT.md` - General workspace organization cleanup
- `REORGANIZATION_COMPLETE_REPORT.md` - Final reorganization status

## Achievements:
- Converted 52 print statements to proper logging
- Fixed 95% of empty except blocks
- Achieved 100% star import removal
- Organized workspace structure
EOF

# Create agents/claude/reports/INDEX.md
cat > agents/claude/reports/INDEX.md << 'EOF'
# 🤖 Claude Agent Reports Index

This directory contains reports specific to Claude agent activities.

## Files:
- `CLAUDE_02_PRINT_AUDIT_RECONCILIATION.md` - Print statement audit by CLAUDE_02
- `CLAUDE_LINT_CLEANUP_PROMPTS.md` - Lint cleanup task assignments

## Related Documents:
- [Phase 1 Audit Status](../CLAUDE_PHASE1_AUDIT_STATUS.md)
- [Claude Prompts](../claude_prompts.md)
- [Task Status](../../../TASK_STATUS_AND_ROOT_ORGANIZATION.md)
EOF

echo ""
echo "✅ Reorganization complete!"
echo ""
echo "📊 Summary:"
echo "- Moved general audit reports to: audit/reports/"
echo "- Moved security reports to: audit/security/"
echo "- Moved Claude reports to: agents/claude/reports/"
echo "- Moved analysis docs to: audit/analysis/"
echo "- Moved cleanup reports to: audit/cleanup/"
echo "- Created INDEX.md files in each directory"
echo ""
echo "📝 Note: The following files remain in root as they are standard project files:"
echo "- README.md, LICENSE, CHANGELOG.md"
echo "- CLAUDE.md, AGENT.md, WORKSPACE_INDEX.md"
echo "- Configuration files (.gitignore, .env.example, requirements.txt)"
echo ""
echo "🎯 Next steps:"
echo "1. Update any scripts that reference the moved files"
echo "2. Update WORKSPACE_INDEX.md with the new structure"
echo "3. Commit these changes with message: 'chore: reorganize audit and report files'"