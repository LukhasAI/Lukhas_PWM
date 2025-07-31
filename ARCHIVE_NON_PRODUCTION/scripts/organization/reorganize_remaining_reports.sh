#!/bin/bash

# ═══════════════════════════════════════════════════════════════════════════
# LUKHAS AGI - Reorganize Remaining Reports
# 
# Purpose: Move additional report files that were missed in the first pass
# Date: 2025-07-23
# ═══════════════════════════════════════════════════════════════════════════

set -e  # Exit on error

echo "🔄 Moving remaining report files..."
echo "=================================="

# Move additional security reports to audit/security/
echo ""
echo "🔐 Moving additional security reports..."
if [ -f "CREDENTIAL_ROTATION_GUIDE.md" ]; then
    mv -v CREDENTIAL_ROTATION_GUIDE.md audit/security/
fi
if [ -f "SECURITY_REMEDIATION_COMPLETE.md" ]; then
    mv -v SECURITY_REMEDIATION_COMPLETE.md audit/security/
fi
if [ -f "SECURITY_REMEDIATION_SUMMARY.md" ]; then
    mv -v SECURITY_REMEDIATION_SUMMARY.md audit/security/
fi
if [ -f "SECURITY_VERIFICATION_REPORT_FINAL.md" ]; then
    mv -v SECURITY_VERIFICATION_REPORT_FINAL.md audit/security/
fi

# Move additional audit reports to audit/reports/
echo ""
echo "📋 Moving additional audit reports..."
if [ -f "TODO_CLASSIFICATION_REPORT.md" ]; then
    mv -v TODO_CLASSIFICATION_REPORT.md audit/reports/
fi
if [ -f "DEPENDENCY_UPGRADE_REPORT.md" ]; then
    mv -v DEPENDENCY_UPGRADE_REPORT.md audit/reports/
fi

# Move analysis reports to audit/analysis/
echo ""
echo "📊 Moving analysis reports..."
if [ -f "core_analysis_report.md" ]; then
    mv -v core_analysis_report.md audit/analysis/
fi
if [ -f "core_consolidation_map.md" ]; then
    mv -v core_consolidation_map.md audit/analysis/
fi

# Move documentation to appropriate places
echo ""
echo "📚 Moving documentation files..."
if [ -f "DOCUMENTATION_UPDATE_SUMMARY.md" ]; then
    mv -v DOCUMENTATION_UPDATE_SUMMARY.md audit/cleanup/
fi

# Create docs/guides if it doesn't exist and move guides there
echo ""
echo "📖 Organizing guide files..."
mkdir -p docs/guides
if [ -f "USAGE_GUIDE.md" ]; then
    mv -v USAGE_GUIDE.md docs/guides/
fi

# Files that should stay in root
echo ""
echo "✅ The following files correctly remain in root:"
echo "- README.md (standard project file)"
echo "- CHANGELOG.md (standard project file)"
echo "- CONTRIBUTING.md (standard project file)"
echo "- LICENSE (if exists)"
echo "- CLAUDE.md (agent documentation)"
echo "- AGENT.md (agent documentation)"
echo "- WORKSPACE_INDEX.md (workspace navigation)"
echo "- TASK_STATUS_AND_ROOT_ORGANIZATION.md (current task tracking)"
echo "- REPORT_REORGANIZATION_PLAN.md (this reorganization documentation)"

echo ""
echo "📊 Summary of additional moves:"
echo "- 4 more security reports → audit/security/"
echo "- 2 more audit reports → audit/reports/"
echo "- 2 analysis documents → audit/analysis/"
echo "- 1 documentation summary → audit/cleanup/"
echo "- 1 usage guide → docs/guides/"

echo ""
echo "🎯 Root directory is now properly organized with only essential files!"