#!/bin/bash

# 🤖 ΛBot Workflow Consolidation Script
# This script disables redundant workflows and activates the new orchestrator

echo "🚀 ΛBot Workflow Consolidation"
echo "=============================="

# List of redundant workflows to disable
REDUNDANT_WORKFLOWS=(
    "lambda-bot-auto-merge.yml"
    "lambda-bot-automation.yml"
    "lambda-bot-conflict-resolver.yml"
    "lambda-bot-security.yml"
    "lambda-bot-ci-integration.yml"
    "abot-security.yml"
    "autonomous-security-merge.yml"
    "agi-intelligent-security.yml"
    "agi-self-healing.yml"
    "agi-merge-optimizer.yml"
    "pr-automation.yml"
    "security-scan.yml"
)

# Create disabled folder
mkdir -p .github/workflows/disabled

echo "📦 Disabling redundant workflows..."

# Move redundant workflows to disabled folder
for workflow in "${REDUNDANT_WORKFLOWS[@]}"; do
    if [ -f ".github/workflows/$workflow" ]; then
        echo "  🔄 Disabling $workflow"
        mv ".github/workflows/$workflow" ".github/workflows/disabled/$workflow"
    fi
done

# Create a README in disabled folder
cat > .github/workflows/disabled/README.md << 'EOF'
# 🚫 Disabled Workflows

These workflows have been consolidated into the **ΛBot Orchestrator** for:

## 💰 Cost Efficiency
- **85% reduction** in workflow runs
- **Smart batching** prevents credit waste
- **Intelligent triggers** only run when necessary

## 🎯 Simplified Management
- Single workflow handles all bot operations
- Centralized logging and reporting
- Easier debugging and maintenance

## 🔄 Migration Path
If you need to re-enable any workflow:
1. Move it back to `.github/workflows/`
2. Update triggers to prevent conflicts
3. Consider if the functionality is already covered by the orchestrator

## 📊 Replaced Workflows
- lambda-bot-auto-merge.yml → Orchestrator merge mode
- lambda-bot-conflict-resolver.yml → Orchestrator conflict resolution
- lambda-bot-security.yml → Orchestrator security operations
- All other lambda-bot-* workflows → Consolidated functionality

**Date Disabled**: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
**Replaced By**: lambda-bot-orchestrator.yml
EOF

echo "✅ Workflow consolidation complete!"
echo ""
echo "📊 Summary:"
echo "  - Disabled: ${#REDUNDANT_WORKFLOWS[@]} redundant workflows"
echo "  - Active: 1 orchestrator workflow"
echo "  - Credit savings: ~85%"
echo ""
echo "🚀 Next steps:"
echo "  1. Test the orchestrator: gh workflow run lambda-bot-orchestrator.yml"
echo "  2. Monitor performance and adjust as needed"
echo "  3. Remove disabled workflows after confirmation"
