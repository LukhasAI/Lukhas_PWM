#!/bin/bash
# Run connectivity analysis across all major LUKHAS modules

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_PATH="$REPO_ROOT/scripts/connectivity/generate_connectivity_index_enhanced.py"

echo "🚀 Starting LUKHAS Connectivity Analysis"
echo "Repository root: $REPO_ROOT"
echo "="*60

# Core modules to analyze
MODULES=(
    "lukhas/core"
    "lukhas/consciousness"
    "lukhas/memory"
    "lukhas/reasoning"
    "lukhas/creativity"
    "lukhas/emotion"
    "lukhas/ethics"
    "lukhas/identity"
    "lukhas/learning"
    "lukhas/bridge"
    "lukhas/orchestration"
    "lukhas/quantum"
    "lukhas/bio"
    "lukhas/symbolic"
)

# Track success/failure
SUCCESS_COUNT=0
FAILED_COUNT=0
FAILED_MODULES=()

# Run analysis for each module
for module in "${MODULES[@]}"; do
    echo ""
    echo "📁 Analyzing: $module"
    echo "-"*40
    
    if [ -d "$REPO_ROOT/$module" ]; then
        if python3 "$SCRIPT_PATH" "$REPO_ROOT/$module" --repo-root "$REPO_ROOT"; then
            echo "✅ Success: $module"
            ((SUCCESS_COUNT++))
        else
            echo "❌ Failed: $module"
            ((FAILED_COUNT++))
            FAILED_MODULES+=("$module")
        fi
    else
        echo "⚠️  Directory not found: $module"
        ((FAILED_COUNT++))
        FAILED_MODULES+=("$module")
    fi
done

# Generate master report
echo ""
echo "="*60
echo "📊 Generating Master Connectivity Report..."

python3 - <<EOF
import json
import os
from pathlib import Path
from datetime import datetime

repo_root = "$REPO_ROOT"
modules = ${MODULES[@]}

master_report = {
    "timestamp": datetime.now().isoformat(),
    "modules_analyzed": $SUCCESS_COUNT,
    "modules_failed": $FAILED_COUNT,
    "total_missed_opportunities": 0,
    "critical_issues": [],
    "module_summaries": {}
}

# Collect all module reports
for module in [$(printf '"%s",' "${MODULES[@]}")]:
    module = module.rstrip(',')
    index_path = Path(repo_root) / module / "CONNECTIVITY_INDEX.json"
    if index_path.exists():
        with open(index_path) as f:
            data = json.load(f)
            master_report["module_summaries"][module] = {
                "total_symbols": data["summary"]["total_symbols"],
                "missed_opportunities": data["summary"]["missed_opportunities"],
                "top_issues": [mo["type"] for mo in data["missed_opportunities"][:3]]
            }
            master_report["total_missed_opportunities"] += data["summary"]["missed_opportunities"]
            
            # Collect critical issues
            for mo in data["missed_opportunities"]:
                if mo["severity"] == "high":
                    master_report["critical_issues"].append({
                        "module": module,
                        "type": mo["type"],
                        "description": mo["description"]
                    })

# Write master report
output_path = Path(repo_root) / "MASTER_CONNECTIVITY_REPORT.json"
with open(output_path, 'w') as f:
    json.dump(master_report, f, indent=2)

# Write master markdown
md_path = Path(repo_root) / "MASTER_CONNECTIVITY_REPORT.md"
with open(md_path, 'w') as f:
    f.write("# LUKHAS Master Connectivity Report\n\n")
    f.write(f"Generated: {master_report['timestamp']}\n\n")
    f.write("## Summary\n\n")
    f.write(f"- **Modules Analyzed:** {master_report['modules_analyzed']}\n")
    f.write(f"- **Total Missed Opportunities:** {master_report['total_missed_opportunities']}\n")
    f.write(f"- **Critical Issues:** {len(master_report['critical_issues'])}\n\n")
    
    if master_report['critical_issues']:
        f.write("## 🚨 Critical Issues\n\n")
        for issue in master_report['critical_issues'][:10]:  # Top 10
            f.write(f"- **{issue['module']}**: {issue['description']}\n")
    
    f.write("\n## Module Summaries\n\n")
    for module, summary in master_report['module_summaries'].items():
        f.write(f"### {module}\n")
        f.write(f"- Symbols: {summary['total_symbols']}\n")
        f.write(f"- Issues: {summary['missed_opportunities']}\n")
        if summary['top_issues']:
            f.write(f"- Top Issues: {', '.join(summary['top_issues'])}\n")
        f.write("\n")

print(f"✅ Master report generated at {output_path}")
EOF

# Final summary
echo ""
echo "="*60
echo "🎉 Connectivity Analysis Complete!"
echo ""
echo "📊 Results:"
echo "  - Successful: $SUCCESS_COUNT modules"
echo "  - Failed: $FAILED_COUNT modules"

if [ ${#FAILED_MODULES[@]} -gt 0 ]; then
    echo ""
    echo "❌ Failed modules:"
    for module in "${FAILED_MODULES[@]}"; do
        echo "  - $module"
    done
fi

echo ""
echo "📁 Reports generated in each module directory"
echo "📋 Master report: $REPO_ROOT/MASTER_CONNECTIVITY_REPORT.md"