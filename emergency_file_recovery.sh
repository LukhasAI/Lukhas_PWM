#!/bin/bash

# 🚨 LUKHAS AGI Emergency File Recovery Script
# This script helps recover content from emptied .md files
# Created: July 31, 2025

echo "🚨 LUKHAS AGI Emergency File Recovery"
echo "======================================"

# Set colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Recovery status
echo -e "${BLUE}📊 Current Recovery Status:${NC}"
echo -e "${GREEN}✅ AGENT.md - RECOVERED (183 lines)${NC}"
echo -e "${GREEN}✅ DOCUMENTATION_INDEX.md - RECOVERED (169 lines)${NC}"
echo -e "${GREEN}✅ AGENT1_TASK7_COMPLETION_REPORT.md - RECOVERED (107 lines)${NC}"
echo -e "${GREEN}✅ AGENT1_TASK8_COMPLETION_REPORT.md - RECOVERED (136 lines)${NC}"
echo -e "${GREEN}✅ AGENT1_TASK9_COMPLETION_REPORT.md - RECOVERED (138 lines)${NC}"
echo -e "${GREEN}✅ INTEGRATION_PLAN_STATUS.md - RECOVERED (145 lines)${NC}"
echo -e "${GREEN}✅ GOLDEN_TRIO.md - RECREATED (65 lines)${NC}"
echo -e "${GREEN}✅ AGENT_QUICK_START.md - RECREATED (120 lines)${NC}"
echo ""

# Check what still needs recovery
echo -e "${YELLOW}🔍 Checking remaining empty files...${NC}"

EMPTY_FILES=()
for file in *.md; do
    if [ -f "$file" ] && [ ! -s "$file" ]; then
        EMPTY_FILES+=("$file")
    fi
done

echo -e "${RED}📋 Files still needing recovery:${NC}"
for file in "${EMPTY_FILES[@]}"; do
    echo -e "${RED}❌ $file${NC}"
done

echo ""
echo -e "${BLUE}🔧 Recovery Options:${NC}"
echo "1. Check project-docs/temp-cleanup/ for backups"
echo "2. Use git history: git show HEAD~N:filename"
echo "3. Check *.backup files in subdirectories"
echo "4. Recreate from scratch using template"

echo ""
echo -e "${YELLOW}💡 Next Steps:${NC}"
echo "1. Run: find project-docs/temp-cleanup/ -name '*.md' -size +0c"
echo "2. Copy content files: cp project-docs/temp-cleanup/[FILE] [FILE]"
echo "3. For missing files, check git history or recreate"
echo "4. Commit recovered files: git add . && git commit -m 'Emergency recovery of documentation files'"

echo ""
echo -e "${GREEN}🎯 Priority Recovery Order:${NC}"
echo "1. GOLDEN_TRIO*.md files (architecture documents)"
echo "2. PHASE_*.md files (status reports)"
echo "3. Integration and implementation guides"
echo "4. Test and validation reports"

echo ""
echo -e "${BLUE}📝 Manual Recovery Commands:${NC}"
echo "# Copy from temp-cleanup if available:"
echo "cp project-docs/temp-cleanup/GOLDEN_TRIO.md GOLDEN_TRIO.md"
echo "cp project-docs/temp-cleanup/PHASE_*.md ."
echo ""
echo "# Check git history for content:"
echo "git log --oneline --follow GOLDEN_TRIO.md"
echo "git show HEAD~3:GOLDEN_TRIO.md > GOLDEN_TRIO.md"
