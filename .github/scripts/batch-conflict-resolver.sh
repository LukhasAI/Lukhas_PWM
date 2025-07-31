#!/bin/bash
# ΛBot Batch Conflict Resolution Script
# ΛTAG: conflict-resolution, lambda-bot

echo "🤖 ΛBot Batch Conflict Resolution"
echo "================================="

# Get all PRs with conflicts
echo "📋 Finding PRs with conflicts..."
CONFLICTED_PRS=$(gh pr list --state open --json number,mergeable --jq '.[] | select(.mergeable == "CONFLICTING") | .number')

if [ -z "$CONFLICTED_PRS" ]; then
    echo "✅ No PRs with conflicts found!"
    exit 0
fi

echo "⚠️ Found conflicted PRs:"
echo "$CONFLICTED_PRS"

# Ask for confirmation
echo ""
read -p "🤖 Resolve conflicts for these PRs? (y/N): " -r
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Operation cancelled"
    exit 1
fi

echo ""
echo "🔄 Starting batch conflict resolution..."

# Process each PR
SUCCESS_COUNT=0
TOTAL_COUNT=0

for PR_NUMBER in $CONFLICTED_PRS; do
    TOTAL_COUNT=$((TOTAL_COUNT + 1))
    echo ""
    echo "📋 Processing PR #$PR_NUMBER..."

    # Get PR details
    PR_DATA=$(gh pr view $PR_NUMBER --json headRefName,baseRefName,title,mergeable 2>/dev/null)

    if [ $? -ne 0 ]; then
        echo "❌ Failed to get PR details for #$PR_NUMBER"
        continue
    fi

    HEAD_REF=$(echo "$PR_DATA" | jq -r '.headRefName')
    BASE_REF=$(echo "$PR_DATA" | jq -r '.baseRefName')
    PR_TITLE=$(echo "$PR_DATA" | jq -r '.title')
    MERGEABLE=$(echo "$PR_DATA" | jq -r '.mergeable')

    echo "  📋 Title: $PR_TITLE"
    echo "  🔀 $HEAD_REF → $BASE_REF"
    echo "  📊 Status: $MERGEABLE"

    # Skip if not actually conflicting
    if [ "$MERGEABLE" != "CONFLICTING" ]; then
        echo "  ✅ PR is not conflicting, skipping"
        continue
    fi

    # Fetch the branches
    echo "  📥 Fetching branches..."
    git fetch origin $HEAD_REF:$HEAD_REF 2>/dev/null || true
    git fetch origin $BASE_REF:$BASE_REF 2>/dev/null || true

    # Switch to head branch
    git checkout $HEAD_REF 2>/dev/null

    if [ $? -ne 0 ]; then
        echo "  ❌ Failed to checkout $HEAD_REF"
        continue
    fi

    # Try to merge and detect conflicts
    echo "  🔍 Checking for conflicts..."
    if git merge origin/$BASE_REF --no-commit --no-ff 2>/dev/null; then
        echo "  ✅ No conflicts detected (may have been resolved)"
        git merge --abort 2>/dev/null || true
        continue
    fi

    # Get conflicted files
    CONFLICTED_FILES=$(git diff --name-only --diff-filter=U)

    if [ -z "$CONFLICTED_FILES" ]; then
        echo "  ❌ No conflicted files found"
        git merge --abort 2>/dev/null || true
        continue
    fi

    echo "  📋 Conflicted files:"
    echo "$CONFLICTED_FILES" | sed 's/^/    /'

    # Auto-resolve conflicts
    echo "  🔧 Resolving conflicts..."

    # Use the Python conflict resolution script
    python3 .github/scripts/lambda-bot-conflict-resolver.py $CONFLICTED_FILES

    if [ $? -eq 0 ]; then
        echo "  ✅ Conflicts resolved successfully"

        # Commit the resolved conflicts
        git commit -m "🤖 ΛBot: Auto-resolved merge conflicts

Automatically resolved conflicts using intelligent merging:
- Security files: Kept HEAD (newer fixes)
- Documentation: Intelligent merge with symbolic tag preservation
- Python files: Import deduplication and symbolic header preservation
- Config files: Kept HEAD (safer)

Files resolved: $(echo '$CONFLICTED_FILES' | tr '\n' ' ')

ΛTAG: auto-conflict-resolution, lambda-bot
ΛORIGIN_AGENT: ΛBot-BatchConflictResolver"

        # Push resolved changes
        git push origin $HEAD_REF

        if [ $? -eq 0 ]; then
            echo "  🚀 Changes pushed successfully"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))

            # Update PR status
            gh pr edit $PR_NUMBER --remove-label "merge-conflict,needs-manual-review" --add-label "auto-resolved,lambda-bot-processed" 2>/dev/null || true

            # Add comment
            gh pr comment $PR_NUMBER --body "🤖 **ΛBot Batch Conflict Resolution** ✅

Conflicts have been automatically resolved using intelligent merging strategies.

**Files Resolved:**
\`\`\`
$CONFLICTED_FILES
\`\`\`

The PR is now ready for review and should be mergeable.

*Powered by ΛBot Batch Conflict Resolution*" 2>/dev/null || true
        else
            echo "  ❌ Failed to push changes"
        fi
    else
        echo "  ❌ Failed to resolve conflicts"
        git merge --abort 2>/dev/null || true
    fi
done

echo ""
echo "🎯 Batch Resolution Summary:"
echo "  📊 Total PRs: $TOTAL_COUNT"
echo "  ✅ Resolved: $SUCCESS_COUNT"
echo "  ❌ Failed: $((TOTAL_COUNT - SUCCESS_COUNT))"

if [ $SUCCESS_COUNT -gt 0 ]; then
    echo ""
    echo "🚀 Successfully resolved conflicts in $SUCCESS_COUNT PRs!"
    echo "📋 Check the PRs for review and merge when ready."
fi
