#!/bin/bash
# Emergency VS Code Terminal Reset Script
# Use this when terminal freezes with dquote> prompt

echo "🚨 EMERGENCY TERMINAL RESET"
echo "=========================="

# Kill any stuck processes
echo "1. Killing stuck Python processes..."
pkill -f python 2>/dev/null || true
pkill -f pylance 2>/dev/null || true

# Reset terminal
echo "2. Resetting terminal..."
reset 2>/dev/null || true

# Clear any jobs
echo "3. Clearing background jobs..."
jobs -l 2>/dev/null || true

# Clear command history if needed
echo "4. Clearing command buffer..."
history -c 2>/dev/null || true

echo "✅ Terminal reset complete!"
echo ""
echo "💡 If VS Code is still frozen:"
echo "   1. Press Cmd+Shift+P"
echo "   2. Type: 'Developer: Reload Window'"
echo "   3. Or restart VS Code completely"
echo ""
echo "🔧 To prevent future freezing:"
echo "   • Avoid very long commands in terminal"
echo "   • Use Ctrl+C to interrupt stuck commands"
echo "   • Close Problems panel if it has too many items"
