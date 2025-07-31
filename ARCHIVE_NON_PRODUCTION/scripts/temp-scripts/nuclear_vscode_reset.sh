#!/bin/bash
# Ultimate VS Code Copilot Reset - NUCLEAR OPTION

echo "🚨 ULTIMATE NUCLEAR RESET - This will completely reset VS Code..."

# Step 1: Force quit ALL VS Code processes
echo "1. Killing ALL VS Code processes..."
pkill -f "Visual Studio Code"
pkill -f "Code Helper"
pkill -f "Electron"
sleep 5

# Step 2: Clear ALL VS Code data
echo "2. Clearing ALL VS Code data..."
rm -rf ~/Library/Application\ Support/Code/User/workspaceStorage/*/
rm -rf ~/Library/Application\ Support/Code/User/globalStorage/
rm -rf ~/Library/Application\ Support/Code/logs/
rm -rf ~/Library/Application\ Support/Code/CachedData/
rm -rf ~/Library/Caches/com.microsoft.VSCode/

# Step 3: Reset Copilot completely
echo "3. Resetting Copilot completely..."
rm -rf ~/.vscode/extensions/github.copilot*
rm -rf ~/.vscode/extensions/GitHub.copilot*

# Step 4: Clear workspace-specific settings
echo "4. Clearing workspace settings..."
find ~/Downloads -name ".vscode" -type d -exec rm -rf {} + 2>/dev/null || true

echo "🎯 NUCLEAR RESET COMPLETE!"
echo "Now restart VS Code manually - it should be like a fresh install"
echo "You'll need to reinstall/re-enable extensions manually"
