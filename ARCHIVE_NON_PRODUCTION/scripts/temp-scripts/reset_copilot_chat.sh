#!/bin/bash
# VS Code Copilot Chat Reset Script

echo "🚨 Resetting VS Code Copilot Chat..."

# Close VS Code
echo "1. Close VS Code completely"
osascript -e 'quit app "Visual Studio Code"'

sleep 3

# Clear all Copilot chat data
echo "2. Clearing Copilot chat storage..."
rm -rf ~/Library/Application\ Support/Code/User/workspaceStorage/*/GitHub.copilot-chat
rm -rf ~/Library/Application\ Support/Code/User/globalStorage/github.copilot-chat

# Clear recent workspace storage
rm -rf ~/Library/Application\ Support/Code/User/workspaceStorage/*/vscode.chat

echo "3. Restart VS Code manually"
echo "✅ Copilot chat should be reset!"