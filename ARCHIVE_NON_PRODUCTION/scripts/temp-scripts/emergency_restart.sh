#!/bin/bash

echo "🚨 EMERGENCY VS CODE RESTART 🚨"
echo "======================================"

echo "1. Killing all VS Code processes..."
pkill -f "Visual Studio Code"
pkill -f "Code Helper"
pkill -f "code"

echo "2. Waiting 3 seconds..."
sleep 3

echo "3. Clearing VS Code cache..."
rm -rf ~/Library/Caches/com.microsoft.VSCode*
rm -rf ~/Library/Application\ Support/Code/CachedExtensions*

echo "4. Restarting VS Code with minimal settings..."
code /Users/agi_dev/Downloads/Consolidation-Repo

echo "✅ VS Code restart complete!"
echo "🔧 Emergency minimal settings are now active"
echo "💡 Check if performance has improved"
