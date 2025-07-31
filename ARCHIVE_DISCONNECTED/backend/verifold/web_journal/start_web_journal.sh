#!/bin/bash
# VeriFold Web Journal Setup and Launch Script
# =============================================

echo "🌐 VeriFold Web Journal Setup & Launch"
echo "======================================"

# Check if we're in the right directory
if [ ! -f "web_journal_requirements.txt" ]; then
    echo "❌ Please run this script from the web_journal/ directory"
    exit 1
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python
if ! command_exists python3; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Check pip
if ! command_exists pip; then
    echo "❌ pip is required but not installed"
    exit 1
fi

# Install dependencies
echo "📦 Installing web journal dependencies..."
pip install -r web_journal_requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Check for OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  OpenAI API key not set - GPT features will be limited"
    echo "   Set with: export OPENAI_API_KEY=your-key-here"
else
    echo "✅ OpenAI API key found"
fi

# Check for parent journal files
if [ ! -f "../journal_mode.py" ]; then
    echo "⚠️  journal_mode.py not found in parent directory"
    echo "   Some features may not work correctly"
fi

if [ ! -f "../verifold_logbook.jsonl" ]; then
    echo "⚠️  verifold_logbook.jsonl not found in parent directory"
    echo "   Using sample data for demonstration"
fi

echo ""
echo "🚀 Starting VeriFold Web Journal..."
echo "📱 Open your browser to: http://localhost:5001"
echo "🎯 Press Ctrl+C to stop the server"
echo ""

# Launch the web journal
python3 launch_web_journal.py "$@"
