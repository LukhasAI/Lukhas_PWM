#!/bin/bash
# LUKHAS VeriFold Scanner Startup Script

echo "🧠 LUKHAS VeriFold Scanner System"
echo "=================================="
echo ""

# Check if in correct directory
if [ ! -f "scanner_api.py" ]; then
    echo "❌ Error: Please run this script from the scanner_pwa directory"
    exit 1
fi

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is required but not installed"
    exit 1
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    python3 -m pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "⚠️  Warning: Could not install some dependencies. Continuing anyway..."
    fi
fi

echo ""
echo "🚀 Starting LUKHAS VeriFold Scanner..."
echo ""
echo "📱 PWA will be available at: http://localhost:5000"
echo "🔗 API endpoints at: http://localhost:5000/api/"
echo ""
echo "🛑 Press Ctrl+C to stop the server"
echo ""

# Start the Flask API server
python3 scanner_api.py
