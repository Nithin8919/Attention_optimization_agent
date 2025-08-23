#!/bin/bash

# Attention Optimization AI - Startup Script
echo "🚀 Starting Attention Optimization AI..."

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Please run setup first."
    echo "   Run: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source .venv/bin/activate

# Check if required packages are installed
if ! python -c "import fastapi, uvicorn, openai, playwright" 2>/dev/null; then
    echo "❌ Required packages not installed. Installing now..."
    pip install -r requirements.txt
    echo "🔧 Installing Playwright browsers..."
    playwright install
fi

# Check if OpenAI API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  Warning: OPENAI_API_KEY environment variable not set."
    echo "   The app will work but you won't be able to analyze images with AI."
    echo "   Set it with: export OPENAI_API_KEY=your_key_here"
    echo ""
fi

# Start the application
echo "🌟 Starting server on http://localhost:8080"
echo "   Press Ctrl+C to stop"
echo ""
uvicorn main:app --reload --port 8080
