#!/bin/bash
# Simple launcher for the Telugu STT App

echo "=========================================="
echo "  Telugu Speech-to-Text App"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Installing dependencies (this may take a few minutes)..."
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

echo ""
echo "Starting the web app..."
echo "Open the URL shown below in your browser."
echo ""

python app.py
