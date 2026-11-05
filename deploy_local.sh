#!/usr/bin/env bash
set -e

echo "==================================================="
echo "  YouTube Lecture Synthesizer - Launching App...   "
echo "==================================================="

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Checking and installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Starting Streamlit Web Application..."
streamlit run app.py
