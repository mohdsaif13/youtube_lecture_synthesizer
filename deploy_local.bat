@echo off
title YouTube Lecture Synthesizer Launcher
echo ===================================================
echo   YouTube Lecture Synthesizer - Launching App...
echo ===================================================

cd /d "%~dp0"

if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Checking and installing dependencies...
pip install -r requirements.txt

echo Starting Streamlit Web Application...
streamlit run app.py

pause
