@echo off
echo ==============================================
echo   AI Video Summarizer - Grader Boot Script
echo ==============================================
echo.

echo [1/3] Checking environment...
if not exist venv (
    echo Creating Python Virtual Environment...
    python -m venv venv
)

echo [2/3] Installing Dependencies (This might take a moment if first time)...
call venv\Scripts\activate
pip install -r requirements.txt --quiet

echo [3/3] Booting Application Server...
echo Starting High-Performance AI Server on Port 8000...
start cmd /k "venv\Scripts\uvicorn src.api.main:app --host 0.0.0.0 --port 8000"

echo.
echo ==============================================
echo SUCCESS! 
echo Please wait 15 seconds for the AI models to load.
echo Then go to: http://localhost:8000 in your browser.
echo ==============================================
pause
