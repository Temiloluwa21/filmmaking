@echo off
setlocal enabledelayedexpansion
echo ==============================================
echo   AI Video Summarizer - Stable Boot Script
echo ==============================================
echo.

echo [1/3] Checking environment...

:: --- Better Python Detection ---
set PYTHON_CMD=python
python --version >nul 2>&1
if errorlevel 1 (
    set PYTHON_CMD=py
    py --version >nul 2>&1
    if errorlevel 1 (
        set PYTHON_CMD=python3
        python3 --version >nul 2>&1
        if errorlevel 1 (
            echo [ERROR] Python not found.
            echo Please install Python 3.10+ from python.org
            pause
            exit /b
        )
    )
)

echo Found Python command: !PYTHON_CMD!

if not exist venv (
    echo Creating Python Virtual Environment...
    !PYTHON_CMD! -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create venv. 
        echo Check your Python installation.
        pause
        exit /b
    )
)

echo [2/3] Verifying Dependencies...
call venv\Scripts\activate
if errorlevel 1 (
    echo [ERROR] Failed to activate venv at venv\Scripts\activate
    pause
    exit /b
)

python -m pip install -r requirements.txt --quiet
python -m pip install moviepy==2.2.1 --quiet

echo [3/3] Booting Application Server...
echo Starting High-Performance AI Server...
echo.
echo [IMPORTANT] ACCESS YOUR SITE AT: http://localhost:8000
echo.
echo (This window will automatically restart the server if it crashes)
echo.

:server_loop
echo [%time%] Starting Backend...
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --log-level warning
echo.
echo [!] Server crashed or stopped. Restarting in 5 seconds...
echo [!] Press Ctrl+C now to stop the loop.
timeout /t 5
goto server_loop
