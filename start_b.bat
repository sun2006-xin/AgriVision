@echo off
title AgriVision - Monitoring System (System B)

echo ============================================
echo   AgriVision System B - Real-time Monitor
echo ============================================

cd /d "%~dp0system_b\core"

:: Check virtual environment
if not exist "..\.venv\Scripts\activate" (
    echo [Init] First run, creating virtual environment...
    python -m venv ..\.venv
    echo [Init] venv created, installing dependencies...
    call ..\.venv\Scripts\activate
    pip install -r "%~dp0requirements-b.txt" -i https://mirrors.aliyun.com/pypi/simple/
    echo [Init] Dependencies installed!
) else (
    call ..\.venv\Scripts\activate
)

echo [1/2] Starting monitor backend...
start "AgriVision-Monitor" cmd /k "cd /d "%~dp0system_b\core" && call ..\.venv\Scripts\activate && python app.py"

echo Waiting for backend init (3s)...
timeout /t 3 /nobreak >nul

echo [2/2] Opening monitor page...
start "" "http://127.0.0.1:5000"

echo ============================================
echo   System B started!
echo   Monitor: http://127.0.0.1:5000
echo   Do not close the backend window
echo ============================================
pause
