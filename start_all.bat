@echo off
title AgriVision - Smart Agriculture Platform

echo ========================================================
echo   AgriVision - Smart Agriculture Pest Monitoring
echo   Launching System A (AI Diagnosis) + System B (Monitor)
echo ========================================================
echo.

echo [System A] Starting AI Diagnosis Station...
start "AgriVision-A" cmd /k "cd /d "%~dp0system_a\core" && if exist ..\.venv\Scripts\activate (call ..\.venv\Scripts\activate) else (echo Please run start_a.bat first to init environment) && uvicorn app_fastapi:app --reload --host 127.0.0.1 --port 8000"

echo Waiting for System A init (3s)...
timeout /t 3 /nobreak >nul

echo [System A] Opening frontend page...
start "" "%~dp0system_a\core\frontend.html"

echo.
echo [System B] Starting Real-time Monitor...
start "AgriVision-B" cmd /k "cd /d "%~dp0system_b\core" && if exist ..\.venv\Scripts\activate (call ..\.venv\Scripts\activate) else (echo Please run start_b.bat first to init environment) && python app.py"

echo Waiting for System B init (3s)...
timeout /t 3 /nobreak >nul

echo [System B] Opening monitor page...
start "" "http://127.0.0.1:5000"

echo.
echo [System A] API docs...
start "" "http://127.0.0.1:8000/docs"

echo.
echo ========================================================
echo   All systems started!
echo   System A: http://127.0.0.1:8000 (AI Diagnosis)
echo   System B: http://127.0.0.1:5000 (Real-time Monitor)
echo   API docs: http://127.0.0.1:8000/docs
echo   Do not close the backend windows
echo ========================================================
pause
