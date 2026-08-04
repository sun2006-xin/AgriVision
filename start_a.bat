@echo off
title AgriVision - AI Diagnosis Station (System A)

echo ============================================
echo   AgriVision System A - AI Diagnosis
echo ============================================

cd /d "%~dp0system_a\core"

:: Check virtual environment
if not exist "..\.venv\Scripts\activate" (
    echo [Init] First run, creating virtual environment...
    python -m venv --system-site-packages ..\.venv
    echo [Init] venv created, installing dependencies...
    call ..\.venv\Scripts\activate
    pip install fastapi uvicorn python-multipart pydantic onnxruntime torch torchvision ultralytics transformers qwen-vl-utils opencv-python pillow numpy -i https://mirrors.aliyun.com/pypi/simple/
    echo [Init] Dependencies installed!
) else (
    call ..\.venv\Scripts\activate
)

echo [1/3] Starting AI backend...
start "AgriVision-AI" cmd /k "cd /d "%~dp0system_a\core" && call ..\.venv\Scripts\activate && uvicorn app_fastapi:app --reload --host 0.0.0.0 --port 8000"

echo Waiting for backend init (3s)...
timeout /t 3 /nobreak >nul

echo [2/3] Opening frontend page...
start "" "frontend.html"

echo [3/3] Opening API docs...
start "" "http://127.0.0.1:8000/docs"

echo ============================================
echo   System A started!
echo   Frontend: opened
echo   API docs: http://127.0.0.1:8000/docs
echo   Do not close the backend window
echo ============================================
pause
