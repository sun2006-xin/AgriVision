@echo off
REM AgriVision System A 快捷启动（需先通过 start_a.bat 初始化 venv）
REM 直接启动 FastAPI 后端，监听 8000 端口
cd /d "%~dp0core"
"%~dp0.venv\Scripts\python.exe" -m uvicorn app_fastapi:app --host 0.0.0.0 --port 8000
pause
