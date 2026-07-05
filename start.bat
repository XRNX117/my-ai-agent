@echo off
echo ==========================================
echo 正在启动 AI 智能体服务...
echo ==========================================
call venv\Scripts\activate.bat
echo.
echo 环境已激活，正在启动 Web 服务器...
echo 请在浏览器打开：http://localhost:8000
echo.
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
pause
