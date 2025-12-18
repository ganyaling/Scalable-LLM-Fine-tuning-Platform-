@echo off
REM Windows 本地开发启动脚本 - 自动启动所有组件
REM 确保已安装 Docker, Python, 和 Anaconda

setlocal enabledelayedexpansion
cls

echo.
echo ============================================================
echo Celery 异步任务系统 - 自动启动脚本
echo ============================================================
echo.

REM 检查 conda 是否可用
where conda >nul 2>nul
if !errorlevel! neq 0 (
    echo ❌ 错误：找不到 conda
    echo 请确保已安装 Anaconda 并且 conda 在 PATH 中
    pause
    exit /b 1
)

REM 检查 Docker 是否可用
docker --version >nul 2>nul
if !errorlevel! neq 0 (
    echo ❌ 错误：Docker 不可用
    echo 请检查:
    echo   1. Docker Desktop 是否已安装
    echo   2. Docker Desktop 是否正在运行
    echo   3. 运行: docker --version 来验证
    pause
    exit /b 1
)

echo ✓ 环境检查通过
echo   - Conda: OK
echo   - Docker: OK
echo.
echo 正在启动所有组件...
echo.

REM 启动 Redis (Docker)
echo [1/4] 启动 Redis...
docker run -d --name mini_llm_redis -p 6379:6379 redis:7-alpine
timeout /t 2 >nul

REM 启动 Celery Worker
echo [2/4] 启动 Celery Worker...
start "Celery Worker" cmd /k "conda activate myenv && python start_worker.py"
timeout /t 1 >nul

REM 启动 FastAPI Backend
echo [3/4] 启动 FastAPI Backend...
start "FastAPI Backend" cmd /k "conda activate myenv && cd backend && uvicorn api:app --reload --host 0.0.0.0 --port 8000"
timeout /t 1 >nul

REM 启动 Streamlit Frontend
echo [4/4] 启动 Streamlit Frontend...
start "Streamlit Frontend" cmd /k "conda activate myenv && cd frontend && streamlit run app.py"
timeout /t 1 >nul

echo.
echo ============================================================
echo ✓ 所有组件已启动！
echo ============================================================
echo.
echo 🌐 访问地址:
echo   - API:           http://localhost:8000
echo   - API 文档:      http://localhost:8000/docs
echo   - Streamlit:     http://localhost:8501
echo.
echo 📋 各组件运行状态:
echo   - Redis:         Docker 容器 (已启动)
echo   - Worker:        新终端窗口 (Celery Worker)
echo   - Backend:       新终端窗口 (FastAPI)
echo   - Frontend:      新终端窗口 (Streamlit)
echo.
echo ⚠️  注意事项:
echo   1. 确保所有新开的终端窗口都正常运行（没有红色错误）
echo   2. 如果有错误，检查:
echo      - pip install -r requirements_celery.txt
echo      - conda activate myenv
echo   3. 关闭此窗口不会停止其他组件
echo.
echo 🛑 停止所有服务:
echo   - 关闭其他所有终端窗口
echo   - 运行: docker stop mini_llm_redis
echo.
pause
