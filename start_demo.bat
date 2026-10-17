@echo off
title RAGOps Project - Quick Demo Launcher
echo ======================================================================
echo    RAGOps Enterprise Knowledge Assistant - Demo Instant Launcher
echo ======================================================================
echo Checking and restoring presentation services (takes 5-10 seconds)...
echo.

:: 1. Ensure Docker Compose services are running
echo [1/3] Ensuring Docker services (FastAPI, Prometheus, Grafana) are running...
docker compose up -d

:: 2. Check if MLflow is running on port 5000; if not, launch it in background
echo [2/3] Checking MLflow UI on port 5000...
netstat -ano | findstr 127.0.0.1:5000 >nul 2>&1
if %errorlevel% neq 0 (
    echo Starting MLflow UI in background...
    start /b mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
    timeout /t 3 >nul
) else (
    echo MLflow is already running.
)

:: 3. Open browser tabs for presentation
echo [3/3] Opening presentation dashboard tabs in browser...
start http://localhost:8000/docs
start http://localhost:5000
start http://localhost:3000

echo.
echo ======================================================================
echo All Demo Services are LIVE! Ready for presentation!
echo   - FastAPI Docs:  http://localhost:8000/docs
echo   - MLflow UI:     http://localhost:5000
echo   - Grafana UI:    http://localhost:3000 (admin / admin)
echo ======================================================================
pause
