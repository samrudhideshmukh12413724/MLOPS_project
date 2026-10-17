# Quick Demo Launcher for PowerShell
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "   RAGOps Enterprise Knowledge Assistant - Quick Demo Launcher       " -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# 1. Start Docker stack if containers stopped
Write-Host "`n[1/3] Ensuring Docker services (FastAPI, Prometheus, Grafana) are running..." -ForegroundColor Yellow
docker compose up -d

# 2. Check & start MLflow UI if not running
Write-Host "`n[2/3] Checking MLflow UI on port 5000..." -ForegroundColor Yellow
$mlflowRunning = Get-NetTCPConnection -LocalPort 5000 -ErrorAction SilentlyContinue
if (-not $mlflowRunning) {
    Write-Host "Starting MLflow UI on port 5000..." -ForegroundColor Green
    Start-Process -FilePath "mlflow" -ArgumentList "ui --backend-store-uri sqlite:///mlflow.db --port 5000" -WindowStyle Hidden
    Start-Sleep -Seconds 3
} else {
    Write-Host "MLflow UI is already active on http://127.0.0.1:5000" -ForegroundColor Green
}

# 3. Open Demo Tabs in Default Browser
Write-Host "`n[3/3] Opening presentation tabs in browser..." -ForegroundColor Yellow
Start-Process "http://localhost:8000/docs"
Start-Process "http://localhost:5000"
Start-Process "http://localhost:3000"

Write-Host "`n======================================================================" -ForegroundColor Green
Write-Host "  All Demo Dashboards are LIVE and ready for your presentation!        " -ForegroundColor Green
Write-Host "  - FastAPI Swagger & UI: http://localhost:8000/docs                 " -ForegroundColor White
Write-Host "  - MLflow Experiments:   http://localhost:5000                      " -ForegroundColor White
Write-Host "  - Grafana Metrics:      http://localhost:3000 (admin / admin)      " -ForegroundColor White
Write-Host "======================================================================" -ForegroundColor Green
