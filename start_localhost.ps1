# Spill Sense - Start Both Backend & Frontend on Localhost
$ProjectRoot = $PSScriptRoot

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "    Spill Sense - Starting Local Servers     " -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan

# Start Backend
Write-Host "[1/2] Starting FastAPI Backend on http://localhost:8000 ..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$ProjectRoot\backend'; python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload"

# Start Frontend
Write-Host "[2/2] Starting Vite Frontend on http://localhost:5173 ..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$ProjectRoot\frontend'; npm run dev -- --host --port 5173"

Write-Host "Servers launched!" -ForegroundColor Green
Write-Host "Frontend: http://localhost:5173" -ForegroundColor Green
Write-Host "Backend API: http://localhost:8000" -ForegroundColor Green
Write-Host "Swagger Docs: http://localhost:8000/docs" -ForegroundColor Green
