# Spill Sense - Frontend Runner
Set-Location -Path "$PSScriptRoot\frontend"

Write-Host "Node: $(node --version)"
Write-Host "NPM:  $(npm --version)"

Write-Host "Starting Vite dev server on port 5173..."
npm run dev -- --host
