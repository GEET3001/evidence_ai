# Starts the backend WITHOUT --reload for demo/recording purposes.
# See README: Demo Mode for the full pre-recording checklist.
# Run from the repo root: .\demo.ps1

$root = $PSScriptRoot

Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "Set-Location '$root\backend'; .\venv\Scripts\Activate.ps1; uvicorn app.main:app --port 8000"
)

Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "Set-Location '$root\frontend'; npm run dev"
)

Write-Host "Backend starting on http://localhost:8000 (no --reload) and frontend on http://localhost:3000."
Write-Host "Check http://localhost:8000/health before recording -- see README: Demo Mode."
