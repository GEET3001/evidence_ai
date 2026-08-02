# Starts the backend (FastAPI) and frontend (Next.js) dev servers together.
# Run from the repo root: .\dev.ps1
# Each server opens in its own PowerShell window so their logs stay separate;
# closing either window stops that server.

$root = $PSScriptRoot

Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "Set-Location '$root\backend'; .\venv\Scripts\Activate.ps1; uvicorn app.main:app --reload --port 8000"
)

Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "Set-Location '$root\frontend'; npm run dev"
)

Write-Host "Backend starting on http://localhost:8000 and frontend on http://localhost:3000 in separate windows."
