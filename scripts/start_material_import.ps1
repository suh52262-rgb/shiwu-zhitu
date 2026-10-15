[CmdletBinding()]
param([int]$Port=8504)
$ErrorActionPreference='Stop'
$materialRoot=Split-Path -Parent $PSScriptRoot
$materialPython=Join-Path $materialRoot '.venv\Scripts\python.exe'
if(!(Test-Path -LiteralPath $materialPython)){throw 'Project Python environment not found'}
if(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue){throw 'Material UI port is already in use'}
Push-Location $materialRoot
try { & $materialPython -m streamlit run frontend/material_import_app.py --server.address 127.0.0.1 --server.port $Port }
finally { Pop-Location }
