$PythonCmd = "python"
if (Test-Path ".\.venv\Scripts\python.exe") {
  $PythonCmd = ".\.venv\Scripts\python.exe"
}

Write-Host "Checking Python environment..."
& $PythonCmd -c "import sys; print('Python:', sys.executable); print('Version:', sys.version)"
& $PythonCmd -m pip --version
& $PythonCmd -m pip show torch
& $PythonCmd -m pip show torchvision
& $PythonCmd -m pip show transformers
