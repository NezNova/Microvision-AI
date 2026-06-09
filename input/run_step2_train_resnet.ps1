$PreparedDir = ".\prepared_sem_dataset"
$ModelDir = ".\sem_resnet_model"

$PythonCmd = "python"
if (Test-Path ".\.venv\Scripts\python.exe") {
  $PythonCmd = ".\.venv\Scripts\python.exe"
}

& $PythonCmd -c "import sys; print('Python used:', sys.executable); import torch; print('Torch version:', torch.__version__)"
if ($LASTEXITCODE -ne 0) {
  Write-Host ""
  Write-Host "Torch is not installed in the Python environment used by this script."
  Write-Host "Install it with:"
  Write-Host "  $PythonCmd -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
  exit 1
}

& $PythonCmd .\train_sem_resnet.py `
  --data-dir $PreparedDir `
  --output-dir $ModelDir `
  --model-name "microsoft/resnet-50" `
  --epochs 20 `
  --batch-size 16 `
  --learning-rate 0.00003
