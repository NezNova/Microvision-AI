param(
  [Parameter(Mandatory = $true)]
  [string]$InputDir,

  [int]$MaxImages = 1000
)

$PythonCmd = "C:\Users\kom-e14-1\OneDrive\MLTraining\project\microvision-ai\.venv\Scripts\python.exe"
$ModelDir = ".\sem_resnet_model"
$OutputCsv = ".\unlabeled_predictions.csv"
$CropsDir = ".\unlabeled_cropped_for_review"

& $PythonCmd .\predict_unlabeled_folder.py `
  --model-dir $ModelDir `
  --input-dir $InputDir `
  --output-csv $OutputCsv `
  --top-k 3 `
  --max-images $MaxImages `
  --save-crops-dir $CropsDir
