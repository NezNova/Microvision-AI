$DataDir = "C:\Users\kom-e14-1\OneDrive\MLTraining\project\microvision-ai\SEM\MEMS_devices_and_electrodes (1)\MEMS_devices_and_electrodes"
$OutputDir = ".\prepared_sem_dataset"
$TesseractCmd = ".\prepared_sem_dataset\metadata\tesseract.exe"
$env:TESSDATA_PREFIX = ".\prepared_sem_dataset\metadata\tessdata"

$ArgsList = @(
  ".\prepare_sem_dataset.py",
  "--data-dir", $DataDir,
  "--output-dir", $OutputDir,
  "--image-size", "224",
  "--val-size", "0.2",
  "--metadata-min-height", "35",
  "--augment-minority-to", "200"
)

if ($TesseractCmd) {
  $ArgsList += @("--tesseract-cmd", $TesseractCmd)
}

python @ArgsList
