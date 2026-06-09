$PreparedDir = ".\prepared_sem_dataset"
$TesseractCmd = ".\prepared_sem_dataset\metadata\tesseract.exe"
$env:TESSDATA_PREFIX = ".\prepared_sem_dataset\metadata\tessdata"

python .\extract_metadata_from_existing.py `
  --prepared-dir $PreparedDir `
  --manifest "manifest.csv" `
  --output-csv "metadata_features.csv" `
  --tesseract-cmd $TesseractCmd
