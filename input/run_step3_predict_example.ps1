param(
  [Parameter(Mandatory = $true)]
  [string]$ImagePath
)

$ModelDir = ".\sem_resnet_model"

python .\predict_raw_sem_image.py `
  --model-dir $ModelDir `
  --image $ImagePath `
  --top-k 3
