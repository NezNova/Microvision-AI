# SEM Image Classification with Microsoft ResNet-50

This trains a classifier for your SEM image folders using the pretrained Hugging Face model `microsoft/resnet-50`.

Your detected dataset folders:

| Folder | Image count |
| --- | ---: |
| `3d_edge` | 471 |
| `Bond-Pad-Array` | 825 |
| `cantilever` | 36 |
| `close_up_line` | 660 |
| `Electrode` | 628 |
| `label` | 2 |
| `microfluidic` | 18 |
| `waveguide` | 129 |

`PaxHeader` is excluded by default because it looks like metadata rather than a real class.

## Install

```powershell
cd "C:\Users\kom-e14-1\Documents\Codex\2026-06-05\i-have-sem-images-i-chategorized\outputs"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train

```powershell
python .\train_sem_resnet.py `
  --data-dir "C:\Users\kom-e14-1\OneDrive\MLTraining\project\microvision-ai\SEM\MEMS_devices_and_electrodes (1)\MEMS_devices_and_electrodes" `
  --output-dir ".\sem_resnet_model" `
  --epochs 20 `
  --batch-size 16
```

If your computer has low memory, use `--batch-size 8` or `--batch-size 4`.

## Predict a New SEM Image

```powershell
python .\predict_sem_resnet.py `
  --model-dir ".\sem_resnet_model" `
  --image "C:\path\to\new_sem_image.png"
```

## Important Note

The `label` class has only 2 images and `microfluidic` has only 18 images. A model can train with this, but those classes may be unreliable. For better results, try to collect more images for the smallest classes or remove `label` if it is not a real SEM object class.
