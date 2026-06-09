# Step 1: Load, Label, Resize, and Augment SEM Images

This step prepares your classified SEM folders before model training.

Each original SEM file contains two visual parts:

- the gray SEM image region, which should be used for training
- the bottom white microscope information ribbon, which should be saved as metadata

The script reads each class from the folder name:

- `3d_edge`
- `Bond-Pad-Array`
- `cantilever`
- `close_up_line`
- `Electrode`
- `label`
- `microfluidic`
- `waveguide`

`PaxHeader` is ignored by default.

## What This Step Produces

It creates:

- `prepared_sem_dataset/train/<class_name>/*.jpg`
- `prepared_sem_dataset/val/<class_name>/*.jpg`
- `prepared_sem_dataset/metadata/train/<class_name>/*_metadata.jpg`
- `prepared_sem_dataset/metadata/val/<class_name>/*_metadata.jpg`
- `prepared_sem_dataset/manifest.csv`
- `prepared_sem_dataset/train_manifest.csv`
- `prepared_sem_dataset/val_manifest.csv`
- `prepared_sem_dataset/metadata_features.csv`
- `prepared_sem_dataset/class_mapping.json`
- `prepared_sem_dataset/summary.json`

The model images are automatically cropped above the white microscope ribbon, then resized to `224 x 224`, which is the standard input size for Microsoft ResNet-50.

The metadata ribbon is not used for model training. It is saved separately and referenced in the CSV files through `metadata_image_path`.

If OCR is installed, microscope settings from the ribbon are also parsed into feature columns:

- `scale_value`, `scale_unit`
- `eht_kv`
- `wd_mm`
- `signal`
- `stage_t_deg`
- `stage_z_mm`
- `mag_value`, `mag_unit`
- `aperture_um`
- `ocr_text`

These fields are saved in `metadata_features.csv` and also copied into the manifest files.

For OCR, install both:

```powershell
pip install pytesseract
```

and the Windows Tesseract application. After installing Tesseract, restart PowerShell so it is available on PATH.

Small classes are augmented in the training set until they reach 200 training images by default. Validation images are never augmented.

## Run Step 1

Recommended:

```powershell
cd "C:\Users\kom-e14-1\Documents\Codex\2026-06-05\i-have-sem-images-i-chategorized\outputs"
.\run_step1_prepare_dataset.ps1
```

Or run the command manually:

```powershell
cd "C:\Users\kom-e14-1\Documents\Codex\2026-06-05\i-have-sem-images-i-chategorized\outputs"

python .\prepare_sem_dataset.py `
  --data-dir "C:\Users\kom-e14-1\OneDrive\MLTraining\project\microvision-ai\SEM\MEMS_devices_and_electrodes (1)\MEMS_devices_and_electrodes" `
  --output-dir ".\prepared_sem_dataset" `
  --image-size 224 `
  --val-size 0.2 `
  --metadata-min-height 35 `
  --augment-minority-to 200
```

If Tesseract is installed but not on PATH, add the executable path:

```powershell
python .\prepare_sem_dataset.py `
  --data-dir "C:\Users\kom-e14-1\OneDrive\MLTraining\project\microvision-ai\SEM\MEMS_devices_and_electrodes (1)\MEMS_devices_and_electrodes" `
  --output-dir ".\prepared_sem_dataset" `
  --image-size 224 `
  --val-size 0.2 `
  --metadata-min-height 35 `
  --augment-minority-to 200 `
  --tesseract-cmd "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

In PowerShell, the backtick character `` ` `` must be the very last character on a continued line. If there is a space after it, PowerShell will not pass the next line to Python, and Python will say `--data-dir` is missing.

## Notes

The original SEM folders are not changed.

If a rare image has no detectable bottom ribbon, the full image is used and `metadata_image_path` is left empty.

The `label` class has only 2 original images. Even after augmentation, the model may not learn that class reliably because augmented copies still come from only 2 examples.
