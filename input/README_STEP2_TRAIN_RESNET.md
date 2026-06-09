# Step 2: Train Microsoft ResNet-50 on Cropped SEM Images

Run this after Step 1 has created `prepared_sem_dataset`.

The trainer uses:

- `prepared_sem_dataset/train/<class_name>/*.jpg`
- `prepared_sem_dataset/val/<class_name>/*.jpg`

These images are already cropped so the white microscope ribbon is not part of the model input.

## Run Training

```powershell
cd "C:\Users\kom-e14-1\Documents\Codex\2026-06-05\i-have-sem-images-i-chategorized\outputs"
.\.venv\Scripts\Activate.ps1
.\run_step2_train_resnet.ps1
```

The first run downloads `microsoft/resnet-50`, so internet access is needed.

If your computer has limited memory, edit `run_step2_train_resnet.ps1` and change:

```powershell
--batch-size 16
```

to:

```powershell
--batch-size 8
```

or:

```powershell
--batch-size 4
```

## Output

The trained model is saved to:

```text
sem_resnet_model
```

Important output files:

- `sem_resnet_model\config.json`
- `sem_resnet_model\model.safetensors`
- `sem_resnet_model\preprocessor_config.json`
- `sem_resnet_model\metrics.json`
- `sem_resnet_model\history.json`
- `sem_resnet_model\labels.json`

`metrics.json` shows validation accuracy, classification report, and confusion matrix.

## Predict a Raw SEM Image

Use this for a new SEM image that still contains the bottom white microscope ribbon:

```powershell
.\run_step3_predict_example.ps1 -ImagePath "C:\path\to\new_sem_image.jpg"
```

This crops away the ribbon before prediction, matching the training data.

## Metadata

For this step, the model trains only on cropped SEM images.

After this baseline works, Step 3 can combine:

- ResNet image features
- microscope metadata features from `prepared_sem_dataset\metadata_features.csv`

That is the better order: first prove the image model works, then add metadata.
