import argparse
import csv
from pathlib import Path

import torch
from PIL import Image, ImageFile
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification

from prepare_sem_dataset import IMAGE_EXTENSIONS, split_sem_and_metadata


ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict classes for a folder of unlabeled raw SEM images."
    )
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-csv", default=Path("unlabeled_predictions.csv"), type=Path)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Maximum number of images to predict. Use 0 for all images.",
    )
    parser.add_argument("--metadata-min-height", type=int, default=35)
    parser.add_argument(
        "--save-crops-dir",
        default=None,
        type=Path,
        help="Optional folder for saving cropped SEM images used by the model.",
    )
    return parser.parse_args()


def build_transform(processor):
    size = processor.size.get("shortest_edge", 224)
    crop_size = processor.size.get("height", size)
    return transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])


def label_for(model, idx):
    return model.config.id2label[str(idx)] if str(idx) in model.config.id2label else model.config.id2label[idx]


def list_images(input_dir):
    return sorted(
        path for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def predict_one(model, processor, transform, image_path, device, args):
    with Image.open(image_path) as image:
        sem_image, metadata_image, sem_box, metadata_box = split_sem_and_metadata(
            image,
            args.metadata_min_height,
        )
        if args.save_crops_dir:
            crop_path = args.save_crops_dir / f"{image_path.stem}_cropped.jpg"
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            sem_image.save(crop_path, quality=95)
        else:
            crop_path = ""

        pixel_values = transform(sem_image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(pixel_values=pixel_values).logits
        probs = torch.softmax(logits, dim=1)[0]

    top_k = min(args.top_k, probs.numel())
    values, indices = probs.topk(top_k)

    row = {
        "image_path": str(image_path.resolve()),
        "cropped_image_path": str(crop_path.resolve()) if crop_path else "",
        "metadata_ribbon_detected": "true" if metadata_image is not None else "false",
        "sem_crop_box": str(sem_box),
        "metadata_crop_box": str(metadata_box) if metadata_box else "",
    }

    for rank, (score, idx) in enumerate(zip(values.cpu().tolist(), indices.cpu().tolist()), start=1):
        row[f"top{rank}_label"] = label_for(model, idx)
        row[f"top{rank}_score"] = f"{score:.6f}"

    return row


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoImageProcessor.from_pretrained(args.model_dir)
    model = AutoModelForImageClassification.from_pretrained(args.model_dir).to(device)
    model.eval()
    transform = build_transform(processor)

    image_paths = list_images(args.input_dir)
    if args.max_images > 0:
        image_paths = image_paths[:args.max_images]
    if not image_paths:
        raise ValueError(f"No image files found under {args.input_dir}")

    rows = []
    for index, image_path in enumerate(image_paths, start=1):
        print(f"Predicting {index}/{len(image_paths)}: {image_path.name}")
        try:
            rows.append(predict_one(model, processor, transform, image_path, device, args))
        except Exception as exc:
            rows.append({
                "image_path": str(image_path.resolve()),
                "error": str(exc),
            })

    fieldnames = [
        "image_path",
        "cropped_image_path",
        "metadata_ribbon_detected",
        "sem_crop_box",
        "metadata_crop_box",
    ]
    for rank in range(1, args.top_k + 1):
        fieldnames.extend([f"top{rank}_label", f"top{rank}_score"])
    fieldnames.append("error")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved predictions to: {args.output_csv.resolve()}")


if __name__ == "__main__":
    main()
