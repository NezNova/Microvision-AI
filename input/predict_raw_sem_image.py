import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification

from prepare_sem_dataset import split_sem_and_metadata


def parse_args():
    parser = argparse.ArgumentParser(
        description="Classify a raw SEM image by cropping away the microscope ribbon first."
    )
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--metadata-min-height", type=int, default=35)
    parser.add_argument(
        "--save-cropped-image",
        default="",
        help="Optional path to save the cropped SEM image that was sent to the model.",
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


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoImageProcessor.from_pretrained(args.model_dir)
    model = AutoModelForImageClassification.from_pretrained(args.model_dir).to(device)
    model.eval()

    with Image.open(args.image) as image:
        sem_image, metadata_image, sem_box, metadata_box = split_sem_and_metadata(
            image,
            args.metadata_min_height,
        )
        if args.save_cropped_image:
            Path(args.save_cropped_image).parent.mkdir(parents=True, exist_ok=True)
            sem_image.save(args.save_cropped_image, quality=95)
        pixel_values = build_transform(processor)(sem_image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(pixel_values=pixel_values).logits
        probs = torch.softmax(logits, dim=1)[0]

    print(f"SEM crop box: {sem_box}")
    print(f"Metadata crop box: {metadata_box}")
    print(f"Metadata ribbon detected: {'yes' if metadata_image is not None else 'no'}")

    top_k = min(args.top_k, probs.numel())
    values, indices = probs.topk(top_k)
    for score, idx in zip(values.cpu().tolist(), indices.cpu().tolist()):
        print(f"{label_for(model, idx)}: {score:.4f}")


if __name__ == "__main__":
    main()
