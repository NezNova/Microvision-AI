import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification


def parse_args():
    parser = argparse.ArgumentParser(description="Classify one SEM image with a trained model.")
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def build_transform(processor):
    size = processor.size.get("shortest_edge", 224)
    crop_size = processor.size.get("height", size)
    return transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoImageProcessor.from_pretrained(args.model_dir)
    model = AutoModelForImageClassification.from_pretrained(args.model_dir).to(device)
    model.eval()

    with Image.open(args.image) as image:
        pixel_values = build_transform(processor)(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(pixel_values=pixel_values).logits
        probs = torch.softmax(logits, dim=1)[0]

    top_k = min(args.top_k, probs.numel())
    values, indices = probs.topk(top_k)
    for score, idx in zip(values.cpu().tolist(), indices.cpu().tolist()):
        label = model.config.id2label[str(idx)] if str(idx) in model.config.id2label else model.config.id2label[idx]
        print(f"{label}: {score:.4f}")


if __name__ == "__main__":
    main()
