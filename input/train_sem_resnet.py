import argparse
import json # use orjson
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification


ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune Microsoft's pretrained ResNet-50 on labeled SEM image folders."
    )
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("sem_resnet_model"), type=Path)
    parser.add_argument("--model-name", default="microsoft/resnet-50")
    parser.add_argument("--exclude", nargs="*", default=["PaxHeader"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=3)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_samples(data_dir, excluded):
    excluded = set(excluded or [])
    class_dirs = [
        path for path in sorted(data_dir.iterdir())
        if path.is_dir() and path.name not in excluded
    ]

    samples = []
    class_names = []
    for class_dir in class_dirs:
        images = [
            path for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if images:
            class_names.append(class_dir.name)
            samples.extend((image_path, class_dir.name) for image_path in images)

    if len(class_names) < 2:
        raise ValueError("Expected at least two non-empty class folders.")

    label_to_id = {name: idx for idx, name in enumerate(class_names)}
    indexed = [(path, label_to_id[label]) for path, label in samples]
    return indexed, class_names


def collect_prepared_samples(data_dir):
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    if not train_dir.exists() or not val_dir.exists():
        return None

    class_dirs = [path for path in sorted(train_dir.iterdir()) if path.is_dir()]
    class_names = [path.name for path in class_dirs]
    label_to_id = {name: idx for idx, name in enumerate(class_names)}

    def collect_split(split_dir):
        split_samples = []
        for class_name in class_names:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue
            images = [
                path for path in class_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ]
            split_samples.extend((image_path, label_to_id[class_name]) for image_path in images)
        return split_samples

    train_samples = collect_split(train_dir)
    val_samples = collect_split(val_dir)
    if not train_samples or not val_samples:
        raise ValueError("Prepared dataset must contain images in both train and val folders.")
    return train_samples, val_samples, class_names


class SemDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as image:
            image = image.convert("RGB")
            pixel_values = self.transform(image)
        return pixel_values, torch.tensor(label, dtype=torch.long)


def build_transforms(processor, train):
    size = processor.size.get("shortest_edge", 224)
    crop_size = processor.size.get("height", size)
    mean = processor.image_mean
    std = processor.image_std

    if train:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomResizedCrop(crop_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    return transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def stratified_split(samples, val_size, seed):
    labels = [label for _, label in samples]
    counts = np.bincount(labels)
    use_stratify = counts.min() >= 2 and int(round(len(samples) * val_size)) >= len(counts)
    stratify = labels if use_stratify else None
    return train_test_split(
        samples,
        test_size=val_size,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )


def set_backbone_trainable(model, trainable):
    for name, param in model.named_parameters():
        if not name.startswith("classifier."):
            param.requires_grad = trainable


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    loss_fn = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for pixel_values, labels in loader:
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)
            outputs = model(pixel_values=pixel_values)
            loss = loss_fn(outputs.logits, labels)
            total_loss += loss.item() * labels.size(0)
            all_preds.extend(outputs.logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    return total_loss / len(loader.dataset), float(accuracy), all_labels, all_preds


def main():
    args = parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prepared = collect_prepared_samples(args.data_dir)
    if prepared:
        train_samples, val_samples, class_names = prepared
    else:
        samples, class_names = collect_samples(args.data_dir, args.exclude)
        train_samples, val_samples = stratified_split(samples, args.val_size, args.seed)

    id_to_label = {idx: name for idx, name in enumerate(class_names)}
    label_to_id = {name: idx for idx, name in id_to_label.items()}

    processor = AutoImageProcessor.from_pretrained(args.model_name)
    train_ds = SemDataset(train_samples, build_transforms(processor, train=True))
    val_ds = SemDataset(val_samples, build_transforms(processor, train=False))
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = AutoModelForImageClassification.from_pretrained(
        args.model_name,
        num_labels=len(class_names),
        id2label=id_to_label,
        label2id=label_to_id,
        ignore_mismatched_sizes=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_labels = np.array([label for _, label in train_samples])
    class_counts = np.bincount(train_labels, minlength=len(class_names))
    class_weights = class_counts.sum() / np.maximum(class_counts, 1)
    class_weights = class_weights / class_weights.mean()
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    best_acc = -1.0
    history = []

    print(f"Classes: {class_names}")
    print(f"Training images: {len(train_ds)} | Validation images: {len(val_ds)}")
    print(f"Device: {device}")

    for epoch in range(1, args.epochs + 1):
        if epoch == 1 and args.freeze_backbone_epochs > 0:
            set_backbone_trainable(model, False)
            optimizer = torch.optim.AdamW(
                [param for param in model.parameters() if param.requires_grad],
                lr=args.learning_rate,
                weight_decay=0.01,
            )
        if epoch == args.freeze_backbone_epochs + 1:
            set_backbone_trainable(model, True)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.learning_rate,
                weight_decay=0.01,
            )

        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for pixel_values, labels in progress:
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(pixel_values=pixel_values)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / len(train_ds)
        val_loss, val_acc, y_true, y_pred = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        }
        history.append(row)
        print(json.dumps(row, indent=2))

        if val_acc > best_acc:
            best_acc = val_acc
            model.save_pretrained(args.output_dir)
            processor.save_pretrained(args.output_dir)
            with (args.output_dir / "labels.json").open("w", encoding="utf-8") as f:
                json.dump({"id2label": id_to_label, "label2id": label_to_id}, f, indent=2)

            report = classification_report(
                y_true,
                y_pred,
                labels=list(range(len(class_names))),
                target_names=class_names,
                zero_division=0,
                output_dict=True,
            )
            matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
            with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "best_val_accuracy": best_acc,
                        "classification_report": report,
                        "confusion_matrix": matrix.tolist(),
                        "class_names": class_names,
                        "train_images": len(train_ds),
                        "validation_images": len(val_ds),
                    },
                    f,
                    indent=2,
                )

    with (args.output_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Best validation accuracy: {best_acc:.4f}")
    print(f"Saved best model to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
