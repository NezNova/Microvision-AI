import argparse
import csv
import json
import random
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFile, ImageOps


ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
METADATA_FIELDS = [
    "ocr_text",
    "scale_value",
    "scale_unit",
    "eht_kv",
    "wd_mm",
    "signal",
    "stage_t_deg",
    "stage_z_mm",
    "mag_value",
    "mag_unit",
    "aperture_um",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load SEM images, label them from folder names, resize them, and create train/validation data."
    )
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("prepared_sem_dataset"), type=Path)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude", nargs="*", default=["PaxHeader"])
    parser.add_argument(
        "--metadata-min-height",
        type=int,
        default=35,
        help="Smallest detected bottom ribbon height, in pixels.",
    )
    parser.add_argument(
        "--disable-ocr",
        action="store_true",
        help="Save metadata ribbon images but do not OCR/parse microscope text.",
    )
    parser.add_argument(
        "--tesseract-cmd",
        default="",
        help="Optional full path to tesseract.exe, for example C:\\Program Files\\Tesseract-OCR\\tesseract.exe.",
    )
    parser.add_argument(
        "--augment-minority-to",
        type=int,
        default=200,
        help="Create augmented training images until each class has at least this many training images. Use 0 to disable.",
    )
    return parser.parse_args()


def blank_metadata_features():
    return {field: "" for field in METADATA_FIELDS}


def preprocess_metadata_for_ocr(metadata_image):
    gray = metadata_image.convert("L")
    width, height = gray.size
    enlarged = gray.resize((width * 3, height * 3), Image.Resampling.LANCZOS)
    enhanced = ImageOps.autocontrast(enlarged)
    enhanced = ImageEnhance.Contrast(enhanced).enhance(2.0)
    return enhanced


def ocr_metadata_text(metadata_image, tesseract_cmd):
    try:
        import pytesseract
    except ImportError:
        return ""

    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    processed = preprocess_metadata_for_ocr(metadata_image)
    try:
        text = pytesseract.image_to_string(processed, config="--psm 6")
    except Exception:
        return ""
    return " ".join(text.replace("\n", " ").split())


def first_match(pattern, text, group=1, flags=re.IGNORECASE):
    match = re.search(pattern, text, flags)
    if not match:
        return ""
    return match.group(group).strip()


def parse_float(value):
    if not value:
        return ""
    cleaned = value.replace(",", ".").replace(" ", "")
    try:
        return str(float(cleaned))
    except ValueError:
        return ""


def parse_metadata_text(text):
    features = blank_metadata_features()
    features["ocr_text"] = text
    if not text:
        return features

    features["scale_value"] = parse_float(first_match(r"(\d+(?:[.,]\d+)?)\s*(?:u|µ|μ)m", text))
    features["scale_unit"] = "um" if features["scale_value"] else ""
    features["eht_kv"] = parse_float(first_match(r"EHT\s*=\s*(\d+(?:[.,]\d+)?)\s*kV", text))
    features["wd_mm"] = parse_float(first_match(r"WD\s*=\s*(\d+(?:[.,]\d+)?)\s*mm", text))
    features["signal"] = first_match(r"Signal\s*A\s*=\s*([A-Za-z0-9]+)", text)
    features["stage_t_deg"] = parse_float(first_match(r"Stage\s*at\s*T\s*=\s*([+-]?\d+(?:[.,]\d+)?)", text))
    features["stage_z_mm"] = parse_float(first_match(r"Stage\s*at\s*Z\s*=\s*([+-]?\d+(?:[.,]\d+)?)\s*mm", text))
    features["mag_value"] = parse_float(first_match(r"Mag\s*=\s*(\d+(?:[.,]\d+)?)\s*([KkMm]?)X", text, group=1))
    mag_prefix = first_match(r"Mag\s*=\s*\d+(?:[.,]\d+)?\s*([KkMm]?)X", text).upper()
    features["mag_unit"] = f"{mag_prefix}X" if features["mag_value"] else ""
    features["aperture_um"] = parse_float(first_match(r"Aperture\s*Size\s*=\s*(\d+(?:[.,]\d+)?)\s*(?:u|µ|μ)m", text))
    return features


def extract_metadata_features(metadata_image, disable_ocr, tesseract_cmd):
    if metadata_image is None or disable_ocr:
        return blank_metadata_features()
    return parse_metadata_text(ocr_metadata_text(metadata_image, tesseract_cmd))


def list_images(data_dir, excluded):
    excluded = set(excluded or [])
    class_dirs = [
        path for path in sorted(data_dir.iterdir())
        if path.is_dir() and path.name not in excluded
    ]

    samples = []
    class_names = []
    for class_dir in class_dirs:
        image_paths = [
            path for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if image_paths:
            class_names.append(class_dir.name)
            samples.extend((path, class_dir.name) for path in sorted(image_paths))

    if not samples:
        raise ValueError(f"No images found under {data_dir}")

    return samples, class_names


def split_samples(samples, val_size, seed):
    random.seed(seed)
    by_class = {}
    for path, label in samples:
        by_class.setdefault(label, []).append((path, label))

    train = []
    val = []
    for label, label_samples in by_class.items():
        label_samples = label_samples[:]
        random.shuffle(label_samples)

        if len(label_samples) == 1:
            train.extend(label_samples)
            continue

        val_count = max(1, round(len(label_samples) * val_size))
        val_count = min(val_count, len(label_samples) - 1)
        val.extend(label_samples[:val_count])
        train.extend(label_samples[val_count:])

    random.shuffle(train)
    random.shuffle(val)
    return train, val


def detect_bottom_metadata_ribbon(image, min_height):
    """Find the bright microscope information ribbon at the bottom of SEM images."""
    gray = ImageOps.exif_transpose(image.convert("L"))
    width, height = gray.size
    pixels = np.asarray(gray, dtype=np.float32)

    row_mean = pixels.mean(axis=1)
    row_std = pixels.std(axis=1)
    bright_rows = (row_mean > 185) & (row_std > 15)

    bottom_limit = int(height * 0.65)
    candidates = np.where(bright_rows[bottom_limit:])[0]
    if len(candidates) == 0:
        return height, None

    ribbon_start = bottom_limit + int(candidates[0])
    if height - ribbon_start < min_height:
        return height, None

    # Include the thin black border line above the white ribbon when present.
    search_start = max(0, ribbon_start - 8)
    border_region = row_mean[search_start:ribbon_start]
    dark_rows = np.where(border_region < 80)[0]
    if len(dark_rows) > 0:
        ribbon_start = search_start + int(dark_rows[-1])

    ribbon_start = max(1, min(ribbon_start, height - min_height))
    sem_box = (0, 0, width, ribbon_start)
    metadata_box = (0, ribbon_start, width, height)
    return ribbon_start, (sem_box, metadata_box)


def split_sem_and_metadata(image, metadata_min_height):
    image = ImageOps.exif_transpose(image.convert("RGB"))
    _, boxes = detect_bottom_metadata_ribbon(image, metadata_min_height)
    if boxes is None:
        return image, None, (0, 0, image.width, image.height), None

    sem_box, metadata_box = boxes
    return image.crop(sem_box), image.crop(metadata_box), sem_box, metadata_box


def resize_with_padding(image, image_size):
    image = ImageOps.exif_transpose(image.convert("RGB"))
    image.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (image_size, image_size), (0, 0, 0))
    x = (image_size - image.width) // 2
    y = (image_size - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def safe_name(path):
    return path.stem.replace(" ", "_").replace(".", "_")


def save_prepared_image(
    src_path,
    dst_path,
    metadata_path,
    image_size,
    metadata_min_height,
    disable_ocr,
    tesseract_cmd,
):
    with Image.open(src_path) as image:
        sem_image, metadata_image, sem_box, metadata_box = split_sem_and_metadata(
            image,
            metadata_min_height,
        )
        prepared = resize_with_padding(sem_image, image_size)
        prepared.save(dst_path, quality=95)
        if metadata_image is not None:
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            metadata_image.save(metadata_path, quality=95)
            metadata_saved_path = str(metadata_path.resolve())
            metadata_features = extract_metadata_features(metadata_image, disable_ocr, tesseract_cmd)
        else:
            metadata_saved_path = ""
            metadata_features = blank_metadata_features()
    return sem_box, metadata_box, metadata_saved_path, metadata_features


def augment_image(image, variant):
    image = image.copy()
    if variant % 2 == 0:
        image = ImageOps.mirror(image)
    if variant % 3 == 0:
        image = ImageOps.flip(image)

    angle = [-15, -10, -5, 5, 10, 15][variant % 6]
    image = image.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=(0, 0, 0))

    brightness = [0.9, 0.95, 1.05, 1.1][variant % 4]
    contrast = [0.9, 1.0, 1.1, 1.2][(variant + 1) % 4]
    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    return image


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "label",
                "label_id",
                "image_path",
                "metadata_image_path",
                "source_path",
                "is_augmented",
                "sem_crop_box",
                "metadata_crop_box",
                *METADATA_FIELDS,
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def prepare_split(
    samples,
    split_name,
    output_dir,
    class_to_id,
    image_size,
    metadata_min_height,
    disable_ocr,
    tesseract_cmd,
):
    rows = []
    for src_path, label in samples:
        class_dir = output_dir / split_name / label
        metadata_dir = output_dir / "metadata" / split_name / label
        class_dir.mkdir(parents=True, exist_ok=True)
        dst_path = class_dir / f"{safe_name(src_path)}.jpg"
        metadata_path = metadata_dir / f"{safe_name(src_path)}_metadata.jpg"
        sem_box, metadata_box, metadata_saved_path, metadata_features = save_prepared_image(
            src_path,
            dst_path,
            metadata_path,
            image_size,
            metadata_min_height,
            disable_ocr,
            tesseract_cmd,
        )
        row = {
            "split": split_name,
            "label": label,
            "label_id": class_to_id[label],
            "image_path": str(dst_path.resolve()),
            "metadata_image_path": metadata_saved_path,
            "source_path": str(src_path.resolve()),
            "is_augmented": "false",
            "sem_crop_box": json.dumps(sem_box),
            "metadata_crop_box": json.dumps(metadata_box) if metadata_box else "",
        }
        row.update(metadata_features)
        rows.append(row)
    return rows


def augment_minority_classes(train_rows, output_dir, image_size, metadata_min_height, target_count, seed):
    if target_count <= 0:
        return []

    random.seed(seed)
    rows_by_label = {}
    for row in train_rows:
        rows_by_label.setdefault(row["label"], []).append(row)

    augmented_rows = []
    for label, rows in rows_by_label.items():
        needed = target_count - len(rows)
        if needed <= 0:
            continue

        class_dir = output_dir / "train" / label
        for index in range(needed):
            base_row = rows[index % len(rows)]
            src_path = Path(base_row["source_path"])
            dst_path = class_dir / f"{safe_name(src_path)}_aug_{index + 1:04d}.jpg"

            with Image.open(src_path) as image:
                sem_image, _, sem_box, metadata_box = split_sem_and_metadata(
                    image,
                    metadata_min_height,
                )
                prepared = resize_with_padding(sem_image, image_size)
                augmented = augment_image(prepared, index)
                augmented.save(dst_path, quality=95)

            augmented_rows.append({
                "split": "train",
                "label": label,
                "label_id": base_row["label_id"],
                "image_path": str(dst_path.resolve()),
                "metadata_image_path": base_row["metadata_image_path"],
                "source_path": str(src_path.resolve()),
                "is_augmented": "true",
                "sem_crop_box": json.dumps(sem_box),
                "metadata_crop_box": json.dumps(metadata_box) if metadata_box else "",
                **{field: base_row.get(field, "") for field in METADATA_FIELDS},
            })

    return augmented_rows


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples, class_names = list_images(args.data_dir, args.exclude)
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    id_to_class = {idx: name for name, idx in class_to_id.items()}

    train_samples, val_samples = split_samples(samples, args.val_size, args.seed)

    train_rows = prepare_split(
        train_samples,
        "train",
        args.output_dir,
        class_to_id,
        args.image_size,
        args.metadata_min_height,
        args.disable_ocr,
        args.tesseract_cmd,
    )
    val_rows = prepare_split(
        val_samples,
        "val",
        args.output_dir,
        class_to_id,
        args.image_size,
        args.metadata_min_height,
        args.disable_ocr,
        args.tesseract_cmd,
    )
    augmented_rows = augment_minority_classes(
        train_rows,
        args.output_dir,
        args.image_size,
        args.metadata_min_height,
        args.augment_minority_to,
        args.seed,
    )

    all_rows = train_rows + augmented_rows + val_rows
    write_csv(args.output_dir / "manifest.csv", all_rows)
    write_csv(args.output_dir / "train_manifest.csv", train_rows + augmented_rows)
    write_csv(args.output_dir / "val_manifest.csv", val_rows)
    write_csv(args.output_dir / "metadata_features.csv", train_rows + val_rows)

    summary = {
        "source_data_dir": str(args.data_dir.resolve()),
        "prepared_data_dir": str(args.output_dir.resolve()),
        "image_size": args.image_size,
        "classes": class_names,
        "class_to_id": class_to_id,
        "id_to_class": id_to_class,
        "original_images": len(samples),
        "train_original_images": len(train_rows),
        "validation_images": len(val_rows),
        "augmented_training_images": len(augmented_rows),
        "total_prepared_images": len(all_rows),
        "metadata_ribbons_saved": sum(1 for row in train_rows + val_rows if row["metadata_image_path"]),
        "ocr_enabled": not args.disable_ocr,
        "metadata_feature_columns": METADATA_FIELDS,
        "excluded_folders": args.exclude,
    }

    with (args.output_dir / "class_mapping.json").open("w", encoding="utf-8") as f:
        json.dump({"class_to_id": class_to_id, "id_to_class": id_to_class}, f, indent=2)

    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
