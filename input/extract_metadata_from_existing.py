import argparse
import csv
import json
from pathlib import Path

from PIL import Image

from prepare_sem_dataset import METADATA_FIELDS, extract_metadata_features


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract OCR metadata features from already-prepared SEM ribbon images."
    )
    parser.add_argument(
        "--prepared-dir",
        default=Path("prepared_sem_dataset"),
        type=Path,
        help="Existing prepared dataset folder.",
    )
    parser.add_argument(
        "--manifest",
        default="manifest.csv",
        help="Manifest file inside prepared-dir to read image/metadata paths from.",
    )
    parser.add_argument(
        "--output-csv",
        default="metadata_features.csv",
        help="Output CSV file inside prepared-dir.",
    )
    parser.add_argument(
        "--tesseract-cmd",
        default="",
        help="Optional full path to tesseract.exe.",
    )
    return parser.parse_args()


def read_manifest(path):
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    prepared_dir = args.prepared_dir
    manifest_path = prepared_dir / args.manifest
    output_path = prepared_dir / args.output_csv

    rows = read_manifest(manifest_path)
    output_rows = []

    for row in rows:
        metadata_path = row.get("metadata_image_path", "")
        if metadata_path and Path(metadata_path).exists():
            with Image.open(metadata_path) as image:
                features = extract_metadata_features(
                    image,
                    disable_ocr=False,
                    tesseract_cmd=args.tesseract_cmd,
                )
        else:
            features = {field: "" for field in METADATA_FIELDS}

        output_row = {
            "split": row.get("split", ""),
            "label": row.get("label", ""),
            "label_id": row.get("label_id", ""),
            "image_path": row.get("image_path", ""),
            "metadata_image_path": metadata_path,
            "source_path": row.get("source_path", ""),
            "is_augmented": row.get("is_augmented", ""),
        }
        output_row.update(features)
        output_rows.append(output_row)

    fieldnames = [
        "split",
        "label",
        "label_id",
        "image_path",
        "metadata_image_path",
        "source_path",
        "is_augmented",
        *METADATA_FIELDS,
    ]
    write_csv(output_path, output_rows, fieldnames)

    summary = {
        "manifest": str(manifest_path.resolve()),
        "output_csv": str(output_path.resolve()),
        "rows": len(output_rows),
        "metadata_images_found": sum(1 for row in output_rows if row["metadata_image_path"]),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
