"""
Download and prepare a real-world Reddit stress dataset.

The repository originally includes data/generate_dataset.py for educational
experiments. This script adds a real-data path without removing that generator.

Default dataset:
    Dreaddit: A Reddit Dataset for Stress Analysis in Social Media
    https://arxiv.org/abs/1911.00133

Output:
    data/raw/dreaddit_merged.csv
    data/processed/mental_health_dataset.csv

The processed file follows the schema expected by notebooks/train_pipeline.py:
    text,label,label_name,source
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

DREADDIT_URLS = {
    "train": "https://huggingface.co/datasets/asmaab/dreaddit/resolve/main/train_data.csv",
    "validation": "https://huggingface.co/datasets/asmaab/dreaddit/resolve/main/validation_data.csv",
    "test": "https://huggingface.co/datasets/asmaab/dreaddit/resolve/main/test-data.csv",
}


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, destination)
    print(f"Saved {destination}")


def prepare_dreaddit(force: bool = False) -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    frames = []
    for split, url in DREADDIT_URLS.items():
        split_path = RAW_DIR / f"dreaddit_{split}.csv"
        if force or not split_path.exists():
            download_file(url, split_path)
        frame = pd.read_csv(split_path)
        frame["split"] = split
        frames.append(frame)

    raw = pd.concat(frames, ignore_index=True)
    raw_path = RAW_DIR / "dreaddit_merged.csv"
    raw.to_csv(raw_path, index=False)

    if "text" not in raw.columns or "label" not in raw.columns:
        raise ValueError("Dreaddit file must contain 'text' and 'label' columns.")

    processed = raw[["text", "label"]].copy()
    processed["text"] = processed["text"].astype(str).str.strip()
    processed = processed[processed["text"].ne("")]
    processed["label"] = processed["label"].astype(int)
    processed["label_name"] = processed["label"].map({
        0: "normal",
        1: "stress_signal",
    })
    processed["source"] = "dreaddit"

    processed_path = PROCESSED_DIR / "mental_health_dataset.csv"
    processed.to_csv(processed_path, index=False)

    print("\nPrepared dataset")
    print(f"Rows: {len(processed)}")
    print(processed["label_name"].value_counts())
    print(f"Raw merged file: {raw_path}")
    print(f"Training-ready file: {processed_path}")
    return processed


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and prepare real-world NLP datasets.")
    parser.add_argument(
        "--dataset",
        default="dreaddit",
        choices=["dreaddit"],
        help="Dataset to download and prepare.",
    )
    parser.add_argument("--force", action="store_true", help="Re-download files even if they exist.")
    args = parser.parse_args()

    if args.dataset == "dreaddit":
        prepare_dreaddit(force=args.force)
        return 0

    raise ValueError(f"Unsupported dataset: {args.dataset}")


if __name__ == "__main__":
    sys.exit(main())
