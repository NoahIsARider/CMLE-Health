"""PMC-VQA data loading + zip-backed image access (no extraction on disk)."""

from __future__ import annotations

import io
import os
import zipfile

import numpy as np
import pandas as pd
import torch
from PIL import Image

CHOICES = ["A", "B", "C", "D"]

REQUIRED_COLS = ["Figure_path", "Question", "Answer", "Choice A", "Choice B",
                 "Choice C", "Choice D", "Answer_label"]


def load_csv(path: str) -> pd.DataFrame:
    """Load a PMC-VQA CSV and normalize columns."""
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns in {path}: {missing}")
    # drop rows with any missing choice / question / label
    df = df.dropna(subset=REQUIRED_COLS).copy()
    df = df[df["Figure_path"].astype(str).str.strip() != ""]
    df = df.reset_index(drop=True)
    return df


def label_to_idx(label) -> int:
    s = str(label).strip().upper()
    return CHOICES.index(s[0]) if s and s[0] in CHOICES else 0


def options_for(row) -> list[str]:
    return [str(row[f"Choice {c}"]).strip() for c in CHOICES]


class ZipImageDB:
    """Read images from zip archives by basename, without extracting."""

    def __init__(self, zip_paths: list[str]):
        self.zips = [zipfile.ZipFile(p) for p in zip_paths if os.path.exists(p)]
        if not self.zips:
            raise FileNotFoundError(f"no zip archives found in: {zip_paths}")
        self.index: dict[str, tuple[int, str]] = {}
        for zi, zf in enumerate(self.zips):
            for name in zf.namelist():
                self.index[os.path.basename(name)] = (zi, name)
        print(f"[zipdb] {len(self.index)} unique images across {len(self.zips)} zips")

    def read(self, fname: str):
        """Return raw image bytes, or None if the image is missing."""
        hit = self.index.get(os.path.basename(fname))
        if hit is None:
            return None
        zi, entry = hit
        try:
            return self.zips[zi].read(entry)
        except Exception:
            return None

    def open_image(self, fname: str):
        raw = self.read(fname)
        if raw is None:
            return None
        try:
            return Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            return None

    def close(self):
        for z in self.zips:
            z.close()
