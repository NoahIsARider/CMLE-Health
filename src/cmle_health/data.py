"""Data loading for MM-Health (EMNLP 2025 Findings).

split json schema (per source): {train|val|test: [{index, id, label(1=reliable,0=unreliable),
  text: {original, llama3.1, qwen2.5, chatglm4, gemma2, mistral},
  image: {original, flux, pag, sd15, sdxl, vae}, source, is_english}]}

Tasks:
  - Task A "reliability" : original content only, label = 1 (reliable) / 0 (unreliable)
  - Task B "originality" : every content variant becomes an instance; original = human (0), generators = AI (1)
"""

from __future__ import annotations

import json
import os
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

TEXT_KEYS = ["original", "llama3.1", "qwen2.5", "chatglm4", "gemma2", "mistral"]
IMAGE_KEYS = ["original", "flux", "pag", "sd15", "sdxl", "vae"]
AI_TEXT_KEYS = TEXT_KEYS[1:]
AI_IMAGE_KEYS = IMAGE_KEYS[1:]


def load_splits(path: str):
    with open(path) as f:
        raw = json.load(f)
    splits = {"train": [], "val": [], "test": []}
    for src in raw:
        for split in raw[src]:
            splits[split].extend(raw[src][split])
    return splits


class MMHealthDataset(Dataset):
    def __init__(
        self,
        samples: list,
        task: str = "reliability",          # "reliability" | "originality" | "both"
        modality: str = "text",             # "text" | "image" | "both"
        image_root: str = "",
        max_len: int = 384,
        tokenizer=None,
        image_processor=None,
        image_variant: str = "original",    # which image variant to use for reliability/both
    ):
        self.samples = samples
        self.task = task
        self.modality = modality
        self.image_root = image_root
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.image_variant = image_variant

        # build instance list
        self.instances = []
        for s in samples:
            if task in ("reliability", "both"):
                self.instances.append((s, "original", s["label"]))
            if task in ("originality", "both"):
                for k in TEXT_KEYS:
                    self.instances.append((s, k, 0 if k == "original" else 1))

    def __len__(self):
        return len(self.instances)

    def _load_text(self, s, key):
        text = s["text"].get(key)
        if not text:
            text = s["text"].get("original", "")
        return text

    def _load_image_path(self, s, key):
        paths = s["image"].get(key) or s["image"].get("original")
        if not paths:
            return None
        return os.path.join(self.image_root, paths[0])

    def __getitem__(self, i):
        s, key, label = self.instances[i]
        item = {"label": torch.tensor(label, dtype=torch.long)}

        if self.modality in ("text", "both"):
            enc = self.tokenizer(
                self._load_text(s, key),
                max_length=self.max_len,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            item["input_ids"] = enc["input_ids"].squeeze(0)
            item["attention_mask"] = enc["attention_mask"].squeeze(0)

        if self.modality in ("image", "both"):
            vkey = key if key in IMAGE_KEYS else self.image_variant
            path = self._load_image_path(s, vkey)
            if path and os.path.exists(path):
                try:
                    img = Image.open(path).convert("RGB")
                    px = self.image_processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
                    item["pixel_values"] = px
                except Exception:
                    item["pixel_values"] = torch.zeros((3, 224, 224))
            else:
                item["pixel_values"] = torch.zeros((3, 224, 224))
        return item


def collate_fn(batch):
    out = {}
    for k in batch[0]:
        if k == "label":
            out[k] = torch.stack([b[k] for b in batch])
        else:
            out[k] = torch.stack([b[k] for b in batch])
    return out
