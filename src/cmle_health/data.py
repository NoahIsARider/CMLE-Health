"""Data loading for MM-Health (EMNLP 2025 Findings).

split json schema (per source): {train|val|test: [{index, id, label(1=reliable,0=unreliable),
  text: {original, llama3.1, qwen2.5, chatglm4, gemma2, mistral},
  image: {original, flux, pag, sd15, sdxl, vae}, source, is_english}]}

Tasks (both heads trained jointly when task="both"):
  - label_a (reliability): 1 = reliable / 0 = unreliable  (inherited from source article)
  - label_b (originality): 0 = human-generated / 1 = AI-generated

Every sample expands into content variants:
  - text modality    : original + 5 LLM texts (6 variants)
  - image modality   : original + 5 gen images (6 variants)
  - both modalities  : index-aligned pairs (original/original, llama3.1/flux, ...) -> 6 variants
"""

from __future__ import annotations

import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset

TEXT_KEYS = ["original", "llama3.1", "qwen2.5", "chatglm4", "gemma2", "mistral"]
IMAGE_KEYS = ["original", "flux", "pag", "sd15", "sdxl", "vae"]
# index-aligned pairing for modality="both" (text key i <-> image key i)
PAIRED_IMAGE_KEYS = IMAGE_KEYS  # same order, aligned by index


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
        task: str = "both",                 # "reliability" | "originality" | "both"
        modality: str = "text",             # "text" | "image" | "both"
        image_root: str = "",
        max_len: int = 384,
        tokenizer=None,
        image_processor=None,
    ):
        self.samples = samples
        self.task = task
        self.modality = modality
        self.image_root = image_root
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.image_processor = image_processor

        # build instances: (sample, variant_key)
        self.instances = []
        for s in samples:
            if modality == "image":
                keys = IMAGE_KEYS
            elif modality == "both":
                keys = TEXT_KEYS  # paired with aligned image key
            else:
                keys = TEXT_KEYS
            for k in keys:
                self.instances.append((s, k))

    def __len__(self):
        return len(self.instances)

    def _load_text(self, s, key):
        text = s["text"].get(key)
        if not text:
            text = s["text"].get("original", "")
        return text

    def _image_key(self, key):
        """Map a variant key to the aligned image key."""
        if key in IMAGE_KEYS:
            return key
        idx = TEXT_KEYS.index(key) if key in TEXT_KEYS else 0
        return IMAGE_KEYS[min(idx, len(IMAGE_KEYS) - 1)]

    def _load_image_path(self, s, key):
        ikey = self._image_key(key)
        paths = s["image"].get(ikey) or s["image"].get("original")
        if not paths:
            return None
        return os.path.join(self.image_root, paths[0])

    def __getitem__(self, i):
        s, key = self.instances[i]
        label_a = torch.tensor(s["label"], dtype=torch.long)          # reliability
        label_b = torch.tensor(0 if key == "original" else 1, dtype=torch.long)  # originality
        item = {"label_a": label_a, "label_b": label_b}

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
            path = self._load_image_path(s, key)
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
        out[k] = torch.stack([b[k] for b in batch])
    return out
