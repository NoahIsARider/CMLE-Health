"""Precompute frozen-backbone features for MM-Health instances and cache to disk (fp16).

Output: <out>/<split>_<modality>.pt containing:
  - "feats": float16 tensor (N, L, D)  (text: L=max_len; image: L=50 patches)
  - "label_a": long tensor (N,)
  - "label_b": long tensor (N,)
  - "meta": list of (index, variant_key)
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
from transformers import AutoTokenizer, BertModel, CLIPImageProcessor, CLIPVisionModel

from cmle_health.data import IMAGE_KEYS, TEXT_KEYS, load_splits


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/root/mm-health-data/train_test_splited_data.json")
    p.add_argument("--image-root", default="/root/mm-health-data")
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--modality", default="text", choices=["text", "image", "both"])
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", default="/root/cmle-health/features")
    p.add_argument("--hf-mirror", action="store_true")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    if args.hf_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    splits = load_splits(args.data)
    samples = splits[args.split]
    if args.limit > 0:
        samples = samples[: args.limit]
    print(f"[precompute] split={args.split} modality={args.modality} samples={len(samples)}")

    tokenizer = None
    image_processor = None
    if args.modality in ("text", "both"):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        text_model = BertModel.from_pretrained("bert-base-uncased").to(device).eval()
    if args.modality in ("image", "both"):
        image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        image_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()

    keys = IMAGE_KEYS if args.modality == "image" else TEXT_KEYS
    instances = [(s, k) for s in samples for k in keys]
    N = len(instances)
    print(f"[precompute] instances={N}")

    if args.modality in ("text", "both"):
        L = args.max_len
        D = 768
    else:
        L = 50  # CLIP ViT-B/32 patches
        D = 768
    feats = torch.zeros(N, L, D, dtype=torch.float16)
    label_a = torch.zeros(N, dtype=torch.long)
    label_b = torch.zeros(N, dtype=torch.long)
    meta = []

    t0 = time.time()
    for i in range(0, N, args.batch):
        chunk = instances[i : i + args.batch]
        bs = len(chunk)
        if args.modality in ("text", "both"):
            texts = [s["text"].get(k) or s["text"].get("original", "") for s, k in chunk]
            enc = tokenizer(texts, max_length=args.max_len, truncation=True,
                            padding="max_length", return_tensors="pt")
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            h = text_model(input_ids=ids, attention_mask=mask).last_hidden_state  # (bs, L, 768)
        else:
            h = None
        if args.modality == "both":
            # align image key by index for each text key
            imgs = []
            for s, k in chunk:
                ikey = k if k in IMAGE_KEYS else IMAGE_KEYS[TEXT_KEYS.index(k)]
                paths = s["image"].get(ikey) or s["image"].get("original")
                imgs.append(os.path.join(args.image_root, paths[0]) if paths else None)
            from PIL import Image
            px_list = []
            for p in imgs:
                if p and os.path.exists(p):
                    try:
                        px_list.append(image_processor(images=Image.open(p).convert("RGB"),
                                                       return_tensors="pt")["pixel_values"].squeeze(0))
                    except Exception:
                        px_list.append(torch.zeros(3, 224, 224))
                else:
                    px_list.append(torch.zeros(3, 224, 224))
            px = torch.stack(px_list).to(device)
            hi = image_model(pixel_values=px).last_hidden_state  # (bs, 50, 768)
            h = torch.cat([h, hi], dim=1)  # (bs, L+50, 768) -- only used for 'both' storage
        elif args.modality == "image":
            from PIL import Image
            px_list = []
            for s, k in chunk:
                paths = s["image"].get(k) or s["image"].get("original")
                p = os.path.join(args.image_root, paths[0]) if paths else None
                if p and os.path.exists(p):
                    try:
                        px_list.append(image_processor(images=Image.open(p).convert("RGB"),
                                                       return_tensors="pt")["pixel_values"].squeeze(0))
                    except Exception:
                        px_list.append(torch.zeros(3, 224, 224))
                else:
                    px_list.append(torch.zeros(3, 224, 224))
            px = torch.stack(px_list).to(device)
            h = image_model(pixel_values=px).last_hidden_state

        feats[i : i + bs] = h.half().cpu()
        for j, (s, k) in enumerate(chunk):
            label_a[i + j] = s["label"]
            label_b[i + j] = 0 if k == "original" else 1
            meta.append({"index": s.get("index"), "key": k})

        if (i // args.batch) % 50 == 0:
            print(f"  [{i}/{N}] {time.time()-t0:.0f}s")

    torch.save({"feats": feats, "label_a": label_a, "label_b": label_b, "meta": meta,
               "max_len": args.max_len, "modality": args.modality},
               os.path.join(args.out, f"{args.split}_{args.modality}.pt"))
    print(f"[precompute] saved {os.path.join(args.out, f'{args.split}_{args.modality}.pt')} "
          f"feats={tuple(feats.shape)} {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
