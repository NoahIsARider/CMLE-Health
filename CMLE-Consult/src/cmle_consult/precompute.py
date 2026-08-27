"""Precompute frozen-backbone pooled features for PMC-VQA.

Reads images directly from the zip archives (no disk extraction), encodes:
  * question -> BERT-base [CLS] pooled  (N, 768)
  * options  -> BERT-base [CLS] pooled  (N, 4, 768)
  * image    -> CLIP ViT-B/32 pooled    (N, 768)
Saves one fp16 cache per split: {split}.pt  {img, q, opt, label}

Disk: ~2GB per split (vs ~21GB images) — fits the 30G P4 container.
"""

from __future__ import annotations

import argparse
import os
import time

import pandas as pd
import torch
from PIL import Image
from transformers import (
    AutoTokenizer,
    BertModel,
    CLIPImageProcessor,
    CLIPVisionModel,
)

from cmle_consult.data import CHOICES, ZipImageDB, load_csv

DIM = 768


def build_encoders(hf_mirror: bool):
    if hf_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # must be set before import actually;
    bert = BertModel.from_pretrained("bert-base-uncased").eval()
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    clip = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return bert, tok, clip, proc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="path to split csv (train.csv / test_clean.csv / ...)")
    ap.add_argument("--data-dir", default="/root/cmle-consult/data")
    ap.add_argument("--out", default="/root/cmle-consult/features")
    ap.add_argument("--tag", default=None, help="cache file name (default: csv basename without .csv)")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--max-q", type=int, default=64, help="max question tokens")
    ap.add_argument("--max-opt", type=int, default=32, help="max option tokens")
    ap.add_argument("--limit", type=int, default=0, help="limit rows (debug)")
    ap.add_argument("--hf-mirror", action="store_true")
    args = ap.parse_args()

    tag = args.tag or os.path.splitext(os.path.basename(args.csv))[0]
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"{tag}.pt")

    df = load_csv(args.csv)
    if args.limit > 0:
        df = df.head(args.limit)
    N = len(df)
    print(f"[data] {tag}: {N} rows from {args.csv}")

    db = ZipImageDB([os.path.join(args.data_dir, "images.zip"),
                     os.path.join(args.data_dir, "images_2.zip")])
    bert, tok, clip, proc = build_encoders(args.hf_mirror)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert.to(device); clip.to(device)

    img_feats = torch.zeros(N, DIM, dtype=torch.float16)
    q_feats = torch.zeros(N, DIM, dtype=torch.float16)
    opt_feats = torch.zeros(N, 4, DIM, dtype=torch.float16)
    labels = torch.zeros(N, dtype=torch.long)
    missing_img = 0

    t0 = time.time()
    for i in range(0, N, args.batch):
        rows = df.iloc[i:i + args.batch]
        bs = len(rows)

        # ---- questions ----
        qtoks = tok(list(rows["Question"].astype(str)), padding=True, truncation=True,
                    max_length=args.max_q, return_tensors="pt").to(device)
        with torch.no_grad():
            qf = bert(**qtoks).pooler_output.float().cpu()          # (bs, 768)
        q_feats[i:i + bs] = qf.half()

        # ---- options: flatten (bs*4,) ----
        opts = []
        for c in CHOICES:
            opts.extend(rows[f"Choice {c}"].astype(str).tolist())
        otoks = tok(opts, padding=True, truncation=True, max_length=args.max_opt,
                    return_tensors="pt").to(device)
        with torch.no_grad():
            of = bert(**otoks).pooler_output.float().cpu()           # (bs*4, 768)
        opt_feats[i:i + bs] = of.view(bs, 4, DIM).half()

        # ---- images ----
        imgs = []
        for p in rows["Figure_path"]:
            im = db.open_image(str(p).strip())
            if im is None:
                missing_img += 1
                im = Image.new("RGB", (224, 224), (0, 0, 0))
            imgs.append(im)
        px = proc(images=imgs, return_tensors="pt")["pixel_values"].to(device)
        with torch.no_grad():
            iff = clip(pixel_values=px).pooler_output.float().cpu()  # (bs, 768)
        img_feats[i:i + bs] = iff.half()

        labels[i:i + bs] = torch.tensor([label_to_idx(r) for _, r in rows.iterrows()])

        el = time.time() - t0
        print(f"[precompute {tag}] {min(i + bs, N)}/{N}  ({el:.0f}s, ~{el / max(min(i + bs, N), 1) * (N - i - bs):.0f}s left)")

    print(f"[precompute {tag}] missing images: {missing_img}/{N}")
    torch.save({"img": img_feats, "q": q_feats, "opt": opt_feats, "label": labels}, out_path)
    sz = os.path.getsize(out_path) / 1e6
    print(f"[precompute {tag}] saved {out_path} ({sz:.0f} MB)")
    db.close()


def label_to_idx(label) -> int:
    s = str(label).strip().upper()
    return CHOICES.index(s[0]) if s and s[0] in CHOICES else 0


if __name__ == "__main__":
    main()
