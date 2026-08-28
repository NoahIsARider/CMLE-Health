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
from concurrent.futures import ThreadPoolExecutor

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
N_IMG_WORKERS = 8


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
    ap.add_argument("--pair-max-len", type=int, default=48, help="pair [q SEP opt] max length (cap = huge speedup; covers p95)")
    ap.add_argument("--limit", type=int, default=0, help="limit rows (debug)")
    ap.add_argument("--memory", action="store_true", help="load zip into RAM for parallel decode (HDD boxes)")
    ap.add_argument("--pair", action="store_true", help="encode options as [question SEP option] pairs (cross-encoder)")
    ap.add_argument("--hf-mirror", action="store_true")
    args = ap.parse_args()

    tag = args.tag or os.path.splitext(os.path.basename(args.csv))[0]
    if args.pair:
        tag = f"{tag}_pair"
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"{tag}.pt")

    df = load_csv(args.csv)
    if args.limit > 0:
        df = df.head(args.limit)
    N = len(df)
    print(f"[data] {tag}: {N} rows from {args.csv}")

    db = ZipImageDB([os.path.join(args.data_dir, "images.zip"),
                     os.path.join(args.data_dir, "images_2.zip")], memory=args.memory)
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
            with torch.autocast("cuda", dtype=torch.float16):
                qf = bert(**qtoks).pooler_output.float().cpu()          # (bs, 768)
        q_feats[i:i + bs] = qf.half()

        # ---- options: flatten (bs*4,) or pair-encode [q SEP opt_i] ----
        if args.pair:
            pair_texts = []
            for _, r in rows.iterrows():
                q = str(r["Question"]).strip()
                for c in CHOICES:
                    pair_texts.append(f"{q} [SEP] {str(r[f'Choice {c}']).strip()}")
            ptoks = tok(pair_texts, padding=True, truncation=True,
                        max_length=args.pair_max_len, return_tensors="pt").to(device)
            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.float16):
                    pf = bert(**ptoks).pooler_output.float().cpu()
            opt_feats[i:i + bs] = pf.view(bs, 4, DIM).half()
        else:
            opts = []
            for c in CHOICES:
                opts.extend(rows[f"Choice {c}"].astype(str).tolist())
            otoks = tok(opts, padding=True, truncation=True, max_length=args.max_opt,
                        return_tensors="pt").to(device)
            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.float16):
                    of = bert(**otoks).pooler_output.float().cpu()           # (bs*4, 768)
            opt_feats[i:i + bs] = of.view(bs, 4, DIM).half()

        # ---- images: parallel decode (thread-safe zipdb) + chunked batch resize ----
        # per-call CLIPImageProcessor overhead is ~100ms — NEVER call it per-image
        # in threads; chunked batch calls amortize it.
        paths = [str(p).strip() for p in rows["Figure_path"]]
        IMG_CHUNK = 16

        def load_chunk(chunk_paths):
            ims = [db.open_image(p) for p in chunk_paths]
            miss = sum(1 for im in ims if im is None)
            ims = [im if im is not None else Image.new("RGB", (224, 224), (0, 0, 0)) for im in ims]
            return proc(images=ims, return_tensors="pt")["pixel_values"], miss

        with ThreadPoolExecutor(max_workers=N_IMG_WORKERS) as ex:
            chunks = [paths[i:i + IMG_CHUNK] for i in range(0, len(paths), IMG_CHUNK)]
            results = list(ex.map(load_chunk, chunks))
        missing_img += sum(r[1] for r in results)
        px = torch.cat([r[0] for r in results]).to(device)
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.float16):
                iff = clip(pixel_values=px).pooler_output.float().cpu()  # (bs, 768)
        img_feats[i:i + bs] = iff.half()

        labels[i:i + bs] = torch.tensor([label_to_idx(r["Answer_label"]) for _, r in rows.iterrows()])

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
