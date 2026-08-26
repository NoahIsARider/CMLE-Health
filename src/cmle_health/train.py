"""Training / evaluation entry point for CMLE-Health on MM-Health."""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    BertModel,
    CLIPImageProcessor,
    CLIPVisionModel,
)

from cmle_health.data import MMHealthDataset, collate_fn, load_splits
from cmle_health.model import BaselineMLP, CMLEHealth

SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/root/mm-health-data/train_test_splited_data.json")
    p.add_argument("--image-root", default="/root/mm-health-data")
    p.add_argument("--task", default="both", choices=["reliability", "originality", "both"])
    p.add_argument("--modality", default="both", choices=["text", "image", "both"])
    p.add_argument("--variant", default="full",
                   choices=["full", "w-o-universal", "w-o-specialized", "w-o-dgm", "w-o-mim",
                            "no-experts", "bert-only", "clip-only", "concat"])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lambda-mim", type=float, default=1.0)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--proj-dim", type=int, default=512)
    p.add_argument("--max-len", type=int, default=384)
    p.add_argument("--no-consistency", action="store_true")
    p.add_argument("--out", default="/root/cmle-health/runs")
    p.add_argument("--tag", default="exp")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--limit", type=int, default=0, help="limit train samples (debug)")
    p.add_argument("--hf-mirror", action="store_true", help="use HF_ENDPOINT mirror")
    return p.parse_args()


def build_encoders(use_text=True, use_image=True, hf_mirror=False):
    if hf_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    encoders = {}
    if use_text:
        encoders["text_model"] = BertModel.from_pretrained("bert-base-uncased").eval()
        encoders["tokenizer"] = AutoTokenizer.from_pretrained("bert-base-uncased")
    if use_image:
        encoders["image_model"] = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-base-patch32").eval()
        encoders["image_processor"] = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-base-patch32")
    return encoders


@torch.no_grad()
def extract_features(batch, encoders, device, modality):
    if modality in ("text", "both"):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        text_feats = encoders["text_model"](input_ids=ids, attention_mask=mask).last_hidden_state
    else:
        text_feats = None
    if modality in ("image", "both"):
        px = batch["pixel_values"].to(device)
        img_feats = encoders["image_model"](pixel_values=px).last_hidden_state
    else:
        img_feats = None
    return text_feats, img_feats


def baseline_forward(model, variant, tf, imf):
    """Pool frozen features and forward through BaselineMLP."""
    if variant in ("concat", "no-experts"):
        parts = []
        if tf is not None:
            parts.append(tf.mean(1))
        if imf is not None:
            parts.append(imf.mean(1))
        x = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
    elif variant == "bert-only":
        x = tf.mean(1)
    else:  # clip-only
        x = imf.mean(1)
    return model(x)


def evaluate(model, loader, encoders, device, modality, task, criterion, variant=None):
    model.eval()
    res = {}
    losses = []
    preds = {"a": [], "b": []}
    gts = {"a": [], "b": []}
    with torch.no_grad():
        for batch in loader:
            tf, imf = extract_features(batch, encoders, device, modality)
            out = model(tf, imf)
            if isinstance(model, BaselineMLP):
                out = baseline_forward(model, variant, tf, imf)
            la = batch["label_a"].to(device)
            lb = batch["label_b"].to(device)
            if isinstance(out, dict):
                logits_a, logits_b = out["logits_a"], out["logits_b"]
            else:
                logits_a = logits_b = out
            if task in ("reliability", "both"):
                losses.append(criterion(logits_a, la).item())
                preds["a"].extend(logits_a.argmax(-1).cpu().tolist())
                gts["a"].extend(la.cpu().tolist())
            if task in ("originality", "both"):
                losses.append(criterion(logits_b, lb).item())
                preds["b"].extend(logits_b.argmax(-1).cpu().tolist())
                gts["b"].extend(lb.cpu().tolist())
    res["loss"] = float(np.mean(losses)) if losses else 0.0
    for h in ("a", "b"):
        if preds[h]:
            res[f"{h}_acc"] = accuracy_score(gts[h], preds[h])
            res[f"{h}_f1"] = f1_score(gts[h], preds[h], average="macro", zero_division=0)
            res[f"{h}_p"] = precision_score(gts[h], preds[h], average="macro", zero_division=0)
            res[f"{h}_r"] = recall_score(gts[h], preds[h], average="macro", zero_division=0)
    return res


def main():
    args = parse_args()
    set_seed()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device} variant={args.variant} task={args.task} modality={args.modality}")

    if args.variant in ("bert-only", "clip-only", "concat") and args.task == "both":
        raise SystemExit("baseline variants support single task only (reliability | originality)")

    splits = load_splits(args.data)
    train_samples, val_samples, test_samples = splits["train"], splits["val"], splits["test"]
    if args.limit > 0:
        train_samples = train_samples[: args.limit]
        val_samples = val_samples[: max(args.limit // 4, 50)]
        test_samples = test_samples[: max(args.limit // 4, 50)]
    print(f"[data] train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}")

    use_text = args.modality in ("text", "both")
    use_image = args.modality in ("image", "both")
    encoders = build_encoders(use_text, use_image, args.hf_mirror)
    for k, v in encoders.items():
        if hasattr(v, "to"):
            v.to(device)

    tokenizer = encoders.get("tokenizer")
    image_processor = encoders.get("image_processor")
    make_ds = lambda samples: MMHealthDataset(
        samples, task=args.task, modality=args.modality,
        image_root=args.image_root, max_len=args.max_len,
        tokenizer=tokenizer, image_processor=image_processor,
    )
    train_ds = make_ds(train_samples)
    val_ds = make_ds(val_samples)
    test_ds = make_ds(test_samples)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=4, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=4, collate_fn=collate_fn)

    num_a, num_b = 2, 2
    if args.variant in ("bert-only", "clip-only", "concat", "no-experts"):
        if args.variant == "concat" or args.variant == "no-experts":
            in_dim = 768 * 2 if use_image else 768
        else:
            in_dim = 768
        model = BaselineMLP(in_dim, num_b if args.task == "originality" else num_a)
    else:
        model = CMLEHealth(
            text_dim=768, image_dim=768, proj_dim=args.proj_dim,
            lora_rank=args.lora_rank,
            num_classes_a=num_a, num_classes_b=num_b,
            use_consistency=not args.no_consistency,
            use_universal=args.variant != "w-o-universal",
            use_specialized=args.variant != "w-o-specialized",
            use_dgm=args.variant != "w-o-dgm",
            use_mim=args.variant != "w-o-mim",
        )
    model.to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {type(model).__name__} trainable params: {n_train:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.eval_only:
        assert args.ckpt
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        res = evaluate(model, test_loader, encoders, device, args.modality, args.task, criterion, args.variant)
        print("[test]", json.dumps(res))
        return

    best_val, best_state = -1, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        tot = 0.0
        for batch in train_loader:
            tf, imf = extract_features(batch, encoders, device, args.modality)
            la = batch["label_a"].to(device)
            lb = batch["label_b"].to(device)
            out = model(tf, imf)
            if isinstance(model, BaselineMLP):
                out = baseline_forward(model, args.variant, tf, imf)

            if isinstance(out, dict):
                logits_a, logits_b = out["logits_a"], out["logits_b"]
            else:
                logits_a = logits_b = out
            loss = torch.tensor(0.0, device=device)
            if args.task in ("reliability", "both"):
                loss = loss + criterion(logits_a, la)
            if args.task in ("originality", "both"):
                loss = loss + criterion(logits_b, lb)
            if isinstance(model, CMLEHealth):
                loss = loss + args.lambda_mim * model.infonce_loss(out["reps"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tot += loss.item()

        val = evaluate(model, val_loader, encoders, device, args.modality, args.task, criterion, args.variant)
        key = "b_f1" if args.task == "originality" else ("a_f1" if args.task == "reliability" else "a_f1")
        if val[key] > best_val:
            best_val = val[key]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"[epoch {epoch}] loss={tot:.3f} val={json.dumps({k: round(v, 4) for k, v in val.items()})} "
              f"({time.time() - t0:.0f}s)")

    model.load_state_dict(best_state)
    test = evaluate(model, test_loader, encoders, device, args.modality, args.task, criterion, args.variant)
    print("[test]", json.dumps(test))

    ckpt = os.path.join(args.out, f"{args.tag}.pt")
    torch.save(best_state, ckpt)
    meta = {"args": vars(args), "val_best": best_val, "test": test,
            "trainable_params": n_train}
    with open(os.path.join(args.out, f"{args.tag}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[saved] {ckpt} + meta")


if __name__ == "__main__":
    main()
