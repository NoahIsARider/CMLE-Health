"""Train / evaluate ConsultNet on precomputed PMC-VQA features.

Task: 4-choice closed-set medical VQA. Headline metric: accuracy on the
test split; secondary: accuracy @ coverage (HRO referral curve).

Variants:
  bert-only  : MLP([q, opt_i])                      (text-only baseline)
  clip-only  : MLP([img, opt_i])                    (image-only baseline)
  concat     : MLP([q, img, opt_i])                 (no experts / no DGM)
  full       : ConsultNet (experts + DGM + MU + MIM)
  w-o-dgm    : ConsultNet, fixed equal expert weights
  w-o-mu     : ConsultNet, no consensus loss (lambda-mu = 0)
  w-o-univ   : ConsultNet without the universal expert
  w-o-spec   : ConsultNet without the specialized (pathologist) expert
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset, Subset

from cmle_consult.model import ConsultNet, RoleExpert

SEED = 42


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------- dataset
class FeatDS(Dataset):
    """Precomputed feature cache: img (N,768), q (N,768), opt (N,4,768), label (N,)."""

    def __init__(self, path: str):
        c = torch.load(path, map_location="cpu")
        self.img = c["img"]
        self.q = c["q"]
        self.opt = c["opt"]
        self.label = c["label"]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, i):
        return {"img": self.img[i], "q": self.q[i], "opt": self.opt[i], "label": self.label[i]}


def collate(batch):
    return {
        "img": torch.stack([b["img"] for b in batch]).float(),
        "q": torch.stack([b["q"] for b in batch]).float(),
        "opt": torch.stack([b["opt"] for b in batch]).float(),
        "label": torch.stack([b["label"] for b in batch]),
    }


# ---------------------------------------------------------------- models
class FlatMLP(nn.Module):
    """Single-net baseline: (B, 4, in_dim) -> (B, 4)."""

    def __init__(self, in_dim: int, hidden: int = 256, n_opts: int = 4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(hidden, n_opts))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(args, device):
    if args.variant in ("bert-only", "clip-only", "concat"):
        if args.variant == "bert-only":
            in_dim = 768 * 2          # [q, opt_i]
        elif args.variant == "clip-only":
            in_dim = 768 * 2          # [img, opt_i]
        else:
            in_dim = 768 * 3          # [q, img, opt_i]
        model = FlatMLP(in_dim).to(device)
    else:
        model = ConsultNet(
            use_universal=args.variant != "w-o-univ",
            use_specialized=args.variant != "w-o-spec",
            use_dgm=args.variant != "w-o-dgm",
        ).to(device)
    return model


def forward_baseline(model, variant, batch, device):
    q = batch["q"].to(device); img = batch["img"].to(device); opt = batch["opt"].to(device)
    B = q.size(0)
    qe = q.unsqueeze(1).expand(B, 4, -1)
    imge = img.unsqueeze(1).expand(B, 4, -1)
    if variant == "bert-only":
        return model(torch.cat([qe, opt], -1))
    if variant == "clip-only":
        return model(torch.cat([imge, opt], -1))
    return model(torch.cat([qe, imge, opt], -1))       # concat


def compute_loss(model, args, batch, device):
    """Returns (loss, logits, label)."""
    label = batch["label"].to(device)
    if isinstance(model, FlatMLP):
        logits = forward_baseline(model, args.variant, batch, device)
        return F.cross_entropy(logits, label), logits, label

    q = batch["q"].to(device); img = batch["img"].to(device); opt = batch["opt"].to(device)
    out = model(q, img, opt)
    loss = F.cross_entropy(out["logits"], label)
    if args.lambda_mu > 0:
        loss = loss + args.lambda_mu * model.consensus_loss(out["expert_probs"])
    if args.lambda_mim > 0:
        loss = loss + args.lambda_mim * model.infonce_loss(out["rep_q"], out["rep_img"])
    return loss, out["logits"], label


# ---------------------------------------------------------------- eval
def evaluate(model, loader, device, args, coverage_levels=(1.0, 0.95, 0.9, 0.8, 0.7)):
    model.eval()
    all_logits, all_labels, all_agr = [], [], []
    with torch.no_grad():
        for batch in loader:
            label = batch["label"].to(device)
            if isinstance(model, FlatMLP):
                logits = forward_baseline(model, args.variant, batch, device)
                agr = torch.ones(len(label), device=device)
            else:
                q = batch["q"].to(device); img = batch["img"].to(device); opt = batch["opt"].to(device)
                out = model(q, img, opt)
                logits = out["logits"]
                agr = out["agreement"]
            all_logits.append(logits.cpu())
            all_labels.append(label.cpu())
            all_agr.append(agr.cpu())
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    agr = torch.cat(all_agr)
    probs = torch.softmax(logits, -1)
    conf, pred = probs.max(-1)
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum(-1)

    res = {"acc": accuracy_score(labels.numpy(), pred.numpy()),
           "mean_entropy": float(entropy.mean()),
           "mean_agreement": float(agr.mean())}
    # HRO referral curve: keep the most confident (1-cov) fraction, report acc
    for cov in coverage_levels:
        k = max(int(len(labels) * cov), 1)
        keep = torch.argsort(entropy)[:k]
        res[f"acc@{cov:.0%}cov"] = round(accuracy_score(labels[keep].numpy(), pred[keep].numpy()), 4)
    res["acc"] = round(res["acc"], 4)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", default="/root/cmle-consult/features")
    ap.add_argument("--train", default="train.pt")
    ap.add_argument("--test", default="test_clean.pt")
    ap.add_argument("--variant", default="full",
                    choices=["bert-only", "clip-only", "concat", "full",
                             "w-o-dgm", "w-o-mu", "w-o-univ", "w-o-spec"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-mu", type=float, default=0.1)
    ap.add_argument("--lambda-mim", type=float, default=0.1)
    ap.add_argument("--val-frac", type=float, default=0.1, help="hold-out from train for model selection")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default="/root/cmle-consult/runs")
    ap.add_argument("--tag", default="exp")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--limit", type=int, default=0, help="limit train rows (debug / MVP)")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device} seed={args.seed} variant={args.variant}")

    train_ds = FeatDS(os.path.join(args.features_dir, args.train))
    test_ds = FeatDS(os.path.join(args.features_dir, args.test))
    if args.limit > 0:
        train_ds = Subset(train_ds, range(min(args.limit, len(train_ds))))
    n_train = len(train_ds)
    n_val = max(int(n_train * args.val_frac), 1)
    rng = random.Random(args.seed)
    idx = list(range(n_train))
    rng.shuffle(idx)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    val_loader = DataLoader(Subset(train_ds, val_idx), batch_size=args.batch, shuffle=False, collate_fn=collate)
    train_loader = DataLoader(Subset(train_ds, tr_idx), batch_size=args.batch, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, collate_fn=collate)
    print(f"[data] train={len(tr_idx)} val={len(val_idx)} test={len(test_ds)}")

    model = build_model(args, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {type(model).__name__} trainable params: {n_params:,}")

    if args.eval_only:
        if args.ckpt:
            model.load_state_dict(torch.load(args.ckpt, map_location=device))
            print(f"[ckpt] loaded {args.ckpt}")
        res = evaluate(model, test_loader, device, args)
        print("[test]", json.dumps(res))
        return

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        tot = 0.0
        for batch in train_loader:
            loss, _, _ = compute_loss(model, args, batch, device)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        val = evaluate(model, val_loader, device, args)
        if val["acc"] > best_val:
            best_val = val["acc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"[epoch {epoch}] loss={tot:.3f} val={json.dumps(val)} ({time.time() - t0:.0f}s)")

    model.load_state_dict(best_state)
    test = evaluate(model, test_loader, device, args)
    print("[test]", json.dumps(test))

    ckpt = os.path.join(args.out, f"{args.tag}.pt")
    torch.save(best_state, ckpt)
    meta = {"args": vars(args), "val_best": best_val, "test": test,
            "trainable_params": n_params, "variant": args.variant}
    with open(os.path.join(args.out, f"{args.tag}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[saved] {ckpt} + meta")


if __name__ == "__main__":
    main()
