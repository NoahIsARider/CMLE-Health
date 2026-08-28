"""Analyze ConsultNet gate behavior on test set: gate weight distribution + per-expert accuracy.

Usage: python3 analyze_gate.py --ckpt /root/cmle-consult/runs-final/f_full.pt
"""
import argparse
import sys
import torch
from cmle_consult.model import ConsultNet

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--features-dir", default="/root/cmle-consult/features")
    ap.add_argument("--split", default="test_clean_pair.pt")
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    meta = ckpt.get("meta", {}) if isinstance(ckpt, dict) else {}
    print("meta keys:", list(meta.keys()) if isinstance(meta, dict) else "-")
    print("meta:", {k: meta[k] for k in list(meta)[:6]} if isinstance(meta, dict) else "-")

    model = ConsultNet().cuda()
    model.load_state_dict(sd)
    model.eval()

    d = torch.load(f"{args.features_dir}/{args.split}", map_location="cpu", weights_only=True)
    img, q, opt, label = d["img"], d["q"], d["opt"], d["label"]
    N = q.size(0)

    gate_sum = torch.zeros(model.n_experts, device="cuda")
    gate_cnt = 0
    expert_correct = torch.zeros(model.n_experts, device="cuda")
    expert_total = torch.zeros(model.n_experts, device="cuda")
    fused_correct = 0
    agree = 0.0
    gate_ent = 0.0
    n_batches = 0

    with torch.no_grad():
        for i in range(0, N, args.batch):
            bq = q[i:i+args.batch].float().cuda(); bi = img[i:i+args.batch].float().cuda()
            bo = opt[i:i+args.batch].float().cuda(); bl = label[i:i+args.batch].cuda()
            out = model(bq, bi, bo)
            w = out["gate_w"]                                   # (B, n_exp)
            ep = out["expert_probs"]                            # (B, n_exp, 4)
            pred_exp = ep.argmax(-1)                            # (B, n_exp)
            pred_fused = out["logits"].argmax(-1)
            fused_correct += (pred_fused == bl).sum().item()
            for e in range(model.n_experts):
                expert_correct[e] += (pred_exp[:, e] == bl).sum().item()
                expert_total[e] += bl.size(0)
            gate_sum += w.sum(0)
            gate_cnt += bl.size(0)
            agree += out["agreement"].sum().item()
            gate_ent += (-(w * w.clamp_min(1e-8).log()).sum(-1)).sum().item()
            n_batches += 1

    print(f"\nsplit={args.split} N={N}")
    print(f"fused acc: {fused_correct / N:.4f}")
    print(f"expert names: {out['expert_names']}")
    for e in range(model.n_experts):
        print(f"  expert[{e}] {out['expert_names'][e]}: acc {expert_correct[e].item() / expert_total[e].item():.4f}")
    print(f"mean gate_w: {[round(x, 4) for x in (gate_sum / gate_cnt).tolist()]}")
    print(f"mean gate entropy: {gate_ent / N:.4f} (0=one-hot, {model.n_experts}-way uniform={__import__('math').log(model.n_experts):.4f})")
    print(f"mean agreement: {agree / N:.4f}")

if __name__ == "__main__":
    main()
