"""VLM zero-shot baseline for CMLE-Consult on PMC-VQA test_clean.

Reads test_clean.csv + images from the zip archive, calls a vision-language
model (OpenAI-compatible API), asks for the correct choice among A-D,
and reports closed-set accuracy (+ by-image-domain breakdown is a TODO).

Run on the server (needs PIL + requests):
    export DEEPSEEK_API_KEY=...  (from workspace/.secrets/deepseek.env)
    python3 -m cmle_consult.eval_vlm --csv /root/cmle-consult/data/test_clean.csv \
        --data-dir /root/cmle-consult/data --limit 300 --out /root/cmle-consult/runs
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import random
import time
import urllib.request

from PIL import Image

from cmle_consult.data import CHOICES, ZipImageDB

API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-v4-flash-vision-exp"

PROMPT = (
    "You are a medical imaging assistant. Given the medical image and the "
    "question, select the single best answer among the four options. "
    "Reply with ONLY the letter (A, B, C, or D).\n"
    "Question: {question}\n"
    "A: {a}\nB: {b}\nC: {c}\nD: {d}\n"
)


def call_vlm(api_key: str, image_bytes: bytes, prompt: str, timeout: int = 60) -> str:
    b64 = base64.b64encode(image_bytes).decode()
    body = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": prompt},
            ],
        }],
        "temperature": 0.0,
        "max_tokens": 8,
    }
    req = urllib.request.Request(
        API_URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    return data["choices"][0]["message"]["content"].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/root/cmle-consult/data/test_clean.csv")
    ap.add_argument("--data-dir", default="/root/cmle-consult/data")
    ap.add_argument("--limit", type=int, default=300, help="0 = all rows")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="/root/cmle-consult/runs")
    ap.add_argument("--tag", default="vlm_zeroshot")
    args = ap.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY not set (see workspace/.secrets/deepseek.env)")

    rows = list(csv.DictReader(open(args.csv)))
    if args.limit > 0:
        rng = random.Random(args.seed)
        rows = rng.sample(rows, min(args.limit, len(rows)))
    db = ZipImageDB([os.path.join(args.data_dir, "images.zip")])

    correct = total = 0
    errs = []
    t0 = time.time()
    for i, row in enumerate(rows):
        img = db.open_image(row["Figure_path"].strip())
        if img is None:
            errs.append({"row": i, "err": "image missing"})
            continue
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=85)
        prompt = PROMPT.format(question=row["Question"].strip(),
                               a=row["Choice A"].strip(), b=row["Choice B"].strip(),
                               c=row["Choice C"].strip(), d=row["Choice D"].strip())
        try:
            ans = call_vlm(api_key, buf.getvalue(), prompt)
        except Exception as e:                      # rate limit / network — retry once
            time.sleep(5)
            try:
                ans = call_vlm(api_key, buf.getvalue(), prompt)
            except Exception as e2:
                errs.append({"row": i, "err": str(e2), "figure": row["Figure_path"]})
                continue
        letter = ans.strip().upper()[:1]
        gold = str(row["Answer_label"]).strip().upper()[:1]
        ok = letter == gold
        correct += ok
        total += 1
        if (i + 1) % 25 == 0 or not ok:
            print(f"[{i+1}/{len(rows)}] pred={letter} gold={gold} {'OK' if ok else 'X'} "
                  f"({time.time()-t0:.0f}s, acc so far {correct/max(total,1):.3f})")

    acc = correct / max(total, 1)
    print(f"\n=== VLM zero-shot acc: {acc:.4f} ({correct}/{total}) ===")
    os.makedirs(args.out, exist_ok=True)
    meta = {"tag": args.tag, "model": MODEL, "n": total, "acc": round(acc, 4),
            "errors": errs[:20], "n_errors": len(errs)}
    with open(os.path.join(args.out, f"{args.tag}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[saved] {args.out}/{args.tag}.json")


if __name__ == "__main__":
    main()
