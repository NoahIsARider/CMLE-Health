#!/usr/bin/env python3
"""CoAID (COVID-19 health misinformation, Cui & Lee 2020) -> CMLE-Health eval JSON.

Uses the largest snapshot (05-01-2020): NewsFakeCOVID-19.csv + NewsRealCOVID-19.csv.
Label alignment with MM-Health: 1 = reliable (real news), 0 = unreliable (fake news).
Each article becomes one instance with text.original = title (CoAID has no full text).
"""
import csv
import json

BASE = "/root/coaid/05-01-2020"
FAKE = f"{BASE}/NewsFakeCOVID-19.csv"
REAL = f"{BASE}/NewsRealCOVID-19.csv"
OUT = "/root/coaid_news.json"


def read(path, label):
    out = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for row in csv.DictReader(f):
            title = (row.get("title") or "").strip()
            abstract = (row.get("abstract") or "").strip()
            newstitle = (row.get("newstitle") or "").strip()
            text = title or newstitle or abstract
            if not text:
                continue
            out.append({
                "index": len(out),
                "id": f"coaid-{label}-{len(out)}",
                "label": label,  # 1 reliable / 0 unreliable
                "text": {"original": text},
                "image": {},
                "source": "CoAID",
                "is_english": True,
            })
    return out


def main():
    fake = read(FAKE, 0)
    real = read(REAL, 1)
    data = fake + real
    print(f"fake(unreliable)={len(fake)} real(reliable)={len(real)} total={len(data)}")
    n_test = int(len(data) * 0.2)
    test, train = data[:n_test], data[n_test:]
    print(f"train={len(train)} test={len(test)}")
    payload = {"coaid": {"train": train, "val": test[: max(len(test) // 2, 1)], "test": test}}
    with open(OUT, "w") as f:
        json.dump(payload, f)
    print(f"saved {OUT}")
    lens = [len(s["text"]["original"].split()) for s in data]
    print(f"title word count: min={min(lens)} mean={sum(lens)/len(lens):.1f} max={max(lens)}")


if __name__ == "__main__":
    main()
