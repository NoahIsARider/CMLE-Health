"""Validate pair-encoding (cross-encoder) signal on a small subset."""
import os
import sys
import time

import torch

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
sys.path.insert(0, "/root/CMLE-Health/CMLE-Consult/src")

import pandas as pd
from sklearn.linear_model import SGDClassifier
from transformers import AutoTokenizer, BertModel

from cmle_consult.data import load_csv

df = load_csv("/root/cmle-consult/data/train.csv").head(2000)
tok = AutoTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased").cuda().eval()

pairs, labels = [], []
for _, r in df.iterrows():
    q = str(r["Question"]).strip()
    for c in ["A", "B", "C", "D"]:
        pairs.append((q, str(r[f"Choice {c}"]).strip()))
    s = str(r["Answer_label"]).strip().upper()
    labels.append("ABCD".find(s[0]) if s and s[0] in "ABCD" else 0)

t0 = time.time()
texts = [f"{q} [SEP] {o}" for q, o in pairs]
feats = []
B = 256
for i in range(0, len(texts), B):
    enc = tok(texts[i:i + B], padding=True, truncation=True, max_length=96,
              return_tensors="pt").to("cuda")
    with torch.no_grad():
        feats.append(bert(**enc).pooler_output.float().cpu())
F = torch.cat(feats).view(2000, 4, 768)
lab = torch.tensor(labels)
print("pair encode 8K:", round(time.time() - t0, 1), "s")

# distinguishability: correct pair vs others
qn = torch.nn.functional.normalize(F, dim=-1)
sim = (qn.unsqueeze(1) * qn).sum(-1)  # (2000,4,4) pair-pair sims
# per row: sim between pair_i and pair_j; we want the correct pair to be
# distinguishable — use sim to the FIRST pair (index 0) as a proxy? no —
# simpler: MLP probe below is the real test.
X = F.numpy().reshape(2000, 4 * 768)
clf = SGDClassifier(loss="log_loss", max_iter=80, random_state=0)
clf.fit(X[:1500], lab[:1500].numpy())
print("pair MLP probe acc:", round(clf.score(X[1500:], lab[1500:].numpy()), 4))
