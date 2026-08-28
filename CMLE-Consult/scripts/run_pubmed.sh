#!/bin/bash
# PubMedCLIP backbone experiment: precompute image features with
# flaviagiammarino/pubmed-clip-vit-base-patch32 (medical-domain CLIP, ViT-B/32,
# same 768-d as current backbone) -> rerun key configs to test the ceiling.
set -x
export PYTHONPATH=/root/CMLE-Health/CMLE-Consult/src
export PYTHONUNBUFFERED=1
export HF_ENDPOINT=https://hf-mirror.com
cd /root/CMLE-Health/CMLE-Consult
F=/root/cmle-consult/features
D=/root/cmle-consult/data
CM=flaviagiammarino/pubmed-clip-vit-base-patch32
O=/root/cmle-consult/runs-pubmed
mkdir -p $O

echo "############ PRECOMPUTE (PubMedCLIP, pair) ############"
# test_clean first (fast), then train
python3 -u -m cmle_consult.precompute --csv $D/test_clean.csv --data-dir $D --out $F \
  --tag test_clean_pubmed --pair --clip-model $CM --hf-mirror
python3 -u -m cmle_consult.precompute --csv $D/train.csv --data-dir $D --out $F \
  --tag train_pubmed --pair --clip-model $CM --hf-mirror

echo "############ KEY CONFIGS (30ep, lr3e-4) ############"
python3 -u -m cmle_consult.train --features-dir $F --train train_pubmed.pt --test test_clean_pubmed.pt --pair \
  --variant clip-only --epochs 30 --batch 256 --lr 3e-4 --lambda-mu 0 --lambda-mim 0 --out $O --tag m_clip_only
python3 -u -m cmle_consult.train --features-dir $F --train train_pubmed.pt --test test_clean_pubmed.pt --pair \
  --variant concat --epochs 30 --batch 256 --lr 3e-4 --lambda-mu 0 --lambda-mim 0 --out $O --tag m_concat
python3 -u -m cmle_consult.train --features-dir $F --train train_pubmed.pt --test test_clean_pubmed.pt --pair \
  --variant full --epochs 30 --batch 256 --lr 3e-4 --lambda-mu 0 --lambda-mim 0 --lambda-balance 0.1 --out $O --tag m_full

echo "############ GATE ANALYSIS ############"
PYTHONPATH=/root/CMLE-Health/CMLE-Consult/src python3 -u scripts/analyze_gate.py --ckpt $O/m_full.pt 2>&1 | grep -E "fused acc|expert\[|mean gate|mean agreement"

echo "=== PUBMED EXPERIMENT DONE ==="
