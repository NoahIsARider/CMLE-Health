#!/bin/bash
# ============================================================
# Step 4: CoAID cross-domain transfer experiments
#   a) zero-shot : MM-Health-trained checkpoints evaluated on CoAID test
#   b) few-shot  : fine-tune on CoAID train (from scratch vs MM-Health init)
# Prereqs: CoAID CSVs at /root/coaid/05-01-2020/News{Fake,Real}COVID-19.csv
#          (git clone https://github.com/cuilimeng/CoAID)
# ============================================================
set -x
export PYTHONPATH=/root/CMLE-Health/src
export PYTHONUNBUFFERED=1
export HF_ENDPOINT=https://hf-mirror.com
cd /root

# --- data prep: CSV -> JSON (train 1730 / test 432) ---
python3 /root/CMLE-Health/scripts/coaid_prep.py

# --- CoAID text features (BERT frozen) ---
F=/root/coaid_features
mkdir -p $F
for SPLIT in train val test; do
  python3 -u -m cmle_health.precompute --data /root/coaid_news.json --image-root /root \
      --split $SPLIT --modality text --batch 64 --out $F --hf-mirror
done

O=/root/coaid_runs
mkdir -p $O
R=/root/cmle-health/runs   # MM-Health trained checkpoints

echo "===== ZERO-SHOT (MM-Health -> CoAID test) ====="
python3 -u -m cmle_health.train --eval-only --ckpt $R/t_bert_rel.pt --features-dir $F \
    --task reliability --modality text --variant bert-only --out $O --tag coaid_zs_bert
python3 -u -m cmle_health.train --eval-only --ckpt $R/t_full_rel.pt --features-dir $F \
    --task reliability --modality text --variant full --out $O --tag coaid_zs_full
python3 -u -m cmle_health.train --eval-only --ckpt $R/t_full_both.pt --features-dir $F \
    --task reliability --modality text --variant full --out $O --tag coaid_zs_full_both
python3 -u -m cmle_health.train --eval-only --ckpt $R/t_noexp_rel.pt --features-dir $F \
    --task reliability --modality text --variant no-experts --out $O --tag coaid_zs_noexp

echo "===== FEW-SHOT (fine-tune on CoAID train) ====="
python3 -u -m cmle_health.train --features-dir $F --task reliability --modality text \
    --variant full --epochs 10 --batch 64 --lr 1e-4 --out $O --tag coaid_ft_fromscratch_full
python3 -u -m cmle_health.train --features-dir $F --task reliability --modality text \
    --variant full --epochs 10 --batch 64 --lr 5e-5 --out $O --tag coaid_ft_transfer_full \
    --ckpt $R/t_full_rel.pt
python3 -u -m cmle_health.train --features-dir $F --task reliability --modality text \
    --variant bert-only --epochs 10 --batch 64 --lr 5e-5 --out $O --tag coaid_ft_transfer_bert \
    --ckpt $R/t_bert_rel.pt

echo "=== COAID DONE ==="
