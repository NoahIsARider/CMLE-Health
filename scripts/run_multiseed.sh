#!/bin/bash
# ============================================================
# Multi-seed variance runs (3 seeds x 12 configs) on cached features
#   seeds: 42 (original) / 7 / 2026
#   configs: text {full rel, full orig, full both, bert-only rel, bert-only orig}
#            mm   {full rel, full orig, full both, concat rel, concat orig, clip-only rel, clip-only orig}
# Output: /root/cmle-health/runs-multiseed/s{seed}_*.json (+ .pt ckpts)
# ============================================================
set -x
export PYTHONPATH=/root/CMLE-Health/src
export PYTHONUNBUFFERED=1
export HF_ENDPOINT=https://hf-mirror.com
cd /root/CMLE-Health
F=/root/cmle-health/features
O=/root/cmle-health/runs-multiseed
mkdir -p $O
COMMON="--features-dir $F --epochs 10 --batch 64 --lr 1e-4 --lambda-mim 1.0 --lora-rank 8 --out $O"

for SEED in 42 7 2026; do
  echo "######## SEED $SEED ########"
  # --- text modality ---
  python3 -u -m cmle_health.train $COMMON --seed $SEED --task reliability --modality text --variant full --tag s${SEED}_t_full_rel
  python3 -u -m cmle_health.train $COMMON --seed $SEED --task originality --modality text --variant full --tag s${SEED}_t_full_orig
  python3 -u -m cmle_health.train $COMMON --seed $SEED --task both --modality text --variant full --tag s${SEED}_t_full_both
  python3 -u -m cmle_health.train $COMMON --seed $SEED --task reliability --modality text --variant bert-only --tag s${SEED}_t_bert_rel
  python3 -u -m cmle_health.train $COMMON --seed $SEED --task originality --modality text --variant bert-only --tag s${SEED}_t_bert_orig
  # --- multimodal / image ---
  python3 -u -m cmle_health.train $COMMON --seed $SEED --task reliability --modality both --variant full --tag s${SEED}_m_full_rel
  python3 -u -m cmle_health.train $COMMON --seed $SEED --task originality --modality both --variant full --tag s${SEED}_m_full_orig
  python3 -u -m cmle_health.train $COMMON --seed $SEED --task both --modality both --variant full --tag s${SEED}_m_full_both
  python3 -u -m cmle_health.train $COMMON --seed $SEED --task reliability --modality both --variant concat --tag s${SEED}_m_concat_rel
  python3 -u -m cmle_health.train $COMMON --seed $SEED --task originality --modality both --variant concat --tag s${SEED}_m_concat_orig
  python3 -u -m cmle_health.train $COMMON --seed $SEED --task reliability --modality image --variant clip-only --tag s${SEED}_m_clip_rel
  python3 -u -m cmle_health.train $COMMON --seed $SEED --task originality --modality image --variant clip-only --tag s${SEED}_m_clip_orig
done
echo "=== MULTISEED MATRIX DONE ==="
