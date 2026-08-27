#!/bin/bash
# ============================================================
# Step 3c: BERT full fine-tuning baselines (strong baseline, ~2h each on P4)
#   online mode (no feature cache) + --finetune-backbone
#   batch 32 + max-len 256 to fit 8G VRAM (batch 32 x 384 OOMs on P4)
# ============================================================
set -x
export PYTHONPATH=/root/CMLE-Health/src
export PYTHONUNBUFFERED=1
export HF_ENDPOINT=https://hf-mirror.com
cd /root/CMLE-Health
D=/root/mm-health-data/train_test_splited_data.json
R=/root/mm-health-data
O=/root/cmle-health/runs

python3 -u -m cmle_health.train --data $D --image-root $R --task reliability --modality text \
    --variant bert-only --finetune-backbone --epochs 5 --batch 32 --lr 2e-5 --max-len 256 \
    --out $O --tag t_ft_bert_rel

python3 -u -m cmle_health.train --data $D --image-root $R --task originality --modality text \
    --variant bert-only --finetune-backbone --epochs 5 --batch 32 --lr 2e-5 --max-len 256 \
    --out $O --tag t_ft_bert_orig

echo "=== BERT FULL-FT DONE ==="
