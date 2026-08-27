#!/bin/bash
# ============================================================
# Step 2: Precompute frozen-backbone features (fp16 caches)
#   text  : BERT-base token features (N, 384, 768)  -> {split}_text.pt
#   image : CLIP ViT-B/32 patch features (N, 50, 768) -> {split}_image.pt
# This is the KEY speed-up: all training below then runs in
# seconds-per-epoch instead of minutes, on a single 8G GPU.
#
# Disk: ~13G (text) + ~2.5G (image) for MM-Health splits.
# ============================================================
set -x
export PYTHONPATH=/root/CMLE-Health/src
export PYTHONUNBUFFERED=1
export HF_ENDPOINT=https://hf-mirror.com   # MUST be set before python starts:
                                           # transformers 5.x reads it at import time.
cd /root/CMLE-Health
D=/root/mm-health-data/train_test_splited_data.json
R=/root/mm-health-data
O=/root/cmle-health/features
mkdir -p $O

for SPLIT in train val test; do
  python3 -u -m cmle_health.precompute --data $D --image-root $R \
      --split $SPLIT --modality text --batch 64 --out $O --hf-mirror
done

for SPLIT in train val test; do
  python3 -u -m cmle_health.precompute --data $D --image-root $R \
      --split $SPLIT --modality image --batch 64 --out $O --hf-mirror
done

ls -lh $O/
echo "=== FEATURES READY ==="
