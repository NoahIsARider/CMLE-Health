#!/bin/bash
# Precompute PMC-VQA features: question/options via BERT-base, images via CLIP ViT-B/32.
# Output: /root/cmle-consult/features/{train,test,test_clean}.pt
set -x
export PYTHONPATH=/root/CMLE-Health/CMLE-Consult/src
export PYTHONUNBUFFERED=1
export HF_ENDPOINT=https://hf-mirror.com
cd /root/CMLE-Health/CMLE-Consult
D=/root/cmle-consult/data
O=/root/cmle-consult/features
mkdir -p $O

python3 -u -m cmle_consult.precompute --csv $D/train.csv --data-dir $D --out $O --tag train --batch 128 --hf-mirror
python3 -u -m cmle_consult.precompute --csv $D/test.csv --data-dir $D --out $O --tag test --batch 128 --hf-mirror
python3 -u -m cmle_consult.precompute --csv $D/test_clean.csv --data-dir $D --out $O --tag test_clean --batch 128 --hf-mirror

ls -lh $O/
echo "=== FEATURES READY ==="
