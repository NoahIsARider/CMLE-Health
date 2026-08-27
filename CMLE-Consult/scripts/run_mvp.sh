#!/bin/bash
# MVP: end-to-end pipeline validation on a small subset (fast, catches bugs).
set -x
export PYTHONPATH=/root/CMLE-Health/CMLE-Consult/src
export PYTHONUNBUFFERED=1
cd /root/CMLE-Health/CMLE-Consult
F=/root/cmle-consult/features
O=/root/cmle-consult/runs-mvp
mkdir -p $O
COMMON="--features-dir $F --test test_clean.pt --epochs 3 --batch 256 --lr 1e-3 --lambda-mu 0.1 --lambda-mim 0.1 --out $O --limit 5000"

python3 -u -m cmle_consult.train $COMMON --variant bert-only --tag mvp_bert
python3 -u -m cmle_consult.train $COMMON --variant concat --tag mvp_concat
python3 -u -m cmle_consult.train $COMMON --variant full --tag mvp_full
echo "=== MVP DONE ==="
