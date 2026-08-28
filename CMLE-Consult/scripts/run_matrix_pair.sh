#!/bin/bash
# Pair-feature experiment matrix (same as run_matrix.sh but with --pair).
set -x
export PYTHONPATH=/root/CMLE-Health/CMLE-Consult/src
export PYTHONUNBUFFERED=1
cd /root/CMLE-Health/CMLE-Consult
F=/root/cmle-consult/features
O=/root/cmle-consult/runs-pair
mkdir -p $O
COMMON="--features-dir $F --train train.pt --test test_clean.pt --pair --epochs 10 --batch 256 --lr 1e-3 --lambda-mu 0.1 --lambda-mim 0.1 --out $O"

echo "############ BASELINES ############"
python3 -u -m cmle_consult.train $COMMON --variant bert-only --tag p_bert_only
python3 -u -m cmle_consult.train $COMMON --variant clip-only --tag p_clip_only
python3 -u -m cmle_consult.train $COMMON --variant concat    --tag p_concat

echo "############ MAIN ############"
python3 -u -m cmle_consult.train $COMMON --variant full --tag p_full

echo "############ ABLATIONS ############"
python3 -u -m cmle_consult.train $COMMON --variant w-o-dgm  --tag p_abl_nodgm
python3 -u -m cmle_consult.train $COMMON --variant w-o-mu   --tag p_abl_nomu
python3 -u -m cmle_consult.train $COMMON --variant w-o-univ --tag p_abl_nouniv
python3 -u -m cmle_consult.train $COMMON --variant w-o-spec --tag p_abl_nospec

echo "=== PAIR MATRIX DONE ==="
