#!/bin/bash
# Full experiment matrix: 3 baselines + main + 4 ablations.
set -x
export PYTHONPATH=/root/CMLE-Health/CMLE-Consult/src
export PYTHONUNBUFFERED=1
cd /root/CMLE-Health/CMLE-Consult
F=/root/cmle-consult/features
O=/root/cmle-consult/runs
mkdir -p $O
COMMON="--features-dir $F --train train.pt --test test_clean.pt --epochs 10 --batch 256 --lr 1e-3 --lambda-mu 0.1 --lambda-mim 0.1 --out $O"

echo "############ BASELINES ############"
python3 -u -m cmle_consult.train $COMMON --variant bert-only --tag c_bert_only
python3 -u -m cmle_consult.train $COMMON --variant clip-only --tag c_clip_only
python3 -u -m cmle_consult.train $COMMON --variant concat    --tag c_concat

echo "############ MAIN ############"
python3 -u -m cmle_consult.train $COMMON --variant full --tag c_full

echo "############ ABLATIONS ############"
python3 -u -m cmle_consult.train $COMMON --variant w-o-dgm  --tag c_abl_nodgm
python3 -u -m cmle_consult.train $COMMON --variant w-o-mu   --tag c_abl_nomu
python3 -u -m cmle_consult.train $COMMON --variant w-o-univ --tag c_abl_nouniv
python3 -u -m cmle_consult.train $COMMON --variant w-o-spec --tag c_abl_nospec

echo "=== CONSULT MATRIX DONE ==="
