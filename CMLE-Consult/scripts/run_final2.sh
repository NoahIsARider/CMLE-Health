#!/bin/bash
# FINAL matrix v2 — main config: full + load-balancing (lambda-balance LB), lr 3e-4, no aux, 30 ep.
# Usage: run_final2.sh <LB>   (e.g. 0.1)
set -x
export PYTHONPATH=/root/CMLE-Health/CMLE-Consult/src
export PYTHONUNBUFFERED=1
cd /root/CMLE-Health/CMLE-Consult
F=/root/cmle-consult/features
LB=${1:-0.1}
O=/root/cmle-consult/runs-final2
mkdir -p $O
COMMON="--features-dir $F --train train.pt --test test_clean.pt --pair --epochs 30 --batch 256 --lr 3e-4 --lambda-mu 0 --lambda-mim 0 --out $O"

echo "############ BASELINES ############"
python3 -u -m cmle_consult.train $COMMON --variant bert-only --tag g_bert_only
python3 -u -m cmle_consult.train $COMMON --variant clip-only --tag g_clip_only
python3 -u -m cmle_consult.train $COMMON --variant concat    --tag g_concat

echo "############ MAIN (full + lb=$LB) ############"
python3 -u -m cmle_consult.train $COMMON --variant full --lambda-balance $LB --tag g_full

echo "############ ABLATIONS ############"
python3 -u -m cmle_consult.train $COMMON --variant full --lambda-balance $LB --lambda-mu 0.1 --tag g_abl_mu
python3 -u -m cmle_consult.train $COMMON --variant w-o-dgm  --lambda-balance $LB --tag g_abl_nodgm
python3 -u -m cmle_consult.train $COMMON --variant w-o-univ --lambda-balance $LB --tag g_abl_nouniv
python3 -u -m cmle_consult.train $COMMON --variant w-o-spec --lambda-balance $LB --tag g_abl_nospec

echo "############ GATE ANALYSIS ############"
PYTHONPATH=/root/CMLE-Health/CMLE-Consult/src python3 -u scripts/analyze_gate.py --ckpt $O/g_full.pt 2>&1 | grep -E "fused acc|expert\[|mean gate|mean agreement"

echo "=== FINAL MATRIX V2 DONE ==="

echo "=== FINAL MATRIX V2 DONE ==="
