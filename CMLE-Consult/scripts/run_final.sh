#!/bin/bash
# FINAL matrix — best config found in diag: lr 3e-4, no aux losses (mu=0, mim=0), 30 epochs.
# Rationale:
#  - lr 1e-3 -> 3e-4: +0.7pp (full mim0: 0.3535 -> 0.3605; val_best 0.3397 -> 0.3475)
#  - dropping MU consensus loss: 0.3645 (noaux) > 0.3605 (mu=0.1); expert agreement drops
#    0.97 -> 0.48, so the gate has real signal to route on (previously experts were
#    forced identical -> gate useless -> full ~= concat)
#  - InfoNCE: lambda 0 vs 0.01 gives bit-identical results -> no observable effect, drop it
set -x
export PYTHONPATH=/root/CMLE-Health/CMLE-Consult/src
export PYTHONUNBUFFERED=1
cd /root/CMLE-Health/CMLE-Consult
F=/root/cmle-consult/features
O=/root/cmle-consult/runs-final
mkdir -p $O
COMMON="--features-dir $F --train train.pt --test test_clean.pt --pair --epochs 30 --batch 256 --lr 3e-4 --lambda-mu 0 --lambda-mim 0 --out $O"

echo "############ BASELINES ############"
python3 -u -m cmle_consult.train $COMMON --variant bert-only --tag f_bert_only
python3 -u -m cmle_consult.train $COMMON --variant clip-only --tag f_clip_only
python3 -u -m cmle_consult.train $COMMON --variant concat    --tag f_concat

echo "############ MAIN ############"
python3 -u -m cmle_consult.train $COMMON --variant full --tag f_full

echo "############ ABLATIONS ############"
python3 -u -m cmle_consult.train $COMMON --variant w-o-dgm  --tag f_abl_nodgm
python3 -u -m cmle_consult.train $COMMON --variant w-o-univ --tag f_abl_nouniv
python3 -u -m cmle_consult.train $COMMON --variant w-o-spec --tag f_abl_nospec

echo "=== FINAL MATRIX DONE ==="
