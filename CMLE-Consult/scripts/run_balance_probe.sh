#!/bin/bash
# Probe: does load-balancing loss fix the gate mode collapse?
# Run full ConsultNet with lambda-balance in {0.02, 0.1} x lr 3e-4, 15 epochs,
# then analyze gate_w distribution + fused acc vs no-balance (0.3565@30ep, gate [0.17,0.83,0,0]).
set -x
export PYTHONPATH=/root/CMLE-Health/CMLE-Consult/src
export PYTHONUNBUFFERED=1
cd /root/CMLE-Health/CMLE-Consult
F=/root/cmle-consult/features
O=/root/cmle-consult/runs-balance
mkdir -p $O

python3 -u -m cmle_consult.train --features-dir $F --train train.pt --test test_clean.pt --pair \
  --variant full --epochs 15 --batch 256 --lr 3e-4 --lambda-mu 0 --lambda-mim 0 --lambda-balance 0.02 \
  --out $O --tag b_full_lb002
python3 -u -m cmle_consult.train --features-dir $F --train train.pt --test test_clean.pt --pair \
  --variant full --epochs 15 --batch 256 --lr 3e-4 --lambda-mu 0 --lambda-mim 0 --lambda-balance 0.1 \
  --out $O --tag b_full_lb01

echo "=== GATE ANALYSIS ==="
for t in b_full_lb002 b_full_lb01; do
  echo "--- $t ---"
  PYTHONPATH=/root/CMLE-Health/CMLE-Consult/src python3 -u scripts/analyze_gate.py \
    --ckpt $O/$t.pt 2>&1 | grep -E "fused acc|expert\[|mean gate|mean agreement|expert names"
done
echo "=== BALANCE PROBE DONE ==="
