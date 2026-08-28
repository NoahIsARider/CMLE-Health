#!/bin/bash
# Balance loss, full 30 epochs: lambda-balance {0.1, 0.3} + gate analysis.
# 15ep probe: lb0.1 -> 0.3655, gate entropy 1.27 (no collapse), fusion > all experts.
set -x
export PYTHONPATH=/root/CMLE-Health/CMLE-Consult/src
export PYTHONUNBUFFERED=1
cd /root/CMLE-Health/CMLE-Consult
F=/root/cmle-consult/features
O=/root/cmle-consult/runs-balance30
mkdir -p $O

python3 -u -m cmle_consult.train --features-dir $F --train train.pt --test test_clean.pt --pair \
  --variant full --epochs 30 --batch 256 --lr 3e-4 --lambda-mu 0 --lambda-mim 0 --lambda-balance 0.1 \
  --out $O --tag b30_full_lb01
python3 -u -m cmle_consult.train --features-dir $F --train train.pt --test test_clean.pt --pair \
  --variant full --epochs 30 --batch 256 --lr 3e-4 --lambda-mu 0 --lambda-mim 0 --lambda-balance 0.3 \
  --out $O --tag b30_full_lb03

echo "=== GATE ANALYSIS ==="
for t in b30_full_lb01 b30_full_lb03; do
  echo "--- $t ---"
  PYTHONPATH=/root/CMLE-Health/CMLE-Consult/src python3 -u scripts/analyze_gate.py \
    --ckpt $O/$t.pt 2>&1 | grep -E "fused acc|expert\[|mean gate|mean agreement"
done
echo "=== BALANCE30 DONE ==="
