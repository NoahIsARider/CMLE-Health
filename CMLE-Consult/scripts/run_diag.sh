#!/bin/bash
# Diagnostic grid: InfoNCE magnitude dominates (loss ~1120/256≈4.4/row while CE≈1.35)
# Hypothesis: lambda_mim=0.1 * infonce(~30/row) swamps the CE signal -> uniform-ish output.
# Test lambda_mim in {0, 0.01} x lr in {1e-3, 3e-4} on full + concat, 15 epochs.
set -x
export PYTHONPATH=/root/CMLE-Health/CMLE-Consult/src
export PYTHONUNBUFFERED=1
cd /root/CMLE-Health/CMLE-Consult
F=/root/cmle-consult/features
O=/root/cmle-consult/runs-diag
mkdir -p $O

# full, drop InfoNCE entirely
python3 -u -m cmle_consult.train --features-dir $F --train train.pt --test test_clean.pt --pair \
  --variant full --epochs 15 --batch 256 --lr 1e-3 --lambda-mu 0.1 --lambda-mim 0 --out $O --tag d_full_mim0_lr1e3
# full, drop InfoNCE, smaller lr
python3 -u -m cmle_consult.train --features-dir $F --train train.pt --test test_clean.pt --pair \
  --variant full --epochs 15 --batch 256 --lr 3e-4 --lambda-mu 0.1 --lambda-mim 0 --out $O --tag d_full_mim0_lr3e4
# full, tiny InfoNCE
python3 -u -m cmle_consult.train --features-dir $F --train train.pt --test test_clean.pt --pair \
  --variant full --epochs 15 --batch 256 --lr 1e-3 --lambda-mu 0.1 --lambda-mim 0.01 --out $O --tag d_full_mim001_lr1e3
# concat reference with lr 3e-4
python3 -u -m cmle_consult.train --features-dir $F --train train.pt --test test_clean.pt --pair \
  --variant concat --epochs 15 --batch 256 --lr 3e-4 --lambda-mu 0 --lambda-mim 0 --out $O --tag d_concat_lr3e4
# full, both aux losses off, lr 3e-4 (pure CE baseline for ConsultNet)
python3 -u -m cmle_consult.train --features-dir $F --train train.pt --test test_clean.pt --pair \
  --variant full --epochs 15 --batch 256 --lr 3e-4 --lambda-mu 0 --lambda-mim 0 --out $O --tag d_full_noaux_lr3e4

echo "=== DIAG DONE ==="
