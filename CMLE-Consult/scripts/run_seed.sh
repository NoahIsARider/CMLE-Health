#!/bin/bash
# 3-seed variance for headline configs: full+lb and concat. Seeds: 42, 7, 2026.
set -x
export PYTHONPATH=/root/CMLE-Health/CMLE-Consult/src
export PYTHONUNBUFFERED=1
cd /root/CMLE-Health/CMLE-Consult
F=/root/cmle-consult/features
LB=${1:-0.1}
O=/root/cmle-consult/runs-seed
mkdir -p $O

for S in 7 2026 42; do
  python3 -u -m cmle_consult.train --features-dir $F --train train.pt --test test_clean.pt --pair \
    --variant full --epochs 30 --batch 256 --lr 3e-4 --lambda-mu 0 --lambda-mim 0 --lambda-balance $LB \
    --seed $S --out $O --tag s_full_lb${LB}_seed${S}
done
for S in 7 2026 42; do
  python3 -u -m cmle_consult.train --features-dir $F --train train.pt --test test_clean.pt --pair \
    --variant concat --epochs 30 --batch 256 --lr 3e-4 --lambda-mu 0 --lambda-mim 0 \
    --seed $S --out $O --tag s_concat_seed${S}
done

echo "=== SEED VAR DONE ==="
python3 - <<'PY'
import json, glob
import numpy as np
for pat, name in [(f"{__import__('os').path.dirname('/root/cmle-consult/runs-seed')}/*.json", "")]:
    pass
for prefix in ["s_full", "s_concat"]:
    fs = sorted(glob.glob(f"/root/cmle-consult/runs-seed/{prefix}_*.json"))
    accs = [json.load(open(f))["test"]["acc"] for f in fs]
    print(f"{prefix}: accs={[round(a,4) for a in accs]} mean={np.mean(accs):.4f} std={np.std(accs):.4f}")
PY
