#!/bin/bash
# Precompute PMC-VQA features (mmap + parallel decode) with label verification.
# NOTE: does NOT delete images.zip (manual cleanup after verification).
set -x
export PYTHONPATH=/root/CMLE-Health/CMLE-Consult/src
export PYTHONUNBUFFERED=1
export HF_ENDPOINT=https://hf-mirror.com
cd /root/CMLE-Health/CMLE-Consult
D=/root/cmle-consult/data
O=/root/cmle-consult/features
mkdir -p $O

python3 -u -m cmle_consult.precompute --csv $D/test_clean.csv --data-dir $D --out $O --tag test_clean --batch 128 --memory --hf-mirror 2>&1 | tee /root/precompute-test.log
python3 -u -m cmle_consult.precompute --csv $D/train.csv --data-dir $D --out $O --tag train --batch 128 --memory --hf-mirror 2>&1 | tee /root/precompute-train.log

python3 - <<'EOF'
import torch, collections
for tag in ["test_clean", "train"]:
    c = torch.load(f"/root/cmle-consult/features/{tag}.pt", map_location="cpu")
    d = dict(collections.Counter(c["label"].tolist()))
    print(tag, "label dist:", d, "| n:", len(c["label"]), "| img std:", c["img"].float().std().item())
    assert all(d.get(i, 0) > 0 for i in range(4)), f"{tag} labels NOT balanced — BUG!"
EOF
echo PRECOMPUTE_VERIFIED
