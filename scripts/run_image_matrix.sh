#!/bin/bash
# ============================================================
# Step 3b: Image / multimodal experiment matrix (14 runs, cached features)
#   baselines (clip-only x2, concat x2, no-experts) + main (full x3)
#   ablations x reliability + only-image-expert
# NOTE: modality=both loads {split}_text.pt + {split}_image.pt separately
#       (no concatenated cache file needed — see train.py).
# ============================================================
set -x
export PYTHONPATH=/root/CMLE-Health/src
export PYTHONUNBUFFERED=1
export HF_ENDPOINT=https://hf-mirror.com
cd /root/CMLE-Health
F=/root/cmle-health/features
O=/root/cmle-health/runs
mkdir -p $O
COMMON="--features-dir $F --epochs 10 --batch 64 --lr 1e-4 --lambda-mim 1.0 --lora-rank 8 --out $O"

echo "############ BASELINES (image / concat) ############"
python3 -u -m cmle_health.train $COMMON --task reliability --modality image --variant clip-only --tag t_img_clip_rel
python3 -u -m cmle_health.train $COMMON --task originality --modality image --variant clip-only --tag t_img_clip_orig
python3 -u -m cmle_health.train $COMMON --task reliability --modality both  --variant concat    --tag t_img_concat_rel
python3 -u -m cmle_health.train $COMMON --task originality --modality both  --variant concat    --tag t_img_concat_orig
python3 -u -m cmle_health.train $COMMON --task reliability --modality both  --variant no-experts --tag t_img_noexp_rel

echo "############ MAIN (multimodal full) ############"
python3 -u -m cmle_health.train $COMMON --task reliability --modality both --variant full --tag t_img_full_rel
python3 -u -m cmle_health.train $COMMON --task originality --modality both --variant full --tag t_img_full_orig
python3 -u -m cmle_health.train $COMMON --task both --modality both --variant full --tag t_img_full_both

echo "############ ABLATIONS (multimodal, reliability) ############"
python3 -u -m cmle_health.train $COMMON --task reliability --modality both --variant w-o-universal --tag t_img_abl_nouni_rel
python3 -u -m cmle_health.train $COMMON --task reliability --modality both --variant w-o-specialized --tag t_img_abl_nospec_rel
python3 -u -m cmle_health.train $COMMON --task reliability --modality both --variant w-o-consistency --tag t_img_abl_nocons_rel
python3 -u -m cmle_health.train $COMMON --task reliability --modality both --variant w-o-mim --tag t_img_abl_nomim_rel
python3 -u -m cmle_health.train $COMMON --task reliability --modality both --variant w-o-dgm --tag t_img_abl_nodgm_rel
python3 -u -m cmle_health.train $COMMON --task reliability --modality image --variant only-image-expert --tag t_img_abl_onlyimg_rel

echo "=== IMAGE MATRIX DONE ==="
