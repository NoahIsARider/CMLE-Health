#!/bin/bash
# ============================================================
# Step 3a: Text-modality experiment matrix (19 runs, cached features)
#   baselines (bert-only / no-experts) x {reliability, originality}
#   main model (full) x {reliability, originality, both}
#   ablations x reliability (+ both for w-o-mim / w-o-dgm)
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

echo "############ BASELINES ############"
python3 -u -m cmle_health.train $COMMON --task reliability --modality text --variant bert-only --tag t_bert_rel
python3 -u -m cmle_health.train $COMMON --task originality --modality text --variant bert-only --tag t_bert_orig
python3 -u -m cmle_health.train $COMMON --task reliability --modality text --variant no-experts --tag t_noexp_rel
python3 -u -m cmle_health.train $COMMON --task originality --modality text --variant no-experts --tag t_noexp_orig

echo "############ MAIN ############"
python3 -u -m cmle_health.train $COMMON --task reliability --modality text --variant full --tag t_full_rel
python3 -u -m cmle_health.train $COMMON --task originality --modality text --variant full --tag t_full_orig
python3 -u -m cmle_health.train $COMMON --task both --modality text --variant full --tag t_full_both

echo "############ ABLATIONS ############"
python3 -u -m cmle_health.train $COMMON --task reliability --modality text --variant w-o-universal --tag t_abl_nouni_rel
python3 -u -m cmle_health.train $COMMON --task reliability --modality text --variant w-o-specialized --tag t_abl_nospec_rel
python3 -u -m cmle_health.train $COMMON --task reliability --modality text --variant w-o-consistency --tag t_abl_nocons_rel
python3 -u -m cmle_health.train $COMMON --task reliability --modality text --variant w-o-mim --tag t_abl_nomim_rel
python3 -u -m cmle_health.train $COMMON --task reliability --modality text --variant w-o-dgm --tag t_abl_nodgm_rel
python3 -u -m cmle_health.train $COMMON --task reliability --modality text --variant only-text-expert --tag t_abl_onlytext_rel
python3 -u -m cmle_health.train $COMMON --task both --modality text --variant w-o-mim --tag t_abl_nomim_both
python3 -u -m cmle_health.train $COMMON --task both --modality text --variant w-o-dgm --tag t_abl_nodgm_both

echo "=== TEXT MATRIX DONE ==="
