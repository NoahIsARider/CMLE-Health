#!/bin/bash
# Experiment runner for CMLE-Health on MM-Health.
# Usage: bash scripts/run_experiments.sh [phase]
#   phase=baseline | main | ablation | all
set -e
cd "$(dirname "$0")/.."
export PATH=/root/venv/bin:$PATH 2>/dev/null || true
OUT=/root/cmle-health/runs
mkdir -p $OUT
PY="python3 -m cmle_health.train"
export PYTHONPATH=src

COMMON="--data /root/mm-health-data/train_test_splited_data.json --image-root /root/mm-health-data --epochs 10 --batch 32 --lr 1e-4 --lambda-mim 1.0 --lora-rank 8 --hf-mirror"

phase=${1:-all}
echo "===== PHASE: $phase ====="

if [ "$phase" = "baseline" ] || [ "$phase" = "all" ]; then
  echo "--- bert-only (text, reliability) ---"
  $PY $COMMON --task reliability --modality text --variant bert-only --tag baseline_bert_text_rel --out $OUT
  echo "--- clip-only (image, reliability) ---"
  $PY $COMMON --task reliability --modality image --variant clip-only --tag baseline_clip_img_rel --out $OUT
  echo "--- concat (both, reliability) ---"
  $PY $COMMON --task reliability --modality both --variant concat --tag baseline_concat_rel --out $OUT
fi

if [ "$phase" = "main" ] || [ "$phase" = "all" ]; then
  echo "--- CMLE-Health full (reliability, text) ---"
  $PY $COMMON --task reliability --modality text --variant full --tag main_full_text_rel --out $OUT
  echo "--- CMLE-Health full (reliability, both) ---"
  $PY $COMMON --task reliability --modality both --variant full --tag main_full_both_rel --out $OUT
  echo "--- CMLE-Health full (originality, text) ---"
  $PY $COMMON --task originality --modality text --variant full --tag main_full_text_orig --out $OUT
  echo "--- CMLE-Health full (originality, both) ---"
  $PY $COMMON --task originality --modality both --variant full --tag main_full_both_orig --out $OUT
fi

if [ "$phase" = "ablation" ] || [ "$phase" = "all" ]; then
  for v in w-o-universal w-o-specialized w-o-dgm w-o-mim; do
    echo "--- ablation $v (both tasks, both modalities) ---"
    $PY $COMMON --task both --modality both --variant $v --tag abl_${v}_both --out $OUT
  done
fi

echo "===== DONE ====="
