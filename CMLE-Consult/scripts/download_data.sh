#!/bin/bash
# ============================================================
# Download PMC-VQA data (chunked resume, no extraction!)
#   images.zip (19G) + images_2.zip (2.2G) stay as zips;
#   precompute reads images directly from zip (disk: 29G only).
# ============================================================
cd /root/cmle-consult/data || { mkdir -p /root/cmle-consult/data && cd /root/cmle-consult/data; }

dl_loop() {
  local f="$1"; local url="$2"; local expected="$3"
  while true; do
    local cur=$(stat -c %s "$f" 2>/dev/null || echo 0)
    if [ "$cur" -ge "$expected" ]; then echo "$f COMPLETE ($cur bytes)"; break; fi
    echo "$(date +%H:%M:%S) $f: $cur/$expected, resuming..."
    timeout 480 curl -sL -C - --max-time 460 -o "$f" "$url" || true
    local cur2=$(stat -c %s "$f" 2>/dev/null || echo 0)
    if [ "$cur2" -le "$cur" ]; then echo "$f: no progress, waiting 10s..."; sleep 10; fi
  done
}

dl_loop images.zip    "https://hf-mirror.com/datasets/xmcmic/PMC-VQA/resolve/main/images.zip"    19800000000
dl_loop images_2.zip  "https://hf-mirror.com/datasets/xmcmic/PMC-VQA/resolve/main/images_2.zip"  2300000000

echo "=== zips done $(date +%H:%M:%S), fetching CSVs ==="
for c in train.csv test.csv test_clean.csv train_2.csv; do
  curl -sL --max-time 180 -o "$c" "https://hf-mirror.com/datasets/xmcmic/PMC-VQA/resolve/main/$c"
  echo "$c: $(stat -c %s "$c" 2>/dev/null) bytes"
done
ls -lh
echo CONSULT_DATA_DONE
