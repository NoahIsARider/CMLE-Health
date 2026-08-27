#!/bin/bash
# Download PMC-VQA data (chunked resume).
# NOTE: images_2.zip is NOT needed — train.csv + test_clean.csv figures are
# 100% covered by images.zip (verified 2026-08-27: 115,821 + 1,440 images).
cd /root/cmle-consult/data || { mkdir -p /root/cmle-consult/data && cd /root/cmle-consult/data; }

echo "=== CSVs first (tiny) ==="
for c in train.csv test.csv test_clean.csv; do
  curl -sL --max-time 180 -o "$c" "https://hf-mirror.com/datasets/xmcmic/PMC-VQA/resolve/main/$c"
  echo "$c: $(stat -c %s "$c") bytes"
done

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

# exact Content-Length from hf-mirror (verified 2026-08-27): 18,945,102,275 bytes
dl_loop images.zip "https://hf-mirror.com/datasets/xmcmic/PMC-VQA/resolve/main/images.zip" 18945102275

ls -lh
echo CONSULT_DATA_DONE
