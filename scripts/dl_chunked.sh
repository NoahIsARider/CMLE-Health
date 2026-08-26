#!/bin/bash
# Chunked download with connection restart: keeps getting fresh fast connections
set -x
cd /root/mm-health-data
TARGET=1694246692

dl_loop() {
  local f="$1"
  local url="$2"
  local expected="$3"
  while true; do
    local cur=$(stat -c %s "$f" 2>/dev/null || echo 0)
    if [ "$cur" -ge "$expected" ]; then
      echo "$f COMPLETE ($cur bytes)"
      break
    fi
    echo "$(date +%H:%M:%S) $f: $cur/$expected, resuming..."
    # download up to 8 min per attempt, then kill to get a fresh connection
    timeout 480 curl -sL -C - --max-time 460 -o "$f" "$url" || true
    # verify it actually grew
    local cur2=$(stat -c %s "$f" 2>/dev/null || echo 0)
    if [ "$cur2" -le "$cur" ]; then
      echo "$f: no progress, waiting 10s and retrying..."
      sleep 10
    fi
  done
}

dl_loop human_data.zip "https://hf-mirror.com/datasets/zzha6204/MM-Health/resolve/main/human_data.zip" 1694246692
echo "=== human done, unzip ==="
unzip -o -q human_data.zip -d . && echo "human unzip OK: $(ls human_data 2>/dev/null | wc -l) entries"

dl_loop machine_data.zip "https://hf-mirror.com/datasets/zzha6204/MM-Health/resolve/main/machine_data.zip" 4945571842
echo "=== machine done, unzip ==="
unzip -o -q machine_data.zip -d . && echo "machine unzip OK"
echo IMG_ALL_DONE
