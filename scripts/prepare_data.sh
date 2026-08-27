#!/bin/bash
# ============================================================
# Step 1: Download + extract MM-Health dataset
# Target layout:
#   /root/mm-health-data/
#     ├── human_data.zip / machine_data.zip / text_data.zip
#     ├── train_test_splited_data.json
#     ├── human_data/   (extracted images)
#     └── machine_data/ (extracted images)
# ============================================================
set -x
export HF_ENDPOINT=https://hf-mirror.com   # mainland-China mirror; drop if you have direct HF access
cd /root
mkdir -p mm-health-data && cd mm-health-data

# 1) Metadata / text data (small)
python3 - <<'EOF'
from huggingface_hub import hf_hub_download
for f in ["train_test_splited_data.json", "text_data.zip"]:
    p = hf_hub_download(repo_id="zzha6204/MM-Health", filename=f, repo_type="dataset")
    print("got", f, "->", p)
EOF

# 2) Human + machine images (large, chunked download with resume)
bash /root/CMLE-Health/scripts/dl_chunked.sh \
  "https://hf-mirror.com/datasets/zzha6204/MM-Health/resolve/main/human_data.zip" \
  /root/mm-health-data/human_data.zip 1690000000

bash /root/CMLE-Health/scripts/dl_chunked.sh \
  "https://hf-mirror.com/datasets/zzha6204/MM-Health/resolve/main/machine_data.zip" \
  /root/mm-health-data/machine_data.zip 4940000000

# 3) Verify sizes (hf-mirror can silently truncate!)
stat -c '%n %s' human_data.zip machine_data.zip
# expected: human_data.zip ~1.6G, machine_data.zip ~4.7G (exact bytes: 1690xxx / 4943xxx)

# 4) Extract (note: macOS zips contain __MACOSX/._ junk entries — harmless, skip with -x)
unzip -q -o human_data.zip -x '__MACOSX/*' && rm -f human_data.zip
unzip -q -o machine_data.zip -x '__MACOSX/*' && rm -f machine_data.zip
unzip -q -o text_data.zip -d text_data && rm -f text_data.zip

# 5) Sanity checks
echo "human images:  $(find human_data -type f | wc -l)  (expect ~21,017)"
echo "machine images: $(find machine_data -type f | wc -l)  (expect ~109,805)"
ls -lh train_test_splited_data.json
echo "=== DATA READY ==="
