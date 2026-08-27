#!/bin/bash
# ============================================================
# Env fix v2 (after disk-full incident):
#   reinstall torch 2.6.0+cu124 (nvidia wheels already installed),
#   minimal rest deps (transformers pandas scikit-learn pillow;
#   NO torchvision/datasets/peft/accelerate — unused by CMLE-Consult),
#   then precompute test_clean + train features, then delete zips.
# ============================================================
set -x
export PATH=/usr/bin:$PATH
export HF_ENDPOINT=https://hf-mirror.com
PYPI_MIRROR=https://mirrors.aliyun.com/pypi/simple

mkdir -p /root/wheels && cd /root/wheels
curl -sL --max-time 600 -o torch-2.6.0+cu124-cp312-cp312-linux_x86_64.whl \
  "https://mirrors.aliyun.com/pytorch-wheels/cu124/torch-2.6.0%2Bcu124-cp312-cp312-linux_x86_64.whl"
ls -la torch*.whl

pip install --no-input --break-system-packages -i "$PYPI_MIRROR" \
  ./torch-2.6.0+cu124-cp312-cp312-linux_x86_64.whl
echo "torch install exit: $?"

pip install -q --no-input --break-system-packages -i "$PYPI_MIRROR" \
  transformers pandas scikit-learn pillow
echo "rest install exit: $?"

python3 -c "import torch, transformers, pandas, sklearn, PIL; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'); print('transformers', transformers.__version__, '| pandas', pandas.__version__)"
echo ENV2_DONE

# --- free disk: wheels no longer needed ---
rm -rf /root/wheels
df -h / | tail -1

# --- precompute features ---
bash /root/CMLE-Health/CMLE-Consult/scripts/precompute_features.sh 2>&1 | tail -25
echo PRECOMPUTE_DONE

# --- images no longer needed once features are cached ---
rm -f /root/cmle-consult/data/images.zip
df -h / | tail -1
echo ALL_DONE
