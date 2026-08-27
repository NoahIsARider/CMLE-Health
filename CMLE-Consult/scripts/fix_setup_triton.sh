#!/bin/bash
# Fix setup_p4.sh triton bug: cp310 wheel fetched instead of cp312.
# 1) fetch correct triton cp312 wheel, drop the cp310 one
# 2) wait for the running 'pip install rest' to finish
# 3) re-run the local-wheel install (torch + torchvision + nvidia deps)
# 4) verify torch.cuda
set -x
export PATH=/usr/bin:$PATH
PYPI_MIRROR=https://mirrors.aliyun.com/pypi/simple

cd /root/wheels/deps
curl -sL --max-time 300 -o triton-3.2.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl \
  "https://mirrors.aliyun.com/pypi/packages/06/00/59500052cb1cf8cf5316be93598946bc451f14072c6ff256904428eaf03c/triton-3.2.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
rm -f triton-3.2.0-cp310-*.whl
ls -la triton*

echo "=== waiting for running pip to exit ==="
while ps aux | grep -q "[p]ip install"; do sleep 20; done
echo "=== pip idle, installing wheel stack ==="

pip install --no-input --break-system-packages -i "$PYPI_MIRROR" \
  /root/wheels/deps/*.whl \
  /root/wheels/torch-2.6.0+cu124-cp312-cp312-linux_x86_64.whl \
  /root/wheels/torchvision-0.21.0+cu124-cp312-cp312-linux_x86_64.whl
echo "wheel install exit: $?"

python3 -c "import torch, torchvision, transformers, datasets, peft; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'); print('transformers', transformers.__version__)"
echo FIX_DONE
