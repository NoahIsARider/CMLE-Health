#!/bin/bash
# =============================================================================
# P4 GPU server one-shot setup for CMLE-Health (battle-tested 2026-08-26)
#
# Lessons baked in (DO NOT "simplify" these):
#   1. Ubuntu 24.04 pip is PEP-668 externally-managed -> --break-system-packages
#      (python3 -m venv can FAIL silently leaving a broken venv: python symlink
#       but no pip. Don't trust it on these containers.)
#   2. pip downloads get throttled to ~450KB/s on this host, while curl gets
#      ~14MB/s. => ALWAYS fetch wheels with curl, then `pip install ./file.whl`.
#   3. aliyun pytorch-wheels simple-index lists aarch64 FIRST => grep -v aarch64
#      and require x86_64 explicitly. (aarch64 wheels "install" but fail at import)
#   4. triton on the index lists cp310 before cp312 => pick cp312 explicitly.
#   5. Wheel filenames MUST be canonical: {dist}-{ver}-{py}-{abi}-{plat}.whl
#      (use underscores in dist). Rename before pip install or pip rejects.
#   6. Install torch BEFORE transformers: transformers pulls CPU torch as a dep
#      when torch is absent, then the two pip processes fight and you end up
#      with CPU torch + broken cuda.
#   7. pkill -f "pattern" kills YOUR OWN ssh command if the pattern appears in
#      it. Use bracket tricks: pkill -f "pip[ ]install" or run in separate ssh.
#   8. hf-mirror downloads stall/truncate silently. Always verify final size;
#      use chunked resume with fresh connections (see dl_chunked.sh).
#   9. python output to a logfile is block-buffered => always python3 -u.
#  10. BERT-base fp32 forward on P4 is ~0.4s/step (32x384). Never train through
#      the frozen backbone repeatedly: precompute features ONCE (fp16 cache)
#      then train only the small expert head (seconds/epoch).
#  11. Don't write complex scripts via ssh heredoc (quoting hell + ssh drops).
#      Write locally -> scp -> nohup in a SEPARATE ssh command.
# =============================================================================
set -x
export PATH=/usr/bin:$PATH
PYPI_MIRROR=https://mirrors.aliyun.com/pypi/simple
PYWHEELS=https://mirrors.aliyun.com/pytorch-wheels/cu124
HF_ENDPOINT=https://hf-mirror.com
export HF_ENDPOINT

mkdir -p /root/wheels /root/wheels/deps
cd /root/wheels

# --- 0. does torch already work? -------------------------------------------------
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "torch+cuda already OK: $(python3 -c 'import torch; print(torch.__version__)')"
  exit 0
fi

# --- 1. torch + torchvision wheels via curl (fast) --------------------------------
echo "=== fetch torch/torchvision wheels ==="
curl -sL --max-time 1200 -o torch-2.6.0+cu124-cp312-cp312-linux_x86_64.whl \
  "$PYWHEELS/torch-2.6.0%2Bcu124-cp312-cp312-linux_x86_64.whl"
curl -sL --max-time 600 -o torchvision-0.21.0+cu124-cp312-cp312-linux_x86_64.whl \
  "$PYWHEELS/torchvision-0.21.0%2Bcu124-cp312-cp312-linux_x86_64.whl"
ls -la *.whl

# --- 2. nvidia/triton/sympy deps (explicit x86_64, triton cp312) -------------------
echo "=== fetch nvidia deps ==="
SPECS="
nvidia-cuda-nvrtc-cu12==12.4.127
nvidia-cuda-runtime-cu12==12.4.127
nvidia-cuda-cupti-cu12==12.4.127
nvidia-cudnn-cu12==9.1.0.70
nvidia-cublas-cu12==12.4.5.8
nvidia-cufft-cu12==11.2.1.3
nvidia-curand-cu12==10.3.5.147
nvidia-cusolver-cu12==11.6.1.9
nvidia-cusparse-cu12==12.3.1.170
nvidia-cusparselt-cu12==0.6.2
nvidia-nccl-cu12==2.21.5
nvidia-nvtx-cu12==12.4.127
nvidia-nvjitlink-cu12==12.4.127
triton==3.2.0
sympy==1.13.1
"
fetch_one() {
  local spec="$1"; local name="${spec%%==*}"; local ver="${spec##*==}"
  local href
  if [ "$name" = "triton" ]; then
    # triton: aliyun index lists cp310 BEFORE cp312 — force cp312 explicitly
    href=$(curl -s --max-time 25 "https://mirrors.aliyun.com/pypi/simple/${name}/" \
      | grep -oE 'href="[^"]*'"${name//-/_}"'-'"${ver}"'[^"]*cp312[^"]*\.whl[^"]*"' \
      | grep -v aarch64 | grep -v arm64 | grep -E "x86_64|manylinux" | head -1 \
      | sed 's/href="//;s/"$//;s/#sha256=.*//')
  else
    href=$(curl -s --max-time 25 "https://mirrors.aliyun.com/pypi/simple/${name}/" \
      | grep -oE 'href="[^"]*'"${name//-/_}"'-'"${ver}"'[^"]*\.whl[^"]*"' \
      | grep -v aarch64 | grep -v arm64 | grep -E "x86_64|manylinux" | head -1 \
      | sed 's/href="//;s/"$//;s/#sha256=.*//')
  fi
  if [ -z "$href" ]; then echo "MISS $name $ver"; return 1; fi
  case "$href" in http*) full="$href" ;; *) full="https://mirrors.aliyun.com/pypi/simple/${name}/${href}" ;; esac
  cd /root/wheels/deps && curl -sL --max-time 900 -o "$(basename "$href")" "$full" && echo "GOT $(basename "$href")"
  cd /root/wheels
}
export -f fetch_one
echo "$SPECS" | grep -v '^$' | xargs -P 4 -I{} bash -c 'fetch_one "{}"'

# --- 3. install everything locally (single pip process!) ---------------------------
echo "=== pip install torch stack ==="
pip install --no-input --break-system-packages -i "$PYPI_MIRROR" /root/wheels/deps/*.whl \
  ./torch-2.6.0+cu124-cp312-cp312-linux_x86_64.whl \
  ./torchvision-0.21.0+cu124-cp312-cp312-linux_x86_64.whl
echo "torch stack exit: $?"

# --- 4. rest deps AFTER torch (avoid CPU torch pull) --------------------------------
echo "=== pip install rest ==="
pip install -q --no-input --break-system-packages -i "$PYPI_MIRROR" \
  transformers datasets peft accelerate scikit-learn sentencepiece
echo "rest exit: $?"

# --- 5. verify ----------------------------------------------------------------------
python3 -c "import torch, torchvision, transformers, datasets, peft; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'); print('transformers', transformers.__version__)"
echo SETUP_P4_DONE
