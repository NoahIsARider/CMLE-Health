# CMLE-Health

**Multimodal AI-Generated Health Misinformation Detection with Collaborative LoRA Experts**

CMLE-Health migrates the [CMLE](https://doi.org/10.1109/TCE.2026.3677445) framework
(*Collaborative Multi-LoRA Experts*, IEEE TCE 2026) from general fake-news detection to the
**health misinformation** domain, with a focus on **AI-generated content** (LLM text + generative images).

## Why

- Health misinformation is a red ocean; **AI-generated health misinformation** is a 2025–2026 blue-ocean window
  (deepfake doctors, synthetic medical imagery, LLM-fabricated claims).
- Existing benchmarks (FakeHealth, CoAID, MM-COVID, ReCOVery) lack AI-generated content and raw files.
- **MM-Health** (EMNLP 2025 Findings) is the first large-scale multimodal health-misinformation dataset with
  human + AI-generated content — but it was only benchmarked with zero/few-shot VLLMs (GPT-4o et al. struggle).
  No trainable lightweight detector exists on it yet.

## Method

CMLE-Health keeps the CMLE skeleton — parameter-efficient collaborative experts — and remaps them to health:

| Expert | CMLE (original) | CMLE-Health |
|---|---|---|
| Text | BERT on news text | BERT on health-claim text |
| Image | CLIP ViT-B/32 | CLIP ViT-B/32 (AI-generated / manipulated medical imagery) |
| Comment | social comments | Consistency (text–image coherence) / source-context cues |
| Universal | cross-modal global rep | cross-modal global rep |
| MIM | InfoNCE alignment U↔E_m | same |
| DGM | dynamic gating | same |

- All experts are **LoRA adapters** (rank 8) on frozen backbones → trainable params are tiny
  (~824K), single consumer GPU (RTX 4060-class / P4) suffices.
- Multi-task: reliability check (reliable/unreliable) + originality check (human/AI) jointly.
- Benchmarked on **MM-Health** (34,746 articles, 5,776 human + 28,880 AI) with CoAID transfer checks.

## Status

✅ **Full experiment matrix complete (2026-08-27)** — 44 runs:
text matrix (19) + image/multimodal matrix (14) + BERT full-FT baselines (2) + CoAID transfer (9).
See `EXPERIMENTS_2026-08-27_full.md` for the paper-ready report and `remote-runs/` / `coaid_runs/` for all result JSONs.

Headline results (MM-Health test set):
- reliability macro-F1 **0.8409** (multimodal dual-task) vs best VLLM ≤0.39, vs BERT full-FT (110M params) 0.8320
- originality **0.9991** acc / macro-F1 0.9984 vs VLLM ~0.2
- CoAID zero-shot +4.6pp / few-shot +9.5pp over BERT baseline

---

# Reproduction Guide

This guide is written for someone picking up the project to extend it / write the paper.
Every command below was actually executed on a **single NVIDIA P4 8GB** (no tensor cores, fp32 only).
Total wall-clock for the full matrix: ~2 h after features are cached.

## 0. Directory layout (as used in all scripts)

```
/root/CMLE-Health/            # this repo
/root/mm-health-data/         # MM-Health dataset (json + extracted images)
/root/cmle-health/features/   # precomputed fp16 feature caches (train/val/test x text/image)
/root/cmle-health/runs/       # experiment outputs (json + best-state checkpoints)
/root/coaid/                  # CoAID repo (git clone https://github.com/cuilimeng/CoAID)
/root/coaid_news.json         # CoAID -> CMLE-Health JSON
/root/coaid_features/         # CoAID text feature caches
/root/coaid_runs/             # CoAID experiment outputs
```

Change the paths in `scripts/*.sh` if you use a different layout.

## 1. Environment setup

Hardware: any CUDA GPU with ≥8 GB VRAM. Python 3.10+.

```bash
# Ubuntu 24.04 (blank machine): one-shot setup script with all known pitfalls baked in
bash /root/CMLE-Health/scripts/setup_p4.sh

# Manual equivalent (key steps):
pip install --break-system-packages \
  torch==2.6.0+cu124 torchvision --index-url https://download.pytorch.org/whl/cu124
pip install --break-system-packages transformers scikit-learn pillow
```

Pitfalls baked into `setup_p4.sh` (all hit for real):
- Ubuntu 24.04 blocks system pip → always `--break-system-packages` (a venv **silently** ends up without pip).
- pip is throttled (~450 KB/s) on some Chinese hosts while `curl` is full speed → download wheels with curl,
  install with pip from local files. See `setup_p4.sh` for the exact recipe.
- Install **torch first, transformers after** (transformers pulls CPU torch as a dependency if torch is missing).
- **`HF_ENDPOINT` must be exported in the shell before Python starts** — transformers 5.x reads it at import time;
  setting it inside Python (`os.environ[...]`) in `main()` is too late. Always:
  ```bash
  export HF_ENDPOINT=https://hf-mirror.com   # mainland-China mirror; omit if you have direct HF access
  ```

## 2. Data preparation

### 2.1 MM-Health (main dataset)

```bash
bash /root/CMLE-Health/scripts/prepare_data.sh
```

- Downloads `train_test_splited_data.json`, `text_data.zip`, `human_data.zip` (~1.6G), `machine_data.zip` (~4.7G)
  from HF (mirror-aware, chunked download with resume).
- **Verify byte sizes after download** — hf-mirror can silently truncate large files (curl exits 0 anyway).
- Extract with `unzip -x '__MACOSX/*'` — macOS zips carry ~1 junk metadata entry per real file.
- Sanity: `human_data/` ~21,017 files, `machine_data/` ~109,805 files.

### 2.2 CoAID (cross-domain transfer)

```bash
git clone https://github.com/cuilimeng/CoAID /root/coaid
```
Only `05-01-2020/News{Fake,Real}COVID-19.csv` are used (largest snapshot; titles only, no full text).

## 3. Precompute feature caches (the key speed-up)

```bash
bash /root/CMLE-Health/scripts/precompute_features.sh
```

- Produces `{split}_{text|image}.pt` (fp16): BERT token features (N, 384, 768) and CLIP patch features (N, 50, 768),
  plus `label_a/label_b/meta`.
- ~15 min on P4 for all 6 files; ~15.5 GB disk.
- After this, every training run below takes **seconds per epoch** instead of minutes.

## 4. Run the experiments

All scripts assume the layout of §0 and write JSON results + checkpoints into `runs/`.

```bash
# 4a. Text matrix — 19 runs (~10 min total)
bash /root/CMLE-Health/scripts/run_text_matrix.sh

# 4b. Image / multimodal matrix — 14 runs (~30 min)
bash /root/CMLE-Health/scripts/run_image_matrix.sh

# 4c. BERT full fine-tuning strong baselines — 2 runs (~2 h each on P4)
bash /root/CMLE-Health/scripts/run_bert_fullft.sh

# 4d. CoAID cross-domain — prep + features + zero-shot + few-shot (~15 min)
bash /root/CMLE-Health/scripts/run_coaid.sh
```

### CLI cheat sheet (`python3 -m cmle_health.train`)

| Flag | Meaning |
|---|---|
| `--features-dir DIR` | train from precomputed caches (fast path; no encoders needed) |
| `--modality text\|image\|both` | which modalities; `both` loads `{split}_text.pt` + `{split}_image.pt` separately |
| `--variant full\|w-o-universal\|w-o-specialized\|w-o-consistency\|w-o-mim\|w-o-dgm\|only-text-expert\|only-image-expert\|no-experts\|bert-only\|clip-only\|concat` | model configuration / ablation |
| `--task reliability\|originality\|both` | task head(s) |
| `--eval-only --ckpt PATH` | evaluate a checkpoint; in features-dir mode only the `test` split is loaded (used for cross-domain eval) |
| `--finetune-backbone` | online mode only: fine-tune BERT end-to-end (strong baseline; needs `--data`/`--image-root`, not `--features-dir`) |

Example — evaluate the MM-Health-trained text main model on CoAID (zero-shot transfer):
```bash
python3 -m cmle_health.train --eval-only --ckpt /root/cmle-health/runs/t_full_rel.pt \
  --features-dir /root/coaid_features --task reliability --modality text --variant full
```

## 5. Expected numbers (reproduction targets)

Full tables in `EXPERIMENTS_2026-08-27_full.md`. Key ones (MM-Health test, seed 42):

| Run | task | acc | macro-F1 |
|---|---|---|---|
| `t_bert_rel` (frozen BERT + MLP) | reliability | 0.8886 | 0.8077 |
| `t_full_rel` (main, text) | reliability | 0.8966 | 0.8213 |
| `t_img_full_both` (main, multimodal dual-task) | reliability | 0.9044 | **0.8409** |
| `t_ft_bert_rel` (BERT full-FT, 110M params) | reliability | 0.9037 | 0.8320 |
| `t_full_orig` (main, text) | originality | 0.9988 | 0.9979 |
| `t_img_full_orig` (main, multimodal) | originality | 0.9991 | **0.9984** |
| CoAID zero-shot `t_full_rel` → CoAID | reliability | 0.8032 | 0.4454 |
| CoAID few-shot from scratch | reliability | 0.8866 | 0.4699 |

Tiny run-to-run variance (±0.1–0.3 pp) is expected; all ablations should stay below the full model on `val`.

## 6. Extending / writing the paper

Suggested next steps (not yet done):
1. **Multi-seed variance**: run the main models with `--seed` support (currently hard-coded 42; add a flag) for 2–3 seeds and report mean ± std.
2. VLLM comparison table is ready (vs. MM-Health paper Tables 3/4, macro-F1 aligned) — see report §4.1.
3. CoAID literature comparison is only reference-grade (published CoAID numbers use inconsistent splits; 81.8–98.6% acc).
4. Fine-grained AI detection (Task 3 of the dataset paper, 25 generation-model combinations) is unexplored — natural next contribution.

## License

MIT (code). Dataset: MM-Health is CC BY-NC 4.0 (research use only).

## Citation

If you use this work, please cite both the original CMLE paper and the MM-Health dataset paper
(see `CITATION.cff` / README).
