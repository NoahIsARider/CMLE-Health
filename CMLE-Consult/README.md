# CMLE-Consult

**Multimodal Medical Consultation as an Organization: Role Experts, Dynamic
Gatekeeping, and Mutual Understanding for Closed-Set Medical VQA**

> Sub-project of [CMLE-Health](../README.md). Migrates the CMLE framework
> (LoRA role experts + dynamic gating + cross-modal alignment, IEEE TCE 2026,
> DOI 10.1109/TCE.2026.3677445) from misinformation detection to a *medical
> consultation* setting — the task CMLE was originally designed for.

---

## 1. Motivation & Theory

CMLE-Health's original design intent was **multimodal medical question
answering / clinical problem solving** — not misinformation detection. This
sub-project returns to that intent, grounded in organization theory:

- **Galbraith (1973), *Designing Complex Organizations* — information
  processing theory.** Organizations handle uncertainty by increasing
  information-processing capacity: specialization (roles), lateral relations
  (coordination between roles), and hierarchy (escalation/referral).
- **Joseph, Wilson, Park & Chow (2026), *"Information processing, mutual understanding, and organization design in healthcare"*, Strategic Management Journal (Early View, online Aug 20, 2026; DOI: 10.1002/smj.70116)** — mutual understanding (MU) among professional roles is the mechanism that makes HRO-style healthcare teams reliable under uncertainty. This paper is the **direct inspiration** for CMLE-Consult: it provides the theoretical account of how healthcare organizations process information and build mutual understanding, which we operationalize here as role experts, a consensus (MU) loss, and HRO referral.
- **CMLE lineage** — LoRA role experts + dynamic gating mechanism (DGM) +
  cross-modal InfoNCE, previously validated for health misinformation
  (0.9044 both-task accuracy on MM-Health).

We map the theory to architecture as:

| Organization concept        | Architecture component                |
|-----------------------------|---------------------------------------|
| Organizational units        | Role LoRA experts (clinician / radiologist / pathologist / attending) |
| Architecture of attention   | Dynamic Gating Mechanism (DGM) — per-case allocation of decision rights |
| Mutual understanding (MU)   | Cross-expert consensus loss (KL between expert option distributions) |
| HRO referral / escalation   | Uncertainty-calibrated referral: low-confidence cases are abstained/referred (accuracy @ coverage) |
| Cross-modal alignment       | InfoNCE between question and image representations |

### Testable hypotheses

1. **H1 (division of labor):** specialized role experts outperform a single
   generalist net of equal size on closed-set medical VQA.
2. **H2 (dynamic rights):** per-case DGM gating beats fixed equal expert
   weights.
3. **H3 (mutual understanding):** the consensus loss improves accuracy and
   calibration (lower entropy / higher agreement on correct cases).
4. **H4 (HRO referral):** referral of low-confidence cases (via entropy
   threshold) raises accuracy on the remaining coverage — a clinically
   meaningful operating curve.

---

## 2. Task & Dataset: PMC-VQA (closed-set)

[PMC-VQA](https://arxiv.org/abs/2305.10415) (Zhang et al., ICLR 2024) is the
largest medical VQA dataset: **227K VQA pairs over 149K images** spanning
modalities (X-ray, CT, MRI, microscopy, clinical photos, …) and diseases,
automatically collected from PubMed Central.

We tackle the **closed-set 4-choice task**: given an image + question +
4 answer options (A–D), predict the correct choice.

| Split | Rows | Notes |
|-------|------|-------|
| `train.csv` | 176,949 | official training QA pairs |
| `train_2.csv` | ~50K | extended training QA pairs (optional) |
| `test_clean.csv` | 2,000 | curated closed-set test (headline eval) |
| `test.csv` | ~4K | full test (secondary eval) |

**Why this is a good target for a lightweight model:** published closed-set
accuracy is strikingly low — MedVInT-TE (PMC-CLIP) ≈ **37.6%**, LLaVA-Med
≈ 34.8%, BioMedCLIP ≈ 33%, MedICap-GPT-4 ≈ 27.2% — barely above the 25%
chance level. The task is hard, clinically meaningful, and leaves a clear
opening for a transparent, uncertainty-aware framework like ours.

Data: [HuggingFace `xmcmic/PMC-VQA`](https://huggingface.co/datasets/xmcmic/PMC-VQA)
(`images.zip` 19 GB + `images_2.zip` 2.2 GB, CSV annotations).

---

## 3. Method

### 3.1 Frozen backbones + feature caching

| Modality | Encoder | Cached feature |
|----------|---------|----------------|
| Image    | CLIP ViT-B/32 (frozen) | pooled `(768,)` fp16 |
| Question | BERT-base (frozen)     | pooled `(768,)` fp16 |
| Option ×4| BERT-base (frozen)     | pooled `(4, 768)` fp16 |

Images are read **directly from the zip archives** (no disk extraction —
the 21 GB of zips fit the 30 GB P4 container; extracted images would not).
All training below runs on tiny cached tensors (seconds per epoch).

### 3.2 ConsultNet

Per option *i* the model builds three role-specific views and scores each
option:

- **Clinician (text-centric):** `MLP_t([q, opt_i])`
- **Radiologist (vision-centric):** `MLP_v([img, opt_i])`
- **Pathologist (alignment):** `MLP_a([q, img, opt_i])`
- **Attending (generalist):** `MLP_u([q, img, opt_i])`

Training objective:

```
L = CE(fused) + λ_mu · Σ KL(expert_i || expert_j)   # mutual understanding
              + λ_mim · InfoNCE(q, img)              # cross-modal alignment
```

At inference the DGM gate `softmax(MLP_g([q, img]))` weights the experts
per case; a per-case **agreement score** `exp(−mean pairwise KL)` quantifies
mutual understanding. Cases with high predictive entropy are **referred**
(abstained), giving an **accuracy @ coverage** operating curve (HRO
escalation).

---

## 4. Experiments

Run with the scripts in `scripts/` (see §6). Headline metric: closed-set
accuracy on `test_clean.csv`; secondary: accuracy @ coverage.

### 4.1 Baselines

| Model | Acc (closed-set) |
|-------|------|
| BERT-base + MLP (text only) | *TBD* |
| CLIP ViT-B/32 + MLP (image + options) | *TBD* |
| Concatenation MLP (q+img+opt) | *TBD* |
| **ConsultNet (full)** | *TBD* |

### 4.2 Published SOTA comparison (reported numbers)

| Model | Acc (closed-set) | Source |
|-------|------|--------|
| MedVInT-TE (PMC-CLIP) | 37.6 | Zhang et al. 2024 |
| LLaVA-Med | 34.8 | Li et al. 2023 |
| BioMedCLIP | 33.0 | Zhang et al. 2023 |
| MedICap-GPT-4 | 27.2 | Zhang et al. 2024 |
| Chance | 25.0 | — |
| **ConsultNet (ours, 0.4M params)** | *TBD* | this work |

### 4.3 Ablations

| Variant | Acc | Δ |
|---------|-----|---|
| full | *TBD* | — |
| w/o DGM (equal weights) | *TBD* | |
| w/o MU (no consensus loss) | *TBD* | |
| w/o universal expert | *TBD* | |
| w/o specialized expert | *TBD* | |

### 4.4 HRO referral analysis

Accuracy @ coverage (entropy-threshold referral):

| Coverage | Acc |
|----------|-----|
| 100% | *TBD* |
| 95% | *TBD* |
| 90% | *TBD* |
| 80% | *TBD* |
| 70% | *TBD* |

---

## 5. Repository layout

```
CMLE-Consult/
├── README.md               # this file
├── EXPERIMENTS.md          # live experiment log
├── src/cmle_consult/
│   ├── data.py             # CSV loading + zip-backed image access
│   ├── precompute.py       # frozen BERT/CLIP pooled feature cache
│   ├── model.py            # ConsultNet (experts + DGM + MU + InfoNCE)
│   └── train.py            # train/eval entry, referral curve
└── scripts/
    ├── download_data.sh    # chunked zip download (hf-mirror, resume)
    ├── precompute_features.sh
    ├── run_mvp.sh          # 5K-row end-to-end validation
    └── run_matrix.sh       # full baseline + ablation matrix
```

---

## 6. Reproduce

Environment (Ubuntu 24.04, Python 3.12, P4 8G):
torch 2.6.0+cu124, transformers, datasets, peft, accelerate, scikit-learn,
pillow, pandas.

```bash
# 1. data (21 GB zips + CSVs, chunked resume, hf-mirror)
bash scripts/download_data.sh          # on server: nohup + log

# 2. features (BERT + CLIP pooled caches, ~2 GB)
export HF_ENDPOINT=https://hf-mirror.com
bash scripts/precompute_features.sh

# 3. MVP sanity (5K rows, 3 epochs)
bash scripts/run_mvp.sh

# 4. full matrix
bash scripts/run_matrix.sh
```

Key CLI options (train.py): `--variant {bert-only,clip-only,concat,full,
w-o-dgm,w-o-mu,w-o-univ,w-o-spec}` · `--lambda-mu` · `--lambda-mim` ·
`--seed` · `--limit` · `--eval-only --ckpt`.

---

## 7. Status

- [x] Data pipeline (zip-backed, no extraction)
- [x] Feature precompute (BERT + CLIP pooled, fp16)
- [x] ConsultNet implementation (experts + DGM + MU + referral)
- [x] Baseline / ablation matrix scripts
- [ ] Full experiments (in progress on P4)
- [ ] VLM zero-shot baseline (DeepSeek-VL / Qwen2-VL via API)
- [ ] Paper draft (TCE / JBHI)

## 8. References & Inspiration

The design of CMLE-Consult is grounded in organization theory. Its **direct
inspiration** is the information-processing / mutual-understanding account of
healthcare organizations by Joseph and colleagues (SMJ, 2026) — operationalized
here as role experts, dynamic gating, consensus (MU) loss, and HRO referral.

```bibtex
@article{joseph2026information,
  title={Information processing, mutual understanding, and organization design in healthcare},
  author={Joseph, John and Wilson, Alexander J. and Park, Jihae and Chow, Danny},
  journal={Strategic Management Journal},
  note={Early View, online August 20, 2026},
  year={2026},
  doi={10.1002/smj.70116}
}

@book{galbraith1973designing,
  title={Designing Complex Organizations},
  author={Galbraith, Jay R.},
  year={1973},
  publisher={Addison-Wesley}
}

@article{cmle2026,
  title={CMLE: ...},
  author={Zhou, Fang-Yanuo and others},
  journal={IEEE Transactions on Consumer Electronics},
  year={2026},
  doi={10.1109/TCE.2026.3677445}
}

@inproceedings{zhang2024pmcvqa,
  title={PMC-VQA: Visual Instruction Tuning for Medical Visual Question Answering},
  author={Zhang, Xiaoman and others},
  booktitle={ICLR},
  year={2024}
}
```
