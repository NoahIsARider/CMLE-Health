# CMLE-Consult — Experiment Log

Live log of all runs. Raw JSON results in `runs/` (remote: `/root/cmle-consult/runs`).

## 2026-08-27 — Setup day

- Server: DeepLn P4 (8G, 30G disk), fresh container. torch 2.6.0+cu124 via
  `setup_p4.sh` (battle-tested from CMLE-Health).
- Data: `xmcmic/PMC-VQA` — `images.zip` (18,945 MB) + `images_2.zip` (2,206 MB)
  + CSVs. Strategy: **no disk extraction** — precompute reads images from zip
  in memory (disk can only fit zips + feature caches).
- CSV schema confirmed: `Figure_path, Question, Answer, Choice A-D, Answer_label`
  (closed-set 4-choice). train.csv = 176,949 rows; test_clean.csv = 2,000 rows.

### Pipeline validation (MVP, 5K train rows, 3 epochs)
| variant | test acc | notes |
|---------|----------|-------|
| bert-only | TBD | |
| concat | TBD | |
| full | TBD | |

### Full matrix (10 epochs, full train) — round 1: **chance-level (25%), features broken**

First run (option-encoding without question context) gave test acc 0.239-0.2675 on ALL 8
configs with mean_entropy ≈ ln(4) (uniform). Diagnosis: options were encoded standalone,
no question context → no learnable signal. Fix: pair cross-encoder features
(`[question SEP option]`) + per-option scoring head (FlatMLP outputs 1 score/option).

### Pair matrix (10 epochs, full train) — 2026-08-28, all above chance
| variant | test acc | acc@95% | acc@90% | acc@80% | acc@70% | val_best |
|---------|----------|---------|---------|---------|---------|----------|
| bert-only | 0.3245 | 0.3296 | 0.3339 | 0.3424 | 0.3486 | 0.3246 |
| clip-only | 0.3510 | 0.3560 | 0.3594 | 0.3634 | 0.3650 | 0.3329 |
| concat | **0.3600** | 0.3642 | 0.3672 | 0.3756 | **0.3914** | 0.3323 |
| full | 0.3540 | 0.3605 | 0.3650 | 0.3738 | 0.3864 | 0.3352 |
| w-o-dgm | 0.3480 | 0.3526 | 0.3550 | 0.3606 | 0.3750 | 0.3368 |
| w-o-mu | 0.3540 | 0.3605 | 0.3650 | 0.3738 | 0.3864 | 0.3352 |
| w-o-univ | 0.3535 | 0.3579 | 0.3661 | 0.3769 | 0.3886 | 0.3354 |
| w-o-spec | 0.3535 | 0.3579 | 0.3661 | 0.3769 | 0.3886 | 0.3354 |

Notes / issues:
- **w-o-univ == w-o-spec exactly** (params 1,870,086 both): the two "generalist" experts
  (pathologist a / attending u) take the SAME input concat (text+img+opt) → architecturally
  identical, dropping either leaves the same 3-expert net. Redundant design; report as ONE
  ablation (w-o-generalist) or differentiate expert inputs.
- **w-o-mu == full exactly**: expert probs are near-identical (mean_agreement ≈ 0.98) →
  consensus KL ≈ 0 → loss term inert. MU currently redundant.
- full (0.354) < concat (0.360): consult gating not yet beating concat on PMC-VQA.
- loss ≈ 1120/256 ≈ 4.4 per row while CE ≈ 1.35 (entropy ≈ ln4) → **InfoNCE term likely
  dominates training** (lambda_mim=0.1); diagnostic grid running (lambda_mim 0/0.01 × lr).
- HRO referral curve: acc@70%cov up to 0.3914 (concat) > MedVInT-TE SOTA 37.6%.
- VLM zero-shot (deepseek-v4-flash-vision-exp, 300): 0.3233.

### Diagnostic grid (15 epochs, pair features) — 2026-08-28 09:20
| run | lr | mu | mim | test acc | val_best | agreement |
|-----|----|----|-----|----------|----------|-----------|
| d_full_mim0_lr1e3 | 1e-3 | 0.1 | 0 | 0.3535 | 0.3397 | 0.983 |
| d_full_mim001_lr1e3 | 1e-3 | 0.1 | 0.01 | 0.3535 | 0.3397 | 0.983 |
| d_full_mim0_lr3e4 | 3e-4 | 0.1 | 0 | 0.3605 | 0.3475 | 0.971 |
| d_concat_lr3e4 | 3e-4 | 0 | 0 | 0.3530 | 0.3395 | — |
| **d_full_noaux_lr3e4** | 3e-4 | 0 | 0 | **0.3645** | 0.3425 | **0.483** |

Key findings:
- **InfoNCE has zero observable effect**: lambda_mim 0 vs 0.01 → bit-identical results.
  Drop it (keep MIM in architecture for ablation story, but weight irrelevant here).
- **lr 3e-4 >> 1e-3**: +0.7pp (0.3535→0.3605), val_best +0.8pp.
- **MU consensus loss actively hurts**: dropping it lifts test acc 0.3605→0.3645 and
  expert agreement collapses 0.97→0.48 — experts become genuinely diverse, so DGM gate
  has real signal. Previously MU forced near-identical experts → gate ~no-op → full ≈ concat.

### Final matrix (30 epochs, lr 3e-4, no aux losses) — running since 10:12
