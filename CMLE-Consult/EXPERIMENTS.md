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

### Final matrix (30 epochs, lr 3e-4, no aux losses) — 2026-08-28 11:04 done
| variant | test acc | acc@90% | acc@70% | val_best | agreement |
|---------|----------|---------|---------|----------|-----------|
| bert-only | 0.3110 | 0.3167 | 0.3236 | 0.3298 | 1.0 |
| clip-only | **0.3670** | 0.3733 | 0.3893 | 0.3483 | 1.0 |
| concat | 0.3610 | 0.3683 | 0.3836 | 0.3443 | 1.0 |
| full | 0.3565 | 0.3633 | 0.3836 | 0.3444 | 0.147 |
| w-o-dgm | 0.3615 | 0.3633 | 0.3843 | 0.3496 | 0.847 |
| w-o-univ | 0.3610 | 0.3717 | **0.4007** | 0.3498 | 0.251 |
| w-o-spec | 0.3610 | 0.3717 | **0.4007** | 0.3498 | 0.251 |

### Gate diagnosis (analyze_gate.py on f_full.pt, test set)
- expert solo acc: t=0.272, v=0.3475, a=0.2745, u=0.2735
- **mean gate_w = [0.165, 0.831, 0.003, 0.001], gate entropy 0.40** → gate collapsed to
  almost-always-pick-v (visual expert). No role division learned; a/u experts get ~0 weight.
- Why full < clip-only: v expert (same 1536-d img+opt input as clip-only) reaches only
  0.3475 vs clip-only 0.367 — shared training + gate dilution weakens it; t/a/u experts
  add noise, not complementarity.
- verdict: classic MoE mode-collapse; needs load-balancing loss / two-stage expert
  pretrain / confidence-aware gate (see next).
- MedVInT-TE SOTA: 37.6% (our earlier note) — Frontiers 2026 review lists 40.2%;
  both > our best 36.7% acc, but HRO referral acc@70%cov reaches 0.4007 (w-o-univ/spec)
  and 0.3893 (clip-only).

### Load-balancing loss probe (15 epochs, lr 3e-4, noaux) — 2026-08-28 11:30
| run | lambda-balance | test acc | val_best | gate_w | gate entropy | agreement |
|-----|----------------|----------|----------|--------|--------------|-----------|
| d_full_noaux_lr3e4 (ref) | 0 | 0.3645 | 0.3425 | [0.17, 0.83, 0.00, 0.00] | 0.40 | 0.48 |
| b_full_lb002 | 0.02 | 0.3625 | 0.3438 | [0.24, 0.31, 0.21, 0.23] | 1.21 | 0.86 |
| b_full_lb01 | 0.1 | **0.3655** | 0.3445 | [0.25, 0.28, 0.23, 0.24] | 1.27 | 0.85 |

Load balancing kills the mode collapse: gate entropy 0.40 → 1.27 (near-uniform with
slight v edge); every expert gets real weight; fused acc 0.3655 > best expert solo
(v 0.3275) — gating now adds value. Experts solo: t=0.297, v=0.328, a=0.287, u=0.303
(v remains strongest, as expected — image dominates PMC-VQA).

### Balance 30-epoch (lr 3e-4, noaux) — 2026-08-28 12:25
| run | lb | test acc | acc@70% | val_best |
|-----|----|----------|---------|----------|
| b30_full_lb01 | 0.1 | 0.3655 | 0.3850 | 0.3477 |
| b30_full_lb03 | 0.3 | 0.3620 | 0.3871 | 0.3486 |

30ep ≈ 15ep (0.3655 both): saturates ~epoch 15. lb=0.1 is the main config
(30ep for the paper table). Final matrix v2 (main + ablations) launched 13:31.
