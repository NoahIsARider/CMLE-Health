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

### Full matrix (10 epochs, full train)
| variant | test acc | acc@95% | acc@90% | acc@80% |
|---------|----------|---------|---------|---------|
| bert-only | | | | |
| clip-only | | | | |
| concat | | | | |
| full | | | | |
| w-o-dgm | | | | |
| w-o-mu | | | | |
| w-o-univ | | | | |
| w-o-spec | | | | |
