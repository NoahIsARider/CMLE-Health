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

- All experts are **LoRA adapters** (rank 8) on frozen backbones → trainable params are tiny,
  single consumer GPU (RTX 4060-class) suffices.
- Multi-task: reliability check (reliable/unreliable) + originality check (human/AI) jointly.
- Benchmarked on **MM-Health** (34,746 articles, 5,776 human + 28,880 AI) with CoAID transfer checks.

## Status

🚧 Active development (2026-08-26)

## License

MIT (code). Dataset: MM-Health is CC BY-NC 4.0 (research use only).

## Citation

If you use this work, please cite both the original CMLE paper and the MM-Health dataset paper
(see `CITATION.cff` / README).
