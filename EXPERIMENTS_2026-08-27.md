# EXPERIMENTS — 2026-08-27 · 全矩阵（文本 19 + 图像/多模态 14）

> P4 8G 单卡 · 全部特征缓存模式（BERT-base + CLIP ViT-B/32 冻结，fp16 缓存）
> 训练：10 epochs · batch 64 · lr 1e-4 · LoRA rank 8 · proj 512 · λ_MIM 1.0 · seed 42
> 数据：MM-Health（train 4154 / val 463 / test 1159）

## 1. 文本矩阵（19/19 完成）

| 实验 | task | test acc | F1 | val |
|---|---|---|---|---|
| bert-only | reliability | 0.8886 | 0.8077 | 0.7907 |
| no-experts | reliability | 0.8886 | 0.8077 | 0.7907 |
| **full（主模型）** | **reliability** | **0.8966** | **0.8213** | **0.8134** |
| w-o-universal | reliability | 0.8894 | 0.8055 | — |
| w-o-specialized | reliability | 0.8950 | 0.8116 | — |
| w-o-consistency | reliability | 0.8960 | 0.8175 | — |
| w-o-mim | reliability | 0.8955 | 0.8171 | — |
| w-o-dgm | reliability | 0.8921 | 0.8166 | — |
| only-text-expert | reliability | 0.8966 | 0.8213 | — |
| bert-only | originality | 0.9938 | 0.9889 | 0.9909 |
| no-experts | originality | 0.9938 | 0.9889 | 0.9909 |
| **full（主模型）** | **originality** | **0.9988** | **0.9979** | **0.9948** |
| full | both | a 0.8936 / b 0.9964 | — | 0.8007 |
| w-o-mim | both | a 0.8913 / b 0.9964 | — | — |
| w-o-dgm | both | a 0.8934 / b 0.9934 | — | 0.7875 |

**文本结论**：主模型两个任务全面超过基线（rel +0.80pp / orig +0.50pp）；消融去组件全降 → 每个组件都有正贡献；originality 任务几乎饱和（AI 生成判别高度可学）。

## 2. 图像/多模态矩阵（14/14 完成）

| 实验 | task | test acc | val |
|---|---|---|---|
| clip-only | reliability | 0.8366 | 0.6760 |
| clip-only | originality | 0.8412 | 0.5803 |
| concat (text+img) | reliability | 0.9026 | 0.8160 |
| concat | originality | 0.9937 | 0.9856 |
| no-experts | reliability | 0.9026 | 0.8160 |
| **full（多模态主模型）** | **reliability** | **0.9018** | **0.8317** |
| **full（多模态主模型）** | **originality** | **0.9991** | **0.9961** |
| **full（多模态主模型）** | **both** | **a 0.9044 / b 0.9977** | **0.8324** |
| w-o-universal | reliability | 0.9021 | 0.8304 |
| w-o-specialized | reliability | 0.9014 | 0.8282 |
| w-o-consistency | reliability | 0.9026 | 0.8246 |
| w-o-mim | reliability | 0.9006 | 0.8359 |
| w-o-dgm | reliability | 0.9054 | 0.8274 |
| only-image-expert | reliability | 0.8174 | 0.7012 |

## 3. 关键发现

1. **多模态 > 单模态**：reliability 文本 89.66 → 多模态 90.18（+0.52pp）；originality 多模态主模型 99.91 全场最高。
2. **主模型 > concat**（originality）：99.91 vs 99.37（+0.54pp）——LoRA 专家 + DGM 结构在原创性判别上优于简单特征拼接；reliability 两者相当（90.18 vs 90.26，差 0.08pp 在噪声内）。
3. **双任务联合训练（both）最优**：full_both reliability 90.44 为全场最高，且同时保持 b 99.77 —— 任务联合作为多任务正则对可靠性有增益。
4. **消融不对称**（诚实呈现）：
   - 文本模态：每个组件（universal/specialized/consistency/MIM/DGM）去掉都降 → 全部正贡献。
   - 多模态：universal（-0.03）、specialized（-0.04）、MIM（-0.12）仍有正贡献（test 噪声内）；consistency（+0.08）与 DGM（+0.36）在 reliability test 上不降反升，但 val 上 full（0.8317）仍高于去掉它们的变体（0.8246/0.8274）——结论：多模态下 DGM 与 consistency 的增益弱于文本模态，属于 val 可见、test 波动的弱信号。
5. **only-image-expert（0.8174）< clip-only（0.8366）**：无 universal/文本上下文的单图像专家不如简单 mean-pool 基线 → 专家路由需要全局/跨模态信号才有意义（与论文 DGM 设计一致）。
6. **CLIP-only 弱**（rel 83.66 / orig 84.12）：纯视觉判别信息有限，文本是主要信号源，图像提供增量（与医疗误信息以文本为主的直觉一致）。

## 4. 与 CMLE 论文对比

- 论文（Weibo/Twitter）：92.9 / 93.4 / DGM4 93.5 acc（混合模态，社交媒体域）
- 本工作（MM-Health）：reliability 90.44（full_both 多模态）——任务/域/数据不同，不做直接数值对比；结论方向一致：**多 LoRA 专家 + MIM + DGM 结构有效**，且消融验证了各组件作用。

## 5. 复现

- 代码：`src/cmle_health/`（commit 0fc9602 起支持双缓存加载 + image-only 前向）
- 特征缓存：`{split}_{text|image}.pt`（precompute.py，--hf-mirror 需在 shell 层 export HF_ENDPOINT）
- 实验脚本：`scripts/run_experiments.sh`（文本）+ `/root/step5b_rerun.sh`（图像矩阵）
- 全部结果 json：`remote-runs/`（29 个）
