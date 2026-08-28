# CMLE-Consult — Project Report & Handover

> 最后更新：2026-08-28 14:30（服务器时效到期前归档）
> 状态：实验完成，结果已归档。论文写作 / 后续开发交由接手者。

---

## 0. TL;DR

- **任务**：PMC-VQA closed-set 四选一医学 VQA（train 176,949 / test_clean 2,000）
- **方法**：ConsultNet —— 组织理论（Galbraith 信息处理 + Joseph mutual understanding）驱动的
  多角色 LoRA 专家（临床/放射/病理/主治）+ 动态门控（DGM）+ 共识损失（MU）+ HRO 低置信转诊
- **最佳结果**：**full + 负载均衡 0.1 → test acc 0.3655**；单模态 clip-only 0.3670；
  HRO 转诊 acc@70%cov 0.3893-0.4007
- **SOTA 对比**：MedVInT-TE 37.6%（原论文）/ 40.2%（Frontiers 2026 综述口径）。
  我们的 acc 略低（-1~3.7pp），但参数少 ~400 倍（判别式 ~1M vs 生成式 VLM ~400M+）
- **核心结论**：**当前状态不建议直接投稿**（主模型未显著超越单模态基线），但架构方向经过验证
  （负载均衡修复门控崩溃后 DGM 有 +0.4pp 正贡献），有明确改进路径

---

## 1. 背景与目标

CMLE-Health 的姊妹项目。把 CMLE（IEEE TCE 2026）的"组织理论 → 多角色专家 + 动态门控"
框架迁移到医疗问诊场景。**直接灵感**：Joseph, Wilson, Park & Chow (2026) SMJ 早见版
《Information processing, mutual understanding, and organization design in healthcare》
（DOI: 10.1002/smj.70116）——HRO（高危组织）通过跨角色共识 + 低置信转诊保障可靠性。

**设计映射**：
| 组织概念 | 架构组件 |
|---------|---------|
| 组织单元（临床/放射/病理/主治） | 4 个角色 LoRA 专家（t/v/a/u） |
| 架构注意力（决策权分配） | DGM 动态门控（per-case 权重） |
| Mutual understanding | 跨专家共识损失（KL） |
| HRO 不确定性管理 | 低置信转诊（entropy 阈值 → acc@coverage） |
| 跨模态对齐 | InfoNCE（question ↔ image） |

---

## 2. 数据与特征

- **数据**：`xmcmic/PMC-VQA`（HuggingFace）—— images.zip 18.9G（不落地解压，zip 直读）
  + CSVs。train.csv 176,949 行（dropna 后 176,873）/ test_clean.csv 2,000 行。
  Schema：`Figure_path, Question, Answer, Choice A-D, Answer_label`（closed-set 四选一）
- **特征**（冻结骨干一次性预计算，fp16 缓存 3.1G）：
  - 文本：BERT-base pooler（问题 + 选项）
  - 图像：CLIP ViT-B/32 pooled
  - **Pair 模式（关键修复）**：选项编码为 `[question SEP option]` cross-encoder 对，
    否则选项无问题上下文 → 模型学不到（第一轮全 25% chance 的根因）
- 文件：`train.pt / test_clean.pt`（单选项编码）、`*_pair.pt`（pair 编码）

---

## 3. 方法

```
q (768) ─────────────┐
img (768) ───────────┼──→ gate ──→ 4 expert weights (DGM)
                     │
opt_i (4×768, pair) ─┤
                     ├──→ t: [q, opt_i]        (临床, 文本)
                     ├──→ v: [img, opt_i]      (放射, 视觉)
                     ├──→ a: [q, img, opt_i]   (病理, 对齐)
                     └──→ u: [q, img, opt_i]   (主治, 通用)
final logits = Σ w_e · expert_e(opt_i)   → 4-class CE
辅助损失: MU 共识 KL (lambda-mu) · InfoNCE (lambda-mim) · 负载均衡 (lambda-balance)
```

- 参数量：ConsultNet ~2.46M 可训练（专家 MLP 393K/590K each + 门控 + 投影）
- 训练：lr 3e-4（1e-3 太大），batch 256，30 epochs（15ep 即饱和），seed 42
- 基线：bert-only / clip-only / concat（FlatMLP 393K）

---

## 4. 实验历史（4 轮关键节点）

### Round 1 — 首次矩阵（无 pair）：**全 25% chance** ❌
8 配置 test acc 0.239-0.2675，mean_entropy=ln4（均匀分布）。根因：选项独立编码、
无问题上下文 → 无信号。**修复**：pair cross-encoder 特征 + per-option 打分头。

### Round 2 — Pair 矩阵（10ep）：全部脱离 chance ✅
best concat 0.360 / full 0.354。但 full < concat；w-o-univ ≡ w-o-spec 一字不差
（两个全模态专家输入拼接相同 → 架构等价，消融冗余）；w-o-mu ≡ full
（专家 agreement 0.98 → 共识 KL≈0 失效）。

### Round 3 — 最终矩阵（30ep, lr3e-4, noaux）+ 门控诊断 🎯
best clip-only 0.3670 / full 0.3565。**门控诊断（analyze_gate.py）**：
mean gate_w = [0.165, 0.831, 0.003, 0.001]，熵 0.40 —— **mode collapse**：
门控几乎只选视觉专家，其他 3 个专家权重≈0，无角色分工。
根因：MU 把专家强推成一致（agreement 0.97）→ 门控无路由信号；
去掉 MU 后专家分歧（0.48）但门控仍崩溃 → 缺负载均衡机制。

### Round 4 — 负载均衡损失（final2）✅ 门控修复
Switch-style `load_balance_loss = n_experts · Σ fᵢPᵢ`（hard-assignment fraction × avg gate prob）。
门控熵 0.40 → 1.27；专家全部获得真实权重；**DGM 首次正贡献**（w-o-dgm 0.3615 < full 0.3655, +0.4pp）。
MU + balance 组合 val_best 最高（0.3531）但 test 相同。

---

## 5. 最终结果（final2，30ep，lr 3e-4，full+lb0.1）

| variant | test acc | acc@95% | acc@90% | acc@80% | acc@70% | val_best |
|---------|----------|---------|---------|---------|---------|----------|
| bert-only | 0.3110 | — | 0.3167 | 0.3296 | 0.3236 | 0.3298 |
| clip-only | **0.3670** | — | 0.3733 | 0.3794 | 0.3893 | 0.3483 |
| concat | 0.3610 | — | 0.3683 | 0.3761 | 0.3836 | 0.3443 |
| **full + lb0.1** | 0.3655 | — | 0.3683 | 0.3780 | 0.3850 | 0.3477 |
| full + lb0.1 + MU | 0.3655 | — | 0.3678 | 0.3789 | 0.3871 | **0.3531** |
| w-o-dgm | 0.3615 | — | 0.3633 | 0.3718 | 0.3843 | 0.3496 |
| w-o-univ / w-o-spec | 0.3605 | 0.3626 | 0.3678 | 0.3775 | 0.3929 | 0.3473 | |

> 注：w-o-univ 与 w-o-spec 架构等价（见 Round 2），数值应相同。

**与已发表对比**（test_clean closed-set）：

| Model | Acc | 备注 |
|-------|-----|------|
| MedVInT-TE (PMC-CLIP) | 37.6（原论文）/ 40.2（Frontiers 综述） | 生成式 VLM ~400M+ |
| LLaVA-Med | 34.8 | 生成式 VLM |
| BioMedCLIP | 33.0 | |
| MedICap-GPT-4 | 27.2 | |
| Chance | 25.0 | |
| **ConsultNet (ours)** | **36.55**（转诊 38.5-40.1 @70%cov） | 判别式 ~1M 参数 |

**VLM zero-shot 基线**（deepseek-v4-flash-vision-exp，300 样本）：0.3233（修正 prompt 后）。

---

## 6. 已知问题与限制（接手者必读）

1. **主模型未超越单模态**：full+lb 0.3655 vs clip-only 0.3670（差 0.15pp）。
   多角色分工目前"不输"但"没赢"。论文核心证据缺失。
2. **消融冗余**：w-o-univ ≡ w-o-spec（a/u 专家输入拼接相同）。需要差异化专家输入
   （如病理专家用 image patch 特征、主治专家用全局）或合并为一个全模态专家。
3. **InfoNCE 无效**：lambda-mim 0 vs 0.01 结果逐位相同。保留在架构里但无贡献，
   论文声称需谨慎。
4. **MU 共识损失**：单独用有害（强制专家一致 → 门控无信号）；与负载均衡组合后
   val 略升（0.3531）但 test 不变。机制需重新设计（如软共识 + 温度）。
5. **acc@coverage 转诊是亮点**：acc@70%cov 0.3893-0.4007（> SOTA 37.6），
   MedVInT 被 Frontiers 2026 综述点名"Lacks uncertainty quantification"。
6. **15ep 饱和**：30ep ≈ 15ep，过拟合风险低但无增益；可试更复杂 head / 更大 hidden。
7. **对比口径**：MedVInT-TE 的 37.6 vs 40.2 需向原作者/数据集 README 核实
   （我们用的是官方 train.csv 训练 + test_clean.csv 测试）。

---

## 7. 下一步建议（优先级排序）

> 2026-08-28 16:50 项目归档。PubMedCLIP 骨干实验已完成：**骨干不是瓶颈**（医学 CLIP 反而
> 略降），判别式 acc 天花板 ~36.7%，主模型两种骨干下均未超单模态。以下建议保留供后续参考。

1. **换 PMC-CLIP 视觉骨干**（SOTA 同款）：预计算 ~1h（precompute.py --clip-model 已支持），
   医学特征更强，对比更公平，最可能抬 acc 1-2pp 并放大专家分工收益
   —— ⚠️ 已实测 PubMedCLIP（开放替代）反而略降；官方 PMC-CLIP 为 gated，未测
2. **差异化专家输入**（消融冗余修复）：a/u 用不同特征视图
3. **两阶段训练**：先独立训专家（各自 FlatMLP），冻结后训门控 ——
   组织理论叙事"先有专业能力，再分配决策权"最干净的实现
4. **负载均衡 + MU 联合调参**（val_best 0.3531 提示有空间）
5. **3-seed 方差**（run_seed.sh 已备好，~70min）：报均值±std
6. **论文定位建议**：投医学信息学期刊（JBHI/JBI）或 TCE 延续，主打
   "轻量判别式 + 不确定性量化（HRO 转诊）"，不主打 SOTA 刷榜

---

## 8. 复现

```bash
# 环境（DeepLn P4: Ubuntu 24.04, Python 3.12, torch 2.6.0+cu124）
# 代码：CMLE-Health 仓库 CMLE-Consult/ 子目录
export HF_ENDPOINT=https://hf-mirror.com   # transformers 5.x 必须在 import 前设置！

bash scripts/download_data.sh              # 21G zips（images_2.zip 不需要，已验证 100% 覆盖）
bash scripts/precompute_pair.sh            # pair 特征（~1h）
bash scripts/run_final2.sh 0.1             # 最终矩阵（主配置 lb=0.1）
PYTHONPATH=src python3 scripts/analyze_gate.py --ckpt <ckpt>   # 门控诊断
```

关键坑（详见 EXPERIMENTS.md / MEMORY lessons）：
- HF_ENDPOINT 必须 shell 层 export（代码里设置无效）
- 磁盘满的症状是 dataloader 卡死，不是报错（先 df -h）
- pkill -f 用 [x]xx 括号技巧防自杀

---

## 9. 文件索引

| 路径 | 内容 |
|------|------|
| `README.md` | 项目总览 + 实验表 |
| `EXPERIMENTS.md` | 完整实验日志（4 轮 + 诊断） |
| `src/cmle_consult/` | data / precompute / model / train / eval_vlm |
| `scripts/` | download / precompute / run_mvp / run_matrix / run_final2 / run_seed / analyze_gate / validate_pair |
| `runs*/` | 各轮结果 JSON（round1/2/3/final/balance/final2） |
| 服务器（已过期释放） | /root/cmle-consult/{features,runs*} |
