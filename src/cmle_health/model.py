"""CMLE-Health: Collaborative Multi-LoRA Experts for multimodal health misinformation detection.

Faithful migration of CMLE (IEEE TCE 2026, DOI 10.1109/TCE.2026.3677445) to the health domain:

  - Universal LoRA expert  : cross-modal global representation (LoRA on concat [text; image] tokens)
  - Text LoRA expert       : health-claim textual semantics
  - Image LoRA expert      : medical / AI-generated image cues
  - Consistency LoRA expert: text-image coherence cues (replaces the comment expert of CMLE;
                             health datasets rarely carry social comments)
  - MIM                    : InfoNCE alignment between universal and specialized experts
  - DGM                    : dynamic gating mechanism (token-level softmax fusion)
  - Multi-task heads       : Task A reliability (reliable/unreliable) + Task B originality (human/AI)

All backbones are frozen; only LoRA adapters (rank r) + projections + gate + heads are trained.
Ablation switches (use_* / variant flags) replicate the CMLE paper's Table II variants.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAAdapter(nn.Module):
    """Residual low-rank adapter: y = x + (B @ A) @ x, with r << d."""

    def __init__(self, d: int, r: int = 8, scale: float = 1.0):
        super().__init__()
        self.r = r
        self.scale = scale
        self.A = nn.Parameter(torch.zeros(d, r))
        self.B = nn.Parameter(torch.zeros(r, d))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scale * (x @ self.A @ self.B)


class CMLEHealth(nn.Module):
    """Collaborative Multi-LoRA Experts for multimodal health misinformation detection.

    Args:
        text_dim, image_dim  : frozen backbone feature dims
        proj_dim             : shared projection dim
        lora_rank            : LoRA rank r
        use_consistency      : enable the consistency expert (CMLE's comment-expert replacement)
        use_universal        : universal expert on/off  (ablation: w/o universal)
        use_specialized      : specialized experts on/off (ablation: w/o specialized)
        use_dgm              : dynamic gating vs equal weights (ablation: w/o DGM)
        use_mim              : InfoNCE term on/off (ablation: w/o MIM)
    """

    def __init__(
        self,
        text_dim: int = 768,
        image_dim: int = 768,
        proj_dim: int = 512,
        lora_rank: int = 8,
        num_classes_a: int = 2,   # reliability
        num_classes_b: int = 2,   # originality
        num_classes_c: int = 0,   # fine-grained AI detection (25 combos), 0 = disabled
        dropout: float = 0.1,
        use_consistency: bool = True,
        use_universal: bool = True,
        use_specialized: bool = True,
        use_dgm: bool = True,
        use_mim: bool = True,
    ):
        super().__init__()
        self.proj_dim = proj_dim
        self.use_consistency = use_consistency
        self.use_universal = use_universal
        self.use_specialized = use_specialized
        self.use_dgm = use_dgm
        self.use_mim = use_mim

        self.text_proj = nn.Linear(text_dim, proj_dim)
        self.image_proj = nn.Linear(image_dim, proj_dim)

        # --- LoRA experts ---
        self.universal_lora = LoRAAdapter(proj_dim, lora_rank)
        self.text_lora = LoRAAdapter(proj_dim, lora_rank)
        self.image_lora = LoRAAdapter(proj_dim, lora_rank)
        if use_consistency:
            self.consistency_lora = LoRAAdapter(proj_dim, lora_rank)

        # --- Dynamic Gating ---
        n_exp = sum([1 if self.use_universal else 0,
                     1 if self.use_specialized else 0,
                     1 if (self.use_specialized and self.use_consistency) else 0])
        # gate outputs weights for: universal / text / image / consistency
        self.n_exp = n_exp
        self.gate = nn.Linear(proj_dim, 4)   # always 4 slots; unused slots masked

        self.dropout = nn.Dropout(dropout)

        # --- Task heads ---
        self.head_a = nn.Linear(proj_dim, num_classes_a)
        self.head_b = nn.Linear(proj_dim, num_classes_b)
        if num_classes_c > 0:
            self.head_c = nn.Linear(proj_dim, num_classes_c)

    # ------------------------------------------------------------------ #
    def _project(self, feats, proj):
        return self.dropout(F.gelu(proj(feats)))

    def forward(self, text_feats, image_feats):
        """text_feats: (B, L_t, text_dim) frozen BERT token features
           image_feats: (B, L_i, image_dim) frozen CLIP patch features (or None)"""
        ft = self._project(text_feats, self.text_proj) if text_feats is not None else None
        if image_feats is not None:
            fi = self._project(image_feats, self.image_proj)    # (B, L_i, D)
        else:
            fi = None

        # expert outputs
        e_t = self.text_lora(ft) if ft is not None else None
        if fi is not None:
            e_i = self.image_lora(fi)
        else:
            e_i = None

        if self.use_universal:
            feats_all = torch.cat([ft, fi], dim=1) if (ft is not None and fi is not None) \
                else (ft if ft is not None else fi)
            u = self.universal_lora(feats_all)
        else:
            u = None

        if self.use_specialized and self.use_consistency:
            feats_all = torch.cat([ft, fi], dim=1) if (ft is not None and fi is not None) \
                else (ft if ft is not None else fi)
            e_c = self.consistency_lora(feats_all)
        else:
            e_c = None

        # ---- fusion with gating ----
        # build per-token expert output stack and gate weights over active experts
        active = []
        labels = []
        if u is not None:
            active.append(u)
            labels.append("U")
        if e_t is not None and self.use_specialized:
            active.append(e_t)
            labels.append("T")
        if e_i is not None and self.use_specialized:
            active.append(e_i)
            labels.append("I")
        if e_c is not None:
            active.append(e_c)
            labels.append("C")

        # all active experts share the same token layout when combined over [text;image]
        # (u / e_c cover all tokens; e_t covers text tokens; e_i covers image tokens)
        if fi is not None and ft is not None:
            # align to full token grid: text part + image part
            def pad_to_full(part, full_len):
                Lp = part.size(1)
                if Lp == full_len:
                    return part
                pad = torch.zeros(part.size(0), full_len - Lp, part.size(2), device=part.device)
                return torch.cat([part, pad], dim=1)

            L = ft.size(1) + fi.size(1)
            e_t_full = pad_to_full(e_t, L)
            e_i_full = pad_to_full(e_i, L)
            stack = []
            for lab, e in zip(labels, active):
                if lab == "T":
                    stack.append(e_t_full)
                elif lab == "I":
                    # place image expert output in the image segment
                    e_i_seg = torch.cat([torch.zeros(ft.size(0), ft.size(1), self.proj_dim, device=ft.device), e_i], dim=1)
                    stack.append(e_i_seg)
                elif lab == "U":
                    stack.append(u)
                elif lab == "C":
                    stack.append(e_c)
            expert_stack = torch.stack(stack, dim=-2)          # (B, L, n, D)
            if self.use_dgm:
                g_logits = self.gate(feats_all)                # (B, L, 4)
                g = F.softmax(g_logits, dim=-1)                # (B, L, 4)
                # map active-expert order to gate slots [U, T, I, C]
                slot = {"U": 0, "T": 1, "I": 2, "C": 3}
                g_act = torch.stack([g[:, :, slot[lab]] for lab in labels], dim=-1)  # (B, L, n)
            else:
                g_act = torch.full_like(expert_stack[..., 0], 1.0 / len(labels))   # equal weights
            h = (expert_stack * g_act.unsqueeze(-1)).sum(dim=-2)  # (B, L, D)
        elif fi is not None:
            # image only: experts = U (over image), I, C (over image)
            stack = []
            for lab, e in zip(labels, active):
                stack.append(e)
            expert_stack = torch.stack(stack, dim=-2)
            if self.use_dgm:
                g_logits = self.gate(fi)
                g = F.softmax(g_logits, dim=-1)
                slot = {"U": 0, "T": 1, "I": 2, "C": 3}
                g_act = torch.stack([g[:, :, slot[lab]] for lab in labels], dim=-1)
            else:
                g_act = torch.full_like(expert_stack[..., 0], 1.0 / len(labels))
            h = (expert_stack * g_act.unsqueeze(-1)).sum(dim=-2)
        else:
            # text only: experts = U (over text), T, C (over text)
            stack = []
            for lab, e in zip(labels, active):
                stack.append(e)
            expert_stack = torch.stack(stack, dim=-2)
            if self.use_dgm:
                g_logits = self.gate(ft)
                g = F.softmax(g_logits, dim=-1)
                slot = {"U": 0, "T": 1, "I": 2, "C": 3}
                g_act = torch.stack([g[:, :, slot[lab]] for lab in labels], dim=-1)
            else:
                g_act = torch.full_like(expert_stack[..., 0], 1.0 / len(labels))
            h = (expert_stack * g_act.unsqueeze(-1)).sum(dim=-2)

        h_fused = h.mean(dim=1)                                 # mean pooling (paper Eq. 11-12)
        out = {
            "logits_a": self.head_a(h_fused),
            "logits_b": self.head_b(h_fused),
            "fused": h_fused,
        }
        if hasattr(self, "head_c"):
            out["logits_c"] = self.head_c(h_fused)

        # pooled expert reps for MIM
        reps = {}
        if u is not None:
            reps["U"] = u.mean(dim=1)
        if self.use_specialized:
            if e_t is not None:
                reps["T"] = e_t.mean(dim=1)
            if e_i is not None:
                reps["I"] = e_i.mean(dim=1)
            if e_c is not None:
                reps["C"] = e_c.mean(dim=1)
        out["reps"] = reps
        return out

    # ------------------------------------------------------------------ #
    def infonce_loss(self, reps: dict, tau: float = 0.07) -> torch.Tensor:
        """In-batch InfoNCE (paper Eq. 8): align U with each specialized expert E_m."""
        if "U" not in reps or not self.use_mim:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        u = F.normalize(reps["U"], dim=-1)
        loss = torch.tensor(0.0, device=u.device)
        n = 0
        for key in ("T", "I", "C"):
            if key not in reps:
                continue
            e = F.normalize(reps[key], dim=-1)
            sim = u @ e.T / tau
            labels = torch.arange(sim.size(0), device=sim.device)
            loss = loss + (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
            n += 1
        return loss / max(n, 1)


class BaselineMLP(nn.Module):
    """Simple baselines: BERT-only / CLIP-only / concat fusion (no experts)."""

    def __init__(self, in_dim: int, num_classes: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)
