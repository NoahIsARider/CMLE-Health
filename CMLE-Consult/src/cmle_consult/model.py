"""ConsultNet — organization-theoretic multimodal consult model.

Maps Galbraith (1973) information-processing theory + Joseph et al. mutual
understanding (HRO healthcare) onto a lightweight architecture:

  * role LoRA experts  = organizational units (clinician / radiologist /
                         pathologist / attending)
  * DGM                = dynamic allocation of decision rights (architecture
                         of attention) per patient case
  * MU                 = cross-expert agreement (mutual understanding) via
                         a consensus loss (KL between expert distributions)
  * HRO referral       = low-confidence cases are referred (evaluation-time
                         entropy threshold -> accuracy @ coverage)
  * InfoNCE            = cross-modal alignment between question and image
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RoleExpert(nn.Module):
    """A role-specific option scorer: (B, n_opts, in_dim) -> (B, n_opts)."""

    def __init__(self, in_dim: int, n_opts: int = 4, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.n_opts = n_opts
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, n_opts, in) -> (B, n_opts, 1) -> (B, n_opts)
        return self.net(x).squeeze(-1)


class ConsultNet(nn.Module):
    def __init__(self, text_dim: int = 768, img_dim: int = 768, opt_dim: int = 768,
                 n_opts: int = 4, hidden: int = 256, proj_dim: int = 256,
                 dropout: float = 0.1,
                 use_universal: bool = True,
                 use_specialized: bool = True,
                 use_dgm: bool = True):
        super().__init__()
        self.n_opts = n_opts
        self.use_universal = use_universal
        self.use_specialized = use_specialized
        self.use_dgm = use_dgm

        self.expert_t = RoleExpert(text_dim + opt_dim, n_opts, hidden, dropout)      # clinician  (text-centric)
        self.expert_v = RoleExpert(img_dim + opt_dim, n_opts, hidden, dropout)       # radiologist (vision-centric)
        if use_specialized:
            self.expert_a = RoleExpert(text_dim + img_dim + opt_dim, n_opts, hidden, dropout)  # pathologist (alignment)
        if use_universal:
            self.expert_u = RoleExpert(text_dim + img_dim + opt_dim, n_opts, hidden, dropout)  # attending   (generalist)

        n_experts = 2 + int(use_specialized) + int(use_universal)
        self.n_experts = n_experts
        if use_dgm:
            self.gate = nn.Sequential(
                nn.Linear(text_dim + img_dim, 64),
                nn.GELU(),
                nn.Linear(64, n_experts),
            )
        # cross-modal alignment (InfoNCE) projections
        self.proj_q = nn.Linear(text_dim, proj_dim)
        self.proj_img = nn.Linear(img_dim, proj_dim)

    # ------------------------------------------------------------------
    def _expert_logits(self, q, img, opt):
        """q/img: (B, d); opt: (B, n_opts, d). Returns list of (B, n_opts)."""
        B = q.size(0)
        qe = q.unsqueeze(1).expand(B, self.n_opts, -1)
        imge = img.unsqueeze(1).expand(B, self.n_opts, -1)
        logits, names = [], []
        logits.append(self.expert_t(torch.cat([qe, opt], dim=-1))); names.append("t")
        logits.append(self.expert_v(torch.cat([imge, opt], dim=-1))); names.append("v")
        if self.use_specialized:
            logits.append(self.expert_a(torch.cat([qe, imge, opt], dim=-1))); names.append("a")
        if self.use_universal:
            logits.append(self.expert_u(torch.cat([qe, imge, opt], dim=-1))); names.append("u")
        return logits, names

    def forward(self, q, img, opt):
        """
        q, img: (B, 768); opt: (B, 4, 768)
        returns dict: logits (B,4), expert_logits (B,n_exp,4), expert_probs,
                      agreement (B,), gate_w (B,n_exp), rep_q, rep_img
        """
        B = q.size(0)
        logits, names = self._expert_logits(q, img, opt)
        stacked = torch.stack(logits, dim=1)                     # (B, n_exp, 4)

        if self.use_dgm:
            w = torch.softmax(self.gate(torch.cat([q, img], dim=-1)), dim=-1)   # (B, n_exp)
        else:
            w = torch.full((B, self.n_experts), 1.0 / self.n_experts, device=q.device)

        final = (w.unsqueeze(-1) * stacked).sum(dim=1)           # (B, 4)
        probs = torch.softmax(stacked, dim=-1)                   # (B, n_exp, 4)

        agreement = self.agreement_score(probs)                  # (B,)

        return {
            "logits": final,
            "expert_logits": stacked,
            "expert_probs": probs,
            "agreement": agreement,
            "gate_w": w,
            "rep_q": self.proj_q(q),
            "rep_img": self.proj_img(img),
            "expert_names": names,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def load_balance_loss(gate_w: torch.Tensor) -> torch.Tensor:
        """Switch-Transformer style aux loss: penalize expert usage imbalance.

        gate_w: (B, n_experts) softmax gate weights. Encourages uniform routing
        so role experts actually specialize instead of collapsing to one.
        """
        n = gate_w.size(1)
        B = gate_w.size(0)
        f = torch.zeros(n, device=gate_w.device)
        idx = gate_w.argmax(-1)
        if B > 0:
            f = idx.bincount(minlength=n).float() / B          # hard-assignment fraction
        P = gate_w.mean(0)                                     # avg gate probability
        return n * (f * P).sum()

    @staticmethod
    def agreement_score(probs: torch.Tensor) -> torch.Tensor:
        """Mutual-understanding agreement: 1 - mean pairwise KL (exponentiated)."""
        n = probs.size(1)
        if n <= 1:
            return torch.ones(probs.size(0), device=probs.device)
        kls = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                p = probs[:, i].clamp_min(1e-8)
                q = probs[:, j].clamp_min(1e-8)
                kls.append((p * (p.log() - q.log())).sum(-1))
        mean_kl = torch.stack(kls, -1).mean(-1)                  # (B,)
        return torch.exp(-mean_kl)                               # (B,) in (0,1]

    @staticmethod
    def consensus_loss(probs: torch.Tensor) -> torch.Tensor:
        """Mean pairwise KL across experts — minimized => mutual understanding."""
        n = probs.size(1)
        if n <= 1:
            return torch.tensor(0.0, device=probs.device)
        total = 0.0
        cnt = 0
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                p = probs[:, i].clamp_min(1e-8)
                q = probs[:, j].clamp_min(1e-8)
                total = total + (p * (p.log() - q.log())).sum(-1).mean()
                cnt += 1
        return total / cnt

    @staticmethod
    def infonce_loss(rep_q: torch.Tensor, rep_img: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
        """InfoNCE: question <-> image alignment across the batch."""
        q = F.normalize(rep_q, dim=-1)
        v = F.normalize(rep_img, dim=-1)
        logits = q @ v.t() / tau                                # (B, B)
        labels = torch.arange(q.size(0), device=q.device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2
        return loss
