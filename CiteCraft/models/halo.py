"""models/halo.py — HALO: DLOG · SHLA · OAHL · PALE (feeds all 4 tasks)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from configs.config import N_DOMAIN, ANCESTOR_MAP, DOMAIN_LIST


class DLOG:
    """Dynamic Label Ontology Graph — built once from corpus co-occurrence.

    Zero learnable parameters.  One-time construction at O(N·C²).
    Outputs:
        edges   : list of (child_idx, parent_idx) hypernymy edges
        d_ont   : (C×C) Wu-Palmer distance matrix
        spec    : (C,) label specificity scores
    """

    def __init__(self, n: int = N_DOMAIN, tau: float = 0.7):
        self.n = n; self.tau = tau
        self.edges  = []
        self.d_ont  = None
        self.spec   = None

    def build(self, y_train: torch.Tensor) -> "DLOG":
        n = self.n; N = len(y_train)
        pres = (y_train[:, None] == torch.arange(n)[None, :]).float().numpy()

        # Conditional co-occurrence P(l_j | l_i)
        cooc = pres.T @ pres
        df   = pres.sum(0).clip(min=1)
        cp   = cooc / df[:, None]

        self.edges = [(i, j) for i in range(n) for j in range(n)
                      if i != j
                      and cp[i, j] > self.tau
                      and cp[i, j] > cp[j, i]]

        self.spec = torch.tensor(1 - df / N, dtype=torch.float32)

        # Wu-Palmer ontological distance
        depth = {j: 1 + sum(1 for a, b in self.edges if b == j)
                 for j in range(n)}
        d = torch.ones(n, n)
        for i in range(n):
            for j in range(n):
                if i == j:
                    d[i, j] = 0.0
                else:
                    lca = max(0, min(depth.get(i, 1), depth.get(j, 1)) - 1)
                    denom = depth.get(i, 1) + depth.get(j, 1)
                    if denom > 0:
                        d[i, j] = 1 - 2 * lca / denom
        self.d_ont = d
        return self


class HALOHead(nn.Module):
    """HALO T3 classifier: SHLA + OAHL + PALE combined.

    Learnable parameters: C × d prototype vectors (896 total at d=128).
    """

    def __init__(self, hidden: int, n: int = N_DOMAIN,
                 tau: float = 0.07, w_hier: float = 0.3,
                 lreg: float = 0.1, beta: float = 0.5,
                 dropout: float = 0.2):
        super().__init__()
        self.n = n; self.tau = tau; self.w_hier = w_hier
        self.lreg = lreg; self.beta = beta
        self.drop = nn.Dropout(dropout)
        self.prototypes = nn.Parameter(torch.randn(n, hidden) * 0.1)
        self.dlog: DLOG | None = None

    def set_dlog(self, dlog: DLOG):
        self.dlog = dlog

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Returns logits (N, C) via cosine similarity / temperature."""
        hn = F.normalize(self.drop(h), dim=-1)
        pn = F.normalize(self.prototypes, dim=-1)
        return (hn @ pn.T) / self.tau

    # ── SHLA ─────────────────────────────────────────────────────────────────
    def soft_labels(self, y_hard: torch.Tensor, device: str) -> torch.Tensor:
        """Generate SHLA soft label vectors with β=0.5 ancestor decay."""
        N = len(y_hard); n = self.n
        ys = torch.zeros(N, n, device=device)
        ys.scatter_(1, y_hard.unsqueeze(1), 1.0)
        if self.dlog and self.dlog.edges:
            for (ch, pa) in self.dlog.edges:
                mask = (y_hard == ch)
                if mask.any():
                    ys[mask, pa] += self.beta * ys[mask, ch]
        return ys / ys.sum(1, keepdim=True).clamp(min=1)

    # ── OAHL + PALE reg ───────────────────────────────────────────────────────
    def loss(self, logits: torch.Tensor, y_hard: torch.Tensor,
             device: str) -> torch.Tensor:
        """Combined OAHL + PALE prototype regularisation loss."""
        n = self.n
        ys = self.soft_labels(y_hard, device)
        lp = F.log_softmax(logits, dim=-1)
        pr = torch.exp(lp)

        # Specificity-weighted soft CE
        wh = self.dlog.spec.to(device) if self.dlog else torch.ones(n, device=device)
        l_ce = -(ys * lp * wh.unsqueeze(0)).sum(-1).mean()

        # Hierarchical confusion penalty
        if self.dlog and self.dlog.d_ont is not None:
            dont = self.dlog.d_ont.to(device)
            conf = (pr.unsqueeze(2) * ys.unsqueeze(1) *
                    dont.unsqueeze(0)).sum(dim=(1, 2))
            l_hier = self.w_hier * conf.mean()
        else:
            l_hier = torch.tensor(0., device=device)

        # PALE prototype geometry regularisation
        pn = F.normalize(self.prototypes, dim=-1)
        if self.dlog and self.dlog.d_ont is not None:
            dont = self.dlog.d_ont.to(device)
            sim  = pn @ pn.T
            l_reg = self.lreg * ((sim - (1 - dont)) ** 2).mean()
        else:
            l_reg = torch.tensor(0., device=device)

        return l_ce + l_hier + l_reg
