"""models/hgt.py — Heterogeneous Graph Transformer encoder (pure PyTorch)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class HGTLayer(nn.Module):
    """One HGT layer with 9 type-specific K/Q/V projections."""

    def __init__(self, h: int):
        super().__init__()
        self.h = h
        self.scale = h ** -0.5

        # 3 source types × 3 relations → separate K and V per (src,rel)
        for rel in ["pp", "ap", "pk"]:
            setattr(self, f"K_{rel}", nn.Linear(h, h, bias=False))
            setattr(self, f"V_{rel}", nn.Linear(h, h, bias=False))

        # 3 destination types → separate Q per dst_type
        for nt in ["paper", "author", "keyword"]:
            setattr(self, f"Q_{nt}", nn.Linear(h, h, bias=False))
            setattr(self, f"W_{nt}", nn.Linear(h, h))

        self.ln_p = nn.LayerNorm(h)
        self.ln_k = nn.LayerNorm(h)
        self.drop = nn.Dropout(0.1)

    def _attend(self, K, Q_h, V, si, di, n_dst, ew=None):
        if si.numel() == 0:
            return torch.zeros(n_dst, self.h, device=K.device)
        k = K[si]; q = Q_h[di]; v = V[si]
        attn = torch.tanh((k * q).sum(-1) * self.scale)
        if ew is not None:
            attn = attn * ew.clamp(0, 2)
        msg = attn.unsqueeze(1) * v
        out = torch.zeros(n_dst, self.h, device=k.device)
        out.index_add_(0, di, msg)
        cnt = torch.zeros(n_dst, 1, device=k.device)
        cnt.index_add_(0, di, torch.ones(len(si), 1, device=k.device))
        return out / cnt.clamp(min=1)

    def forward(self, hp, ha, hk, r1, r2, r3):
        s1, d1, w1 = r1
        s2, d2, _  = r2
        s3, d3, w3 = r3

        # Paper ← cites (R1) + wrote (R2)
        m1 = self._attend(self.K_pp(hp), self.Q_paper(hp),
                          self.V_pp(hp), s1, d1, hp.size(0), w1)
        m2 = self._attend(self.K_ap(ha), self.Q_paper(hp),
                          self.V_ap(ha), s2, d2, hp.size(0))
        hp2 = self.ln_p(hp + self.drop(self.W_paper(m1 + m2)))

        # Keyword ← tagged (R3)
        m3  = self._attend(self.K_pk(hp), self.Q_keyword(hk),
                           self.V_pk(hp), s3, d3, hk.size(0), w3)
        hk2 = self.ln_k(hk + self.drop(self.W_keyword(m3)))

        return hp2, ha, hk2


class HGTEncoder(nn.Module):
    """Full HGT: input projections + L stacked HGT layers."""

    def __init__(self, in_dims: dict, hidden: int = 128,
                 n_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.proj_p = nn.Sequential(
            nn.Linear(in_dims["paper"],   hidden),
            nn.LayerNorm(hidden), nn.ReLU())
        self.proj_a = nn.Sequential(
            nn.Linear(in_dims["author"],  hidden),
            nn.LayerNorm(hidden), nn.ReLU())
        self.proj_k = nn.Sequential(
            nn.Linear(in_dims["keyword"], hidden),
            nn.LayerNorm(hidden), nn.ReLU())
        self.layers = nn.ModuleList([HGTLayer(hidden) for _ in range(n_layers)])
        self.drop   = nn.Dropout(dropout)

    def forward(self, feat: dict, graph: dict) -> torch.Tensor:
        dev = next(self.parameters()).device
        hp = self.drop(self.proj_p(feat["paper"].to(dev)))
        ha = self.drop(self.proj_a(feat["author"].to(dev)))
        hk = self.drop(self.proj_k(feat["keyword"].to(dev)))
        for layer in self.layers:
            hp, ha, hk = layer(hp, ha, hk,
                                graph["r1"], graph["r2"], graph["r3"])
        return hp   # (N_papers, hidden)
