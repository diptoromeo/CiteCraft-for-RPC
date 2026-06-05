"""models/baselines.py — GCN, GATv2, TGN baseline models."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.heads import LinkHead, IntentHead, RankHead
from configs.config import N_DOMAIN


def _compute_loss_shared(self, h, graph, t1b, t2b, device):
    """Shared compute_loss for homogeneous baselines."""
    ti  = graph["train_mask"].nonzero(as_tuple=True)[0]
    s, d, lb = t1b
    l1 = F.binary_cross_entropy_with_logits(self.fwd_t1(h, s, d), lb.float())
    l3 = F.cross_entropy(self.fwd_t3(h[ti]), graph["y_domain"][ti], label_smoothing=0.1)
    sc, lc = self.fwd_t4(h[ti])
    l4 = self.t4.loss(sc, lc, graph["y_bucket"][ti])
    tot = l1 + l3 + 1.2 * l4
    return tot, {"T1": l1, "T2": torch.tensor(0., device=device), "T3": l3, "T4": l4}


# ── GCN ───────────────────────────────────────────────────────────────────────
class GCNLayer(nn.Module):
    def __init__(self, i, o):
        super().__init__(); self.W = nn.Linear(i, o)
    def forward(self, x, s, d, n):
        out = torch.zeros(n, x.size(1), device=x.device)
        out.index_add_(0, d, x[s])
        cnt = torch.zeros(n, 1, device=x.device)
        cnt.index_add_(0, d, torch.ones(len(s), 1, device=x.device))
        return F.relu(self.W(out / cnt.clamp(min=1) + x))

class GCNModel(nn.Module):
    def __init__(self, ind, h=128, nl=3, drop=0.2):
        super().__init__()
        self.proj  = nn.Sequential(nn.Linear(ind, h), nn.ReLU())
        self.convs = nn.ModuleList([GCNLayer(h, h) for _ in range(nl)])
        self.drop  = nn.Dropout(drop)
        self.t1 = LinkHead(h, drop); self.t2 = IntentHead(h, drop=drop)
        self.t3 = nn.Sequential(nn.Dropout(drop), nn.Linear(h, N_DOMAIN))
        self.t4 = RankHead(h, drop=drop)
    def encode(self, feat, graph):
        x = self.drop(self.proj(feat["paper"].to(next(self.parameters()).device)))
        s, d, _ = graph["r1"]
        for c in self.convs: x = c(x, s, d, x.size(0))
        return x
    def fwd_t1(self, h, s, d): return self.t1(h, s, d)
    def fwd_t2(self, h, ci, cdi): return self.t2(h, ci, cdi)
    def fwd_t3(self, h): return self.t3(h)
    def fwd_t4(self, h): return self.t4(h)
    compute_loss = _compute_loss_shared


# ── GATv2 ─────────────────────────────────────────────────────────────────────
class GATv2Layer(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.W = nn.Linear(i * 2, o); self.a = nn.Parameter(torch.randn(o) * 0.1)
        self.W2 = nn.Linear(o, o)
    def forward(self, x, s, d, n):
        if s.numel() == 0: return x
        h  = F.relu(self.W(torch.cat([x[s], x[d]], -1)))
        e  = torch.tanh((h * self.a).sum(-1))
        ee = torch.exp(e - e.max())
        den = torch.zeros(n, device=x.device); den.index_add_(0, d, ee)
        al  = ee / (den[d] + 1e-8); msg = h * al.unsqueeze(1)
        out = torch.zeros(n, h.size(1), device=x.device); out.index_add_(0, d, msg)
        return F.relu(self.W2(out + (x if x.size(1) == out.size(1) else torch.zeros_like(out))))

class GATv2Model(nn.Module):
    def __init__(self, ind, h=128, nl=3, drop=0.2):
        super().__init__()
        self.proj  = nn.Sequential(nn.Linear(ind, h), nn.ReLU())
        self.convs = nn.ModuleList([GATv2Layer(h, h) for _ in range(nl)])
        self.drop  = nn.Dropout(drop)
        self.t1 = LinkHead(h, drop); self.t2 = IntentHead(h, drop=drop)
        self.t3 = nn.Sequential(nn.Dropout(drop), nn.Linear(h, N_DOMAIN))
        self.t4 = RankHead(h, drop=drop)
    def encode(self, feat, graph):
        x = self.drop(self.proj(feat["paper"].to(next(self.parameters()).device)))
        s, d, _ = graph["r1"]
        for c in self.convs: x = c(x, s, d, x.size(0))
        return x
    def fwd_t1(self, h, s, d): return self.t1(h, s, d)
    def fwd_t2(self, h, ci, cdi): return self.t2(h, ci, cdi)
    def fwd_t3(self, h): return self.t3(h)
    def fwd_t4(self, h): return self.t4(h)
    compute_loss = _compute_loss_shared


# ── TGN (lightweight) ─────────────────────────────────────────────────────────
class TGNModel(nn.Module):
    def __init__(self, ind, h=128, md=64, drop=0.2):
        super().__init__()
        self.proj = nn.Linear(ind, h); self.gru = nn.GRUCell(h + md, md)
        self.agg  = nn.Linear(h + md, h); self.drop = nn.Dropout(drop); self.md = md
        self.t1 = LinkHead(h, drop); self.t2 = IntentHead(h, drop=drop)
        self.t3 = nn.Sequential(nn.Dropout(drop), nn.Linear(h, N_DOMAIN))
        self.t4 = RankHead(h, drop=drop)
    def encode(self, feat, graph):
        dev = next(self.parameters()).device
        x   = F.relu(self.proj(feat["paper"].to(dev)))
        N   = x.size(0); mem = torch.zeros(N, self.md, device=dev)
        s, d, _ = graph["r1"]
        if s.numel() > 1:
            ef  = (x[s] + x[d]) / 2; inp = torch.cat([ef, mem[s]], -1)
            mn  = self.gru(inp, mem[s]); mem = mem.index_put((s,), mn)
        return F.relu(self.agg(torch.cat([x, mem], -1)))
    def fwd_t1(self, h, s, d): return self.t1(h, s, d)
    def fwd_t2(self, h, ci, cdi):
        lo = self.t2(h, ci, cdi)
        return lo if lo.size(0) == len(ci) else torch.zeros(len(ci), 4, device=h.device)
    def fwd_t3(self, h): return self.t3(h)
    def fwd_t4(self, h): return self.t4(h)
    compute_loss = _compute_loss_shared
