"""models/heads.py — T1–T4 decoder heads + full CiteCraftModel."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.halo import HALOHead, DLOG
from configs.config import N_DOMAIN


# ── T1: Citation Link Prediction ──────────────────────────────────────────────
class LinkHead(nn.Module):
    def __init__(self, h: int, drop: float = 0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(h * 3, h), nn.ReLU(),
            nn.Dropout(drop), nn.Linear(h, 1))

    def forward(self, h, s, d):
        return self.mlp(torch.cat([h[s], h[d], h[s] * h[d]], -1)).squeeze(-1)

    def loss(self, lo, lb):
        return F.binary_cross_entropy_with_logits(lo, lb.float())


# ── T2: Citation Intent Classification ───────────────────────────────────────
class IntentHead(nn.Module):
    def __init__(self, h: int, nc: int = 4, drop: float = 0.2, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        self.attn  = nn.Linear(h * 2, h)
        self.cls   = nn.Linear(h, nc)
        self.drop  = nn.Dropout(drop)

    def forward(self, h, ci, cdi):
        z = torch.tanh(self.attn(torch.cat([h[ci], h[cdi]], -1)))
        return self.cls(self.drop(z))

    def loss(self, lo, lb):
        ce = F.cross_entropy(lo, lb, reduction="none")
        return (((1 - torch.exp(-ce)) ** self.gamma) * ce).mean()


# ── T4: Influence Ranking ─────────────────────────────────────────────────────
class RankHead(nn.Module):
    def __init__(self, h: int, nb: int = 3, drop: float = 0.2):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(h, h // 2), nn.ReLU(),
            nn.Dropout(drop), nn.Linear(h // 2, 1))
        self.cls = nn.Linear(h, nb)

    def forward(self, h):
        return self.scorer(h).squeeze(-1), self.cls(h)

    def loss(self, sc, lo, yb):
        Py = F.softmax(yb.float(), dim=0)
        Ps = F.log_softmax(sc, dim=0)
        return 0.6 * F.cross_entropy(lo, yb) + 0.4 * (-(Py * Ps).sum())


# ── Full CiteCraft Model ──────────────────────────────────────────────────────
class CiteCraftModel(nn.Module):
    def __init__(self, encoder, hidden: int = 128,
                 use_halo: bool = True, drop: float = 0.2):
        super().__init__()
        self.encoder   = encoder
        self.use_halo  = use_halo
        self.t1 = LinkHead(hidden, drop)
        self.t2 = IntentHead(hidden, drop=drop)
        self.t3 = HALOHead(hidden, dropout=drop) if use_halo else \
                  nn.Sequential(nn.Dropout(drop), nn.Linear(hidden, N_DOMAIN))
        self.t4 = RankHead(hidden, drop=drop)

    def encode(self, feat, graph):
        return self.encoder(feat, graph)

    def fwd_t1(self, h, s, d):       return self.t1(h, s, d)
    def fwd_t2(self, h, ci, cdi):    return self.t2(h, ci, cdi)
    def fwd_t3(self, h):             return self.t3(h)
    def fwd_t4(self, h):             return self.t4(h)

    def compute_loss(self, h, graph, t1b, t2b, device):
        ti  = graph["train_mask"].nonzero(as_tuple=True)[0]
        ls  = {}
        s, d, lb = t1b
        ls["T1"] = self.t1.loss(self.fwd_t1(h, s, d), lb)
        if t2b and len(t2b[0]) > 0:
            ci, cdi, il = t2b
            ls["T2"] = self.t2.loss(self.fwd_t2(h, ci, cdi), il)
        else:
            ls["T2"] = torch.tensor(0., device=device)
        lt3 = self.fwd_t3(h[ti]); yt3 = graph["y_domain"][ti]
        ls["T3"] = self.t3.loss(lt3, yt3, device) if self.use_halo else \
                   F.cross_entropy(lt3, yt3, label_smoothing=0.1)
        sc, lc = self.fwd_t4(h[ti]); yt4 = graph["y_bucket"][ti]
        ls["T4"] = self.t4.loss(sc, lc, yt4)
        tot = (1.0*ls["T1"] + 1.5*ls["T2"] +
               1.0*ls["T3"] + 1.2*ls["T4"])
        return tot, ls
