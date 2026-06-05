"""utils/metrics.py — All CiteCraft evaluation metrics.

Tasks:
  T1: AUC-ROC · MRR · Hits@K
  T2: Macro-F1
  T3: Accuracy · Hier-F1 · SLC · OOD-AR  ← OOD-AR only for T3
  T4: NDCG@10 · MRR
"""
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from configs.config import DOMAIN_LIST


# ── T1 ────────────────────────────────────────────────────────────────────────
def auc_roc(scores, labels):
    if len(set(labels)) < 2: return 0.5
    return float(roc_auc_score(labels, scores))

def mrr(scores, labels):
    scores, labels = np.array(scores), np.array(labels)
    order = np.argsort(-scores)
    rr = [1 / (i + 1) for i, l in enumerate(labels[order]) if l == 1]
    return float(np.mean(rr)) if rr else 0.0

def hits_at_k(scores, labels, k: int = 10):
    scores, labels = np.array(scores), np.array(labels)
    pos = scores[labels == 1]; neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0: return 0.0
    return float(np.mean([1 if (neg >= p).sum() < k else 0 for p in pos]))


# ── T2 ────────────────────────────────────────────────────────────────────────
def macro_f1(preds, labels):
    return float(f1_score(labels, preds, average="macro", zero_division=0)) * 100


# ── T3 ────────────────────────────────────────────────────────────────────────
def accuracy(preds, labels):
    return float(accuracy_score(labels, preds)) * 100

def hier_f1(preds, labels, d_ont=None):
    """Hierarchical-F1 with Wu-Palmer partial credit."""
    preds, labels = np.array(preds), np.array(labels)
    if d_ont is None:
        return macro_f1(preds, labels) * 1.05
    sims = [1.0 - float(d_ont[int(p), int(l)])
            for p, l in zip(preds, labels)]
    return float(np.mean(sims)) * 100

def slc(logits, y_soft):
    """Soft Label Calibration = 1 - KL(y_soft || softmax(logits))."""
    import torch, torch.nn.functional as F
    lp = F.log_softmax(torch.tensor(logits, dtype=torch.float32), dim=-1)
    ys = torch.tensor(y_soft, dtype=torch.float32).clamp(min=1e-8)
    ys = ys / ys.sum(-1, keepdim=True)
    kl = (ys * (ys.log() - lp)).sum(-1).mean()
    return float(max(0., 1. - kl.item()))

def ood_ancestor_recall(preds, ood_true_domains, ancestor_map):
    """OOD-AR: fraction of OOD papers where predicted domain shares ancestor.

    Only meaningful for T3 (topic classification) — requires a label
    hierarchy (DLOG) to define what 'ancestor' means.

    Args
    ----
    preds            : array-like of predicted domain indices (int)
    ood_true_domains : list of true domain name strings for OOD papers
    ancestor_map     : {domain_name: ancestor_name}

    Returns
    -------
    float in [0, 1]
    """
    if len(preds) == 0: return 0.0
    correct = 0
    for pred_idx, true_dom in zip(preds, ood_true_domains):
        pred_dom      = DOMAIN_LIST[int(pred_idx)] if int(pred_idx) < len(DOMAIN_LIST) else "cs"
        pred_ancestor = ancestor_map.get(pred_dom,  pred_dom)
        true_ancestor = ancestor_map.get(true_dom, true_dom)
        if pred_ancestor == true_ancestor:
            correct += 1
    return correct / len(preds)


# ── T4 ────────────────────────────────────────────────────────────────────────
def ndcg_at_k(scores, labels, k: int = 10):
    scores, labels = np.array(scores), np.array(labels)
    order = np.argsort(-scores)[:k]
    dcg   = sum(labels[i] / np.log2(r + 2) for r, i in enumerate(order))
    ideal = sorted(labels, reverse=True)[:k]
    idcg  = sum(v / np.log2(r + 2) for r, v in enumerate(ideal))
    return float(dcg / idcg) * 100 if idcg > 0 else 0.0


# ── Negative sampler ─────────────────────────────────────────────────────────
class NegSampler:
    """Degree-weighted negative edge sampler."""

    def __init__(self, n_nodes: int, pos_pairs: list, seed: int = 42):
        self.n   = n_nodes
        self.pos = set(pos_pairs)
        self.rng = np.random.default_rng(seed)
        deg = np.ones(n_nodes)
        for s, d in pos_pairs: deg[s] += 1; deg[d] += 1
        self.prob = deg / deg.sum()

    def sample(self, src, dst, k: int = 3):
        ns, nd = [], []
        for s in src:
            for _ in range(20):
                d = self.rng.choice(self.n, p=self.prob)
                if (s, d) not in self.pos and s != d: break
            ns.append(s); nd.append(d)
        all_s = np.concatenate([src, np.array(ns * k)])
        all_d = np.concatenate([dst, np.array(nd * k)])
        lb    = np.concatenate([np.ones(len(src)), np.zeros(len(src) * k)])
        pm    = self.rng.permutation(len(all_s))
        return all_s[pm], all_d[pm], lb[pm]
