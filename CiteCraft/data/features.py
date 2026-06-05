"""data/features.py — TF-IDF feature extraction and HTCG graph construction."""
import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from configs.config import D2I, N_DOMAIN


def extract_features(ds: dict, tfidf_dim: int = 512) -> dict:
    """Extract node features for all three node types.

    Paper nodes  : TF-IDF 512-dim sparse matrix → dense tensor
    Author nodes : Gaussian-initialised 64-dim embeddings
    Keyword nodes: Gaussian-initialised 32-dim embeddings
    """
    papers = ds["papers"]

    # Use all available text (title + abstract) for richer TF-IDF
    texts = [f"{p['title']} {p['abstract']}" for p in papers]
    vec   = TfidfVectorizer(
        max_features=tfidf_dim, min_df=1,
        stop_words="english", sublinear_tf=True,
        ngram_range=(1, 2))   # include bigrams for multi-word terms
    mat   = torch.tensor(vec.fit_transform(texts).toarray(),
                         dtype=torch.float32)

    A = max(len(ds["auth2id"]), 1)
    auth_feat = torch.empty(A, 64);  nn.init.normal_(auth_feat, std=0.1)
    kw_feat   = torch.empty(200, 32); nn.init.normal_(kw_feat,   std=0.1)

    return {"paper": mat, "author": auth_feat, "keyword": kw_feat}


def build_graph(ds: dict, feat: dict, lam: float = 0.1,
                device: str = "cpu") -> dict:
    """Assemble the HTCG tensor representation.

    Edge types
    ----------
    R1 cites  (P→P) : w = exp(-λ|y_i - y_j|)   temporal decay
    R2 wrote  (A→P) : w = 1                      binary authorship
    R3 tagged (P→K) : w = tf·idf                 top-5 per paper
    """
    papers  = ds["papers"]
    auth2id = ds["auth2id"]
    N_p = len(papers)
    N_a = max(len(auth2id), 1)
    N_k = 200

    # R1 ── cites (Paper → Paper)
    r1s, r1d, r1w = [], [], []
    for (s, d) in ds["cite_pairs"]:
        ys = papers[s]["year"] or 2022
        yd = papers[d]["year"] or 2022
        r1s.append(s); r1d.append(d)
        r1w.append(float(np.exp(-lam * abs(ys - yd))))

    # R2 ── wrote (Author → Paper)
    r2s, r2d = [], []
    for p in papers:
        for a in p["authors"]:
            if a in auth2id:
                r2s.append(auth2id[a]); r2d.append(p["pid"])

    # R3 ── tagged (Paper → Keyword) — top-5 TF-IDF per paper
    r3s, r3d, r3w = [], [], []
    tm = feat["paper"].numpy()
    for pi in range(N_p):
        row = tm[pi]
        for ki in np.argsort(row)[-5:]:
            if row[ki] > 0.01 and ki < N_k:
                r3s.append(pi); r3d.append(ki); r3w.append(float(row[ki]))

    def mk(src, dst, w=None):
        if not src:
            return (torch.zeros(1, dtype=torch.long,    device=device),
                    torch.zeros(1, dtype=torch.long,    device=device),
                    torch.ones(1,  dtype=torch.float32, device=device))
        return (torch.tensor(src, dtype=torch.long,    device=device),
                torch.tensor(dst, dtype=torch.long,    device=device),
                torch.tensor(w or [1.0]*len(src), dtype=torch.float32, device=device))

    n  = N_p
    tr = set(ds["train_idx"]); va = set(ds["val_idx"])
    te = set(ds["test_idx"]);  ood = set(ds["ood_idx"])

    return dict(
        N_p=N_p, N_a=N_a, N_k=N_k,
        r1=mk(r1s, r1d, r1w),
        r2=mk(r2s, r2d),
        r3=mk(r3s, r3d, r3w),
        y_domain  = torch.tensor([D2I.get(p["domain"], 6) for p in papers],
                                   dtype=torch.long, device=device),
        y_bucket  = torch.tensor([p["bucket"] for p in papers],
                                   dtype=torch.long, device=device),
        train_mask = torch.tensor([i in tr  for i in range(n)],
                                   dtype=torch.bool, device=device),
        val_mask   = torch.tensor([i in va  for i in range(n)],
                                   dtype=torch.bool, device=device),
        test_mask  = torch.tensor([i in te  for i in range(n)],
                                   dtype=torch.bool, device=device),
        ood_mask   = torch.tensor([i in ood for i in range(n)],
                                   dtype=torch.bool, device=device),
    )


def move_to_device(feat: dict, graph: dict, device: str) -> tuple[dict, dict]:
    """Move feature tensors and graph tensors to the target device."""
    f2 = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
          for k, v in feat.items()}
    g2 = {}
    for k, v in graph.items():
        if isinstance(v, torch.Tensor):
            g2[k] = v.to(device)
        elif isinstance(v, tuple):
            g2[k] = tuple(x.to(device) if isinstance(x, torch.Tensor)
                          else x for x in v)
        else:
            g2[k] = v
    return f2, g2
