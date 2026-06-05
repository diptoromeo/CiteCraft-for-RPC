"""utils/trainer.py — Training loop with OOD-AR evaluation for T3."""
import numpy as np
import torch
import torch.nn.functional as F
from utils.metrics import (NegSampler, auc_roc, mrr, hits_at_k,
                            macro_f1, accuracy, hier_f1, slc,
                            ndcg_at_k, ood_ancestor_recall)
from models.halo import DLOG
from configs.config import DOMAIN_LIST


class Trainer:
    def __init__(self, model, feat, graph, ds, cfg, device="cpu"):
        self.model = model; self.feat = feat; self.graph = graph
        self.ds = ds; self.cfg = cfg; self.device = device

        self.opt   = torch.optim.AdamW(model.parameters(),
                                        lr=cfg["lr"],
                                        weight_decay=cfg["weight_decay"])
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, cfg["epochs"], eta_min=1e-6)
        self.rng   = np.random.default_rng(cfg["seed"])
        self.neg   = NegSampler(graph["N_p"], ds["cite_pairs"], seed=cfg["seed"])

        self.best  = -1.; self.best_state = None; self.best_ep = 0; self._st = 0
        self.t2b   = self._build_t2()

        # Build DLOG for HALO-enabled models
        if hasattr(model, "t3") and hasattr(model.t3, "set_dlog"):
            ti   = graph["train_mask"].nonzero(as_tuple=True)[0]
            dlog = DLOG(tau=cfg.get("tau_hyp", 0.7)).build(graph["y_domain"][ti])
            model.t3.set_dlog(dlog)

    def _build_t2(self):
        pairs = self.ds["cite_pairs"]
        if not pairs: return None
        papers = self.ds["papers"]
        ci, cdi, lb = [], [], []
        for s, d in pairs:
            ci.append(s); cdi.append(d)
            lb.append(1 if papers[s]["domain"] == papers[d]["domain"] else 0)
        return (torch.tensor(ci,  dtype=torch.long,  device=self.device),
                torch.tensor(cdi, dtype=torch.long,  device=self.device),
                torch.tensor(lb,  dtype=torch.long,  device=self.device))

    def _t1_batch(self):
        pairs = self.ds["cite_pairs"]; N = self.graph["N_p"]
        if not pairs:
            src = self.rng.integers(0, N, 20)
            dst = (src + 1 + self.rng.integers(0, N - 1, 20)) % N
            pairs = list(zip(src.tolist(), dst.tolist()))
        sa = np.array([p[0] for p in pairs])
        da = np.array([p[1] for p in pairs])
        as_, ad_, lb = self.neg.sample(sa, da, k=self.cfg.get("neg_ratio", 3))
        return (torch.tensor(as_, dtype=torch.long,    device=self.device),
                torch.tensor(ad_, dtype=torch.long,    device=self.device),
                torch.tensor(lb,  dtype=torch.float32, device=self.device))

    def train_epoch(self) -> dict:
        self.model.train()
        t1b = self._t1_batch()
        h   = self.model.encode(self.feat, self.graph)
        tot, ls = self.model.compute_loss(h, self.graph, t1b, self.t2b, self.device)
        self.opt.zero_grad(); tot.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step(); self.sched.step()
        return {k: float(v) for k, v in ls.items()} | {"total": float(tot)}

    @torch.no_grad()
    def evaluate(self, split: str = "val") -> dict:
        self.model.eval()
        h    = self.model.encode(self.feat, self.graph)
        mask = self.graph[f"{split}_mask"]
        idx  = mask.nonzero(as_tuple=True)[0]
        res  = {}

        # ── T1 ──────────────────────────────────────────────────────────────
        pairs = self.ds["cite_pairs"]
        if pairs:
            sa = np.array([p[0] for p in pairs])
            da = np.array([p[1] for p in pairs])
            as_, ad_, lb = self.neg.sample(sa, da, k=3)
            ts = torch.tensor(as_, dtype=torch.long, device=self.device)
            td = torch.tensor(ad_, dtype=torch.long, device=self.device)
            sc = torch.sigmoid(self.model.fwd_t1(h, ts, td)).cpu().numpy()
            res["T1_AUC"]   = round(auc_roc(sc, lb) * 100, 2)
            res["T1_MRR"]   = round(mrr(sc, lb), 4)
            res["T1_Hits10"] = round(hits_at_k(sc, lb, 10), 4)
        else:
            res["T1_AUC"] = 50.0; res["T1_MRR"] = 0.0; res["T1_Hits10"] = 0.0

        # ── T2 ──────────────────────────────────────────────────────────────
        if self.t2b:
            ci, cdi, il = self.t2b
            lo2 = self.model.fwd_t2(h, ci, cdi)
            if lo2 is not None and lo2.size(-1) > 1:
                res["T2_F1"] = round(macro_f1(lo2.argmax(-1).cpu().numpy(),
                                               il.cpu().numpy()), 2)
            else: res["T2_F1"] = 0.0
        else: res["T2_F1"] = 0.0

        # ── T3 ──────────────────────────────────────────────────────────────
        if len(idx) > 0:
            lo3  = self.model.fwd_t3(h[idx]).cpu().numpy()
            yt   = self.graph["y_domain"][idx].cpu().numpy()
            p3   = lo3.argmax(1)
            res["T3_Acc"]    = round(accuracy(p3, yt), 2)
            res["T3_F1"]     = round(macro_f1(p3, yt), 2)
            dlog = getattr(getattr(self.model, "t3", None), "dlog", None)
            dont = dlog.d_ont.numpy() if (dlog and dlog.d_ont is not None) else None
            res["T3_HierF1"] = round(hier_f1(p3, yt, dont), 2)
            if hasattr(getattr(self.model, "t3", None), "soft_labels"):
                ys = self.model.t3.soft_labels(
                    self.graph["y_domain"][idx], self.device).cpu().numpy()
                res["T3_SLC"] = round(slc(lo3, ys), 4)
            else:
                res["T3_SLC"] = 0.0

        # ── OOD-AR (T3 only) ─────────────────────────────────────────────────
        ood_idx = self.graph["ood_mask"].nonzero(as_tuple=True)[0]
        if len(ood_idx) > 0:
            lo_ood   = self.model.fwd_t3(h[ood_idx]).cpu().numpy()
            pred_ood = lo_ood.argmax(1)
            ood_doms = [self.ds["papers"][i.item()]["domain"] for i in ood_idx]
            res["OOD_AR"] = round(
                ood_ancestor_recall(pred_ood, ood_doms, self.ds["ancestor"]) * 100, 2)
        else:
            res["OOD_AR"] = 0.0

        # ── T4 ──────────────────────────────────────────────────────────────
        if len(idx) > 0:
            sc4, _ = self.model.fwd_t4(h[idx])
            yb     = self.graph["y_bucket"][idx].cpu().numpy()
            res["T4_NDCG"] = round(ndcg_at_k(sc4.cpu().numpy(), yb), 2)
            res["T4_MRR"]  = round(mrr(sc4.cpu().numpy(), (yb > 0).astype(float)), 4)

        return res

    def train(self, verbose: bool = True) -> dict:
        for ep in range(1, self.cfg["epochs"] + 1):
            ls = self.train_epoch()
            if ep % 10 == 0 or ep == self.cfg["epochs"]:
                vr = self.evaluate("val")
                sc = (vr.get("T1_AUC", 50) * 0.3 +
                      vr.get("T3_Acc",  0 ) * 0.7) / 100
                if sc > self.best:
                    self.best = sc; self._st = 0; self.best_ep = ep
                    self.best_state = {k: v.clone()
                                       for k, v in self.model.state_dict().items()}
                else:
                    self._st += 1
                if verbose:
                    print(f"    ep={ep:3d} loss={ls['total']:.4f} "
                          f"T1={vr.get('T1_AUC',50):.1f} "
                          f"T3={vr.get('T3_Acc',0):.1f}% "
                          f"OOD={vr.get('OOD_AR',0):.1f}%")
                if self._st >= self.cfg["patience"]:
                    if verbose: print(f"    Early stop @ ep={ep}")
                    break
        if self.best_state:
            self.model.load_state_dict(self.best_state)
        return self.evaluate("test")
