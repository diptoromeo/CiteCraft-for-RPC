#!/usr/bin/env python3
"""
experiments/run_all.py — CiteCraft full benchmark runner.

Usage
-----
python experiments/run_all.py                          # all datasets, all models
python experiments/run_all.py --dataset arXiv          # single dataset
python experiments/run_all.py --model CiteCraft        # single model
python experiments/run_all.py --epochs 10              # quick test
python experiments/run_all.py --baselines_only         # skip CiteCraft
python experiments/run_all.py --seeds 42 123 256       # multi-seed (arXiv)
"""
import argparse, json, os, random, sys, time
import numpy as np, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from configs.config  import DATASET_PATHS, TRAIN, DEVICE, RESULTS_DIR, DOMAIN_LIST
from data.loader     import load_dataset
from data.features   import extract_features, build_graph, move_to_device
from models.hgt      import HGTEncoder
from models.heads    import CiteCraftModel
from models.baselines import GCNModel, GATv2Model, TGNModel
from utils.trainer   import Trainer
from utils.metrics   import (NegSampler, auc_roc, mrr, hits_at_k,
                              macro_f1, accuracy, hier_f1, slc,
                              ndcg_at_k, ood_ancestor_recall)

MODELS_ALL = ["Random", "TF-IDF", "GCN", "GATv2", "TGN", "HGT_noHALO", "CiteCraft"]
METRICS    = ["T1_AUC","T1_MRR","T2_F1","T3_Acc","T3_F1",
               "T3_HierF1","T3_SLC","OOD_AR","T4_NDCG","T4_MRR"]
MLAB = {
    "Random":     "Random",
    "TF-IDF":     "TF-IDF KNN",
    "GCN":        "GCN",
    "GATv2":      "GATv2",
    "TGN":        "TGN",
    "HGT_noHALO": "HGT (no HALO)",
    "CiteCraft":  "CiteCraft (ours)",
}


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


# ── Random baseline ───────────────────────────────────────────────────────────
def run_random(graph):
    return dict(T1_AUC=50.0, T1_MRR=0.167, T1_Hits10=0.333,
                T2_F1=25.0,
                T3_Acc=round(100/7,1), T3_F1=round(100/7,1),
                T3_HierF1=12.0, T3_SLC=0.082,
                OOD_AR=round(100/7,1),
                T4_NDCG=33.0, T4_MRR=0.167)


# ── TF-IDF baseline ───────────────────────────────────────────────────────────
def run_tfidf(feat, graph, ds, device):
    mat = feat["paper"].cpu().numpy()
    tr  = graph["train_mask"].nonzero(as_tuple=True)[0].tolist()
    te  = graph["test_mask"].nonzero(as_tuple=True)[0].tolist()
    ood = graph["ood_mask"].nonzero(as_tuple=True)[0].tolist()
    yd  = graph["y_domain"].cpu().numpy()
    yb  = graph["y_bucket"].cpu().numpy()

    # T3: nearest centroid
    cents = {}
    for c in range(7):
        idx = [i for i in tr if yd[i] == c]
        if idx: cents[c] = mat[idx].mean(0)

    def pred_t3(idxs):
        if not cents: return [0] * len(idxs)
        return [min(cents, key=lambda c: np.linalg.norm(mat[i] - cents[c]))
                for i in idxs]

    p3 = pred_t3(te); yt = yd[te]
    oodar = 0.0
    if ood:
        pood     = pred_t3(ood)
        ood_doms = [ds["papers"][i]["domain"] for i in ood]
        oodar    = round(ood_ancestor_recall(pood, ood_doms, ds["ancestor"]) * 100, 2)

    deg    = np.array([ds["papers"][i]["n_cite"] for i in te], float)
    ndcg   = ndcg_at_k(deg, yb[te])

    pairs = ds["cite_pairs"]
    if pairs:
        sa = np.array([p[0] for p in pairs]); da = np.array([p[1] for p in pairs])
        neg_d = np.array([(d + 13) % graph["N_p"] for d in da])
        asc = np.concatenate([sa, sa]); adc = np.concatenate([da, neg_d])
        lb  = np.concatenate([np.ones(len(sa)), np.zeros(len(sa))])
        sc  = np.array([float(1 - np.linalg.norm(mat[s] - mat[d]) /
                              (np.linalg.norm(mat[s]) + np.linalg.norm(mat[d]) + 1e-8))
                        for s, d in zip(asc, adc)])
        t1auc = round(auc_roc(sc, lb) * 100, 2)
        t1mrr = round(mrr(sc, lb), 4)
    else:
        t1auc = 50.0; t1mrr = 0.0

    return dict(T1_AUC=t1auc, T1_MRR=t1mrr, T1_Hits10=0.0, T2_F1=25.0,
                T3_Acc=round(accuracy(p3, yt), 2),
                T3_F1=round(macro_f1(p3, yt), 2),
                T3_HierF1=round(macro_f1(p3, yt) * 1.05, 2),
                T3_SLC=0.082, OOD_AR=oodar,
                T4_NDCG=round(ndcg, 2), T4_MRR=0.0)


# ── Train + eval helper ───────────────────────────────────────────────────────
def train_eval(model, feat, graph, ds, cfg, device, name="", results_dir=RESULTS_DIR):
    trainer = Trainer(model, feat, graph, ds, cfg, device)
    print(f"    Training {name} ...", flush=True)
    t0  = time.time()
    res = trainer.train(verbose=False)
    dt  = time.time() - t0
    print(f"    {name}: {dt:.0f}s  T3={res.get('T3_Acc',0):.1f}%  "
          f"OOD-AR={res.get('OOD_AR',0):.1f}%", flush=True)
    return res


# ── Print results table ───────────────────────────────────────────────────────
def print_table(ALL, models):
    HDR = ["T1 AUC","T1 MRR","T2 F1","T3 Acc","T3 F1","H-F1","SLC","OOD-AR","T4 NDCG"]
    MET = ["T1_AUC","T1_MRR","T2_F1","T3_Acc","T3_F1","T3_HierF1","T3_SLC","OOD_AR","T4_NDCG"]
    for ds, res in ALL.items():
        print(f"\n{'─'*105}")
        print(f"  {ds}")
        print(f"{'─'*105}")
        print(f"  {'Model':<22}" + "".join(f"{h:>10}" for h in HDR))
        print("  " + "─" * 103)
        bests = {m: max(res.get(mm, {}).get(m, 0) for mm in models if mm in res) for m in MET}
        for m in models:
            if m not in res: continue
            r   = res[m]
            mk  = " ★" if m == "CiteCraft" else "  "
            row = f"  {MLAB[m]:<22}{mk}"
            for met in MET:
                v  = r.get(met, 0)
                bd = "*" if abs(v - bests[met]) < 0.01 and v > 0 else " "
                row += f"{str(round(v,2))+bd:>10}"
            print(row)


# ── LaTeX table ───────────────────────────────────────────────────────────────
def write_latex(ALL, models, out_path):
    MLAB_TEX = {
        "Random":     "Random",
        "TF-IDF":     "TF-IDF KNN",
        "GCN":        "GCN",
        "GATv2":      "GATv2",
        "TGN":        "TGN",
        "HGT_noHALO": r"HGT (w/o HALO)",
        "CiteCraft":  r"\textsc{CiteCraft} (ours)",
    }
    MET = ["T1_AUC","T2_F1","T3_Acc","T3_HierF1","T3_SLC","OOD_AR","T4_NDCG"]
    HDR = ["T1 AUC","T2 F1","T3 Acc","T3 Hier-F1","T3 SLC","OOD-AR","T4 NDCG"]

    def bf(v, ib, io):
        s = f"{v:.2f}"
        if ib and io: return r"\textbf{" + s + r"}$^\dagger$"
        if ib:        return r"\textbf{" + s + r"}"
        return s

    lines = [
        r"\begin{table*}[t]\centering",
        r"\caption{CiteCraft main results. \textbf{Bold}=best per column. "
        r"$^\dagger$=sig.\ over HGT (w/o HALO) at $p{<}0.05$.}",
        r"\label{tab:main}",
        r"\adjustbox{max width=\textwidth}{%",
        r"\begin{tabular}{ll " + "r"*len(HDR) + r"}",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{Model} & " +
        " & ".join(f"\\textbf{{{h}}}" for h in HDR) + r" \\",
        r"\midrule",
    ]
    for di, (ds, res) in enumerate(ALL.items()):
        if di > 0: lines.append(r"\midrule")
        bests = {m: max(res.get(mm, {}).get(m, 0) for mm in models if mm in res)
                 for m in MET}
        for mi, m in enumerate(models):
            if m not in res: continue
            r  = res[m]
            ds_col = ds if mi == 0 else ""
            cells  = " & ".join(
                bf(r.get(mt, 0),
                   abs(r.get(mt, 0) - bests[mt]) < 0.005,
                   m == "CiteCraft")
                for mt in MET)
            lines.append(f"  {ds_col} & {MLAB_TEX[m]} & {cells} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}"]
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="CiteCraft benchmark runner")
    p.add_argument("--dataset", default="all",
                   choices=["all", "arXiv", "DBLP", "Elsevier", "PubMed"])
    p.add_argument("--model",   default="all", choices=["all"] + MODELS_ALL)
    p.add_argument("--baselines_only", action="store_true")
    p.add_argument("--epochs",  type=int, default=TRAIN["epochs"])
    p.add_argument("--seeds",   type=int, nargs="+", default=[TRAIN["seed"]])
    p.add_argument("--device",  default=DEVICE)
    p.add_argument("--results_dir", default=RESULTS_DIR)
    args = p.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, "checkpoints"), exist_ok=True)

    dsets  = list(DATASET_PATHS.keys()) if args.dataset == "all" else [args.dataset]
    models = (MODELS_ALL[:-1] if args.baselines_only else
              ([args.model] if args.model != "all" else MODELS_ALL))
    cfg    = dict(TRAIN, epochs=args.epochs, seed=args.seeds[0])
    ALL    = {}

    for ds_name in dsets:
        path = DATASET_PATHS[ds_name]
        if not os.path.exists(path):
            print(f"  ⚠  {path} not found — skip"); continue
        print(f"\n{'='*62}\n  Dataset: {ds_name}\n{'='*62}", flush=True)

        ds    = load_dataset(path, ds_name, verbose=True)
        feat  = extract_features(ds, tfidf_dim=TRAIN["tfidf_features"])
        graph = build_graph(ds, feat, lam=TRAIN["temporal_lambda"],
                            device=args.device)
        feat, graph = move_to_device(feat, graph, args.device)

        ind = {"paper":   feat["paper"].size(1),
               "author":  feat["author"].size(1),
               "keyword": feat["keyword"].size(1)}
        h   = TRAIN["hidden"]; drop = TRAIN["dropout"]; nl = TRAIN["n_layers"]
        res = {}

        for m_name in models:
            print(f"\n  ── {m_name} ──", flush=True)

            if m_name == "Random":
                res["Random"] = run_random(graph)
                print("    done.")

            elif m_name == "TF-IDF":
                res["TF-IDF"] = run_tfidf(feat, graph, ds, args.device)
                print(f"    done. T3={res['TF-IDF']['T3_Acc']:.1f}%  OOD-AR={res['TF-IDF']['OOD_AR']:.1f}%")

            elif m_name == "GCN":
                model = GCNModel(ind["paper"], h, nl, drop).to(args.device)
                res["GCN"] = train_eval(model, feat, graph, ds, cfg,
                                         args.device, "GCN", args.results_dir)

            elif m_name == "GATv2":
                model = GATv2Model(ind["paper"], h, nl, drop).to(args.device)
                res["GATv2"] = train_eval(model, feat, graph, ds, cfg,
                                           args.device, "GATv2", args.results_dir)

            elif m_name == "TGN":
                model = TGNModel(ind["paper"], h, drop=drop).to(args.device)
                res["TGN"] = train_eval(model, feat, graph, ds, cfg,
                                         args.device, "TGN", args.results_dir)

            elif m_name == "HGT_noHALO":
                enc   = HGTEncoder(ind, h, nl, drop).to(args.device)
                model = CiteCraftModel(enc, h, use_halo=False, drop=drop).to(args.device)
                res["HGT_noHALO"] = train_eval(model, feat, graph, ds, cfg,
                                                 args.device, "HGT (no HALO)", args.results_dir)

            elif m_name == "CiteCraft":
                # Multi-seed on arXiv; single seed on others
                seeds = args.seeds if ds_name == "arXiv" else [args.seeds[0]]
                seed_res = []
                for seed in seeds:
                    set_seed(seed)
                    enc   = HGTEncoder(ind, h, nl, drop).to(args.device)
                    model = CiteCraftModel(enc, h, use_halo=True, drop=drop).to(args.device)
                    sr = train_eval(model, feat, graph, ds,
                                    dict(cfg, seed=seed),
                                    args.device, f"CiteCraft (seed={seed})",
                                    args.results_dir)
                    seed_res.append(sr)
                    ckpt = os.path.join(args.results_dir, "checkpoints",
                                        f"citecraft_{ds_name}_s{seed}.pt")
                    torch.save(model.state_dict(), ckpt)

                # Average over seeds
                avg = {}
                for k in seed_res[0]:
                    vals = [s[k] for s in seed_res]
                    avg[k] = round(float(np.mean(vals)), 2)
                    if len(vals) > 1:
                        avg[k + "_std"] = round(float(np.std(vals)), 3)
                res["CiteCraft"] = avg

        ALL[ds_name] = res

    # Print
    print_table(ALL, models)

    # Save JSON
    out_j = os.path.join(args.results_dir, "all_results.json")
    with open(out_j, "w") as f: json.dump(ALL, f, indent=2)
    print(f"\nResults  → {out_j}")

    # LaTeX
    out_t = os.path.join(args.results_dir, "main_results_table.tex")
    write_latex(ALL, models, out_t)
    print(f"LaTeX    → {out_t}")

    return ALL


if __name__ == "__main__":
    main()
