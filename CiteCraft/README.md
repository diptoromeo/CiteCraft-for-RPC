# CiteCraft 🔬

**Heterogeneous Temporal Citation Graph Benchmark for Multi-Task Scientific NLP**

> EMNLP 2025 — Track: Information Extraction & Text Mining

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Overview

CiteCraft is the first benchmark that unifies **four interrelated citation analysis tasks** on a
single **Heterogeneous Temporal Citation Graph (HTCG)**:

| Task | Description | Metric |
|------|-------------|--------|
| **T1** | Citation Link Prediction | AUC-ROC · MRR · Hits@K |
| **T2** | Citation Intent Classification | Macro-F1 |
| **T3** | Paper Topic Classification | Accuracy · Hier-F1 · SLC · OOD-AR |
| **T4** | Influence Paper Ranking | NDCG@10 · MRR |

### Key Contributions

- **GRACE-HAE** — Graph-Aware Contextual Encoding with Hierarchical Attention Ensemble
  - HAE-Token: TF-IDF boosted attention pooling
  - HAE-Span: Noun-phrase sliding-window max-pool
  - HAE-Graph: Citation-context contrastive noise augmentation
- **HALO** — Hierarchical Adaptive Label Ontology (feeds all 4 tasks)
  - DLOG: Corpus-derived dynamic label ontology (zero human annotation)
  - SHLA: Soft hierarchical label assignment (β=0.5 ancestor decay)
  - OAHL: Ontology-aware Wu-Palmer distance loss
  - PALE: Prototype-anchored label embeddings
- **3 new metrics** — Hier-F1, Soft Label Calibration (SLC), OOD Ancestor Recall (OOD-AR)
- **OOD-AR** — `rl` domain withheld from training; model must assign ancestor `cs` at test time

---

## Results (arXiv, 5 seeds)

| Model | T1 AUC | T2 F1 | T3 Acc | T3 Hier-F1 | T3 SLC | OOD-AR | T4 NDCG |
|-------|--------|-------|--------|------------|--------|--------|---------|
| Random | 50.0 | 25.0 | 14.3 | 12.0 | 0.082 | 14.3 | 33.0 |
| GCN | 77.8 | 46.1 | 58.4 | 51.3 | 0.321 | 31.4 | 67.9 |
| GATv2 | 81.3 | 51.7 | 63.2 | 55.8 | 0.349 | 35.6 | 71.4 |
| TGN | 83.6 | 49.2 | 61.1 | 52.7 | 0.298 | 33.2 | 74.8 |
| HGT (no HALO) | 86.4 | 57.9 | 68.1 | 59.4 | 0.412 | 38.7 | 77.2 |
| **CiteCraft (ours)** | **89.7** | **63.4** | **73.8** | **67.4** | **0.851** | **61.8** | **81.5** |

---

## Project Structure

```
CiteCraft/
├── README.md
├── requirements.txt
├── LICENSE
├── datasets/                          ← place the 4 JSON files here
│   ├── arXiv_citation_dataset.json
│   ├── DBLP_citation_dataset.json
│   ├── Elsevier_citation_dataset.json
│   └── PubMed_citation_dataset.json
├── configs/
│   └── config.py                      ← all hyperparameters & OOD config
├── data/
│   ├── loader.py                      ← JSON → Paper objects + OOD split
│   └── features.py                    ← TF-IDF features + HTCG construction
├── models/
│   ├── hgt.py                         ← HGT encoder (pure PyTorch)
│   ├── halo.py                        ← DLOG · SHLA · OAHL · PALE
│   ├── heads.py                       ← T1–T4 decoder heads + CiteCraftModel
│   └── baselines.py                   ← GCN · GATv2 · TGN · TF-IDF
├── utils/
│   ├── metrics.py                     ← AUC · MRR · Hier-F1 · SLC · OOD-AR
│   └── trainer.py                     ← training loop + early stopping
├── experiments/
│   ├── run_all.py                     ← full benchmark runner (CLI)
│   └── make_figures.py                ← publication figures
└── results/
    ├── figures/
    └── checkpoints/
```

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/your-org/CiteCraft.git
cd CiteCraft
pip install -r requirements.txt
```

### 2. Datasets

Place the four JSON files inside `datasets/`:

```
datasets/
├── arXiv_citation_dataset.json
├── DBLP_citation_dataset.json
├── Elsevier_citation_dataset.json
└── PubMed_citation_dataset.json
```

Or download and extract the provided `citation_datasets.zip`:

```bash
unzip citation_datasets.zip -d datasets/
```

### 3. Run

```bash
# All datasets, all models (recommended)
python experiments/run_all.py

# Single dataset
python experiments/run_all.py --dataset arXiv

# Single model
python experiments/run_all.py --dataset arXiv --model CiteCraft

# Quick smoke-test
python experiments/run_all.py --dataset arXiv --epochs 5

# Baselines only
python experiments/run_all.py --baselines_only

# Generate publication figures
python experiments/make_figures.py
```

### 4. Outputs

```
results/
├── all_results.json          # all numeric scores
├── main_results_table.tex    # LaTeX Table 1 (camera-ready)
├── figures/
│   ├── fig1_main_results.png
│   ├── fig2_ood_ar.png
│   ├── fig3_hier_f1_scatter.png
│   └── fig4_complexity.png
└── checkpoints/
    ├── citecraft_arXiv.pt
    ├── citecraft_DBLP.pt
    ├── citecraft_Elsevier.pt
    └── citecraft_PubMed.pt
```

---

## Configuration

Edit `configs/config.py`:

```python
TRAIN = dict(
    epochs   = 80,       # training epochs
    lr       = 2e-3,     # AdamW learning rate
    hidden   = 128,      # GNN hidden dimension
    n_layers = 3,        # HGT layers
    dropout  = 0.2,      # dropout probability
    patience = 15,       # early stopping patience
    seed     = 42,
)

# OOD-AR: domain withheld from training per dataset
OOD_HOLDOUT = {
    "arXiv":    ["rl"],
    "DBLP":     ["rl"],
    "Elsevier": ["rl"],
    "PubMed":   ["rl"],
}
```

---

## Citation

```bibtex
@inproceedings{citecraft2025,
  title     = {CiteCraft: A Heterogeneous Temporal Citation Graph Benchmark
               for Multi-Task Scientific NLP},
  booktitle = {Proceedings of EMNLP 2025},
  year      = {2025}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
