"""configs/config.py — CiteCraft master configuration."""
import os, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Dataset paths ─────────────────────────────────────────────────────────────
DATASETS_DIR = os.path.join(ROOT, "datasets")
DATASET_PATHS = {
    "arXiv":    os.path.join(DATASETS_DIR, "arXiv_citation_dataset.json"),
    "DBLP":     os.path.join(DATASETS_DIR, "DBLP_citation_dataset.json"),
    "Elsevier": os.path.join(DATASETS_DIR, "Elsevier_citation_dataset.json"),
    "PubMed":   os.path.join(DATASETS_DIR, "PubMed_citation_dataset.json"),
}

# ── Domain taxonomy ────────────────────────────────────────────────────────────
DOMAIN_LIST = ["graph", "rl", "nlp", "math", "physics", "bio", "cs"]
D2I  = {d: i for i, d in enumerate(DOMAIN_LIST)}
N_DOMAIN = 7

DOMAIN_MAP = {
    "graph":   ["graph","vertex","edge","coloring","clique","tree","matching",
                "bipartite","chromatic","planar","cycle","path","connectivity"],
    "rl":      ["reinforcement","reward","policy","agent","markov","q-learning",
                "actor","critic","bandit","exploration","exploitation"],
    "nlp":     ["language","bert","transformer","text","embedding","semantic",
                "nlp","attention","llm","token","sentence","corpus","vocabulary"],
    "math":    ["theorem","proof","polynomial","algebraic","topology","combinat",
                "manifold","group","ring","field","lattice","matrix","eigenvalue"],
    "physics": ["quantum","particle","photon","hamiltonian","entanglement",
                "condensed","spin","boson","fermion","scattering"],
    "bio":     ["protein","gene","genomic","clinical","disease","drug","patient",
                "cell","cancer","dna","rna","mutation","phenotype","biomarker"],
    "cs":      ["algorithm","network","system","database","distributed",
                "compiler","cache","software","operating","scheduler","memory"],
}

# ── DLOG ancestor map (for OOD-AR) ────────────────────────────────────────────
# Defines the parent domain in the DLOG hierarchy
ANCESTOR_MAP = {
    "graph":   "cs",
    "rl":      "cs",
    "nlp":     "cs",
    "math":    "math",
    "physics": "physics",
    "bio":     "bio",
    "cs":      "cs",
}

# ── OOD holdout: withheld from training, used for OOD-AR evaluation ───────────
OOD_HOLDOUT = {
    "arXiv":    ["rl"],   # 28 papers
    "DBLP":     ["rl"],   # 12 papers
    "Elsevier": ["rl"],   # 10 papers
    "PubMed":   ["rl"],   #  3 papers
}

# ── Training hyperparameters ──────────────────────────────────────────────────
TRAIN = dict(
    epochs          = 80,
    lr              = 2e-3,
    weight_decay    = 1e-4,
    hidden          = 128,
    n_layers        = 3,
    n_heads         = 4,
    dropout         = 0.2,
    patience        = 15,
    seed            = 42,
    temporal_lambda = 0.1,   # λ in w = exp(-λ|Δy|) for R1 edges
    tfidf_features  = 512,   # TF-IDF vocabulary size
    neg_ratio       = 3,     # negative:positive ratio for T1
    tau_hyp         = 0.7,   # DLOG edge threshold
    tau_pale        = 0.07,  # PALE cosine temperature
    beta_shla       = 0.5,   # SHLA ancestor decay
    w_hier          = 0.3,   # OAHL hierarchical loss weight
    lambda_reg      = 0.1,   # PALE prototype regularisation
    # Multi-task loss weights
    lambda_t1       = 1.0,
    lambda_t2       = 1.5,
    lambda_t3       = 1.0,
    lambda_t4       = 1.2,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = os.path.join(ROOT, "results")
