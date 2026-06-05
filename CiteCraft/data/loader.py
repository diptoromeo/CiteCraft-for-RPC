"""data/loader.py — Load and parse all four citation JSON datasets.

Handles schema differences:
  arXiv   : has scraped_main_authors, scraped_main_keywords, extra scholar fields
  DBLP    : title/abstract + citing_articles only
  Elsevier: some titles/abstracts are None
  PubMed  : some titles/abstracts are None
"""
import json, re
import numpy as np
from collections import Counter
from configs.config import DOMAIN_MAP, D2I, OOD_HOLDOUT, ANCESTOR_MAP


# ── helpers ───────────────────────────────────────────────────────────────────
def infer_domain(title: str, abstract: str) -> str:
    text = ((title or "") + " " + (abstract or "")[:300]).lower()
    scores = {d: sum(1 for kw in kws if kw in text)
              for d, kws in DOMAIN_MAP.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "cs"


def citation_bucket(n: int) -> int:
    """T4 label: 0 = no cites, 1 = 1-4, 2 = 5+."""
    if n == 0:   return 0
    if n <= 4:   return 1
    return 2


def extract_year(s: str) -> int | None:
    """Extract 4-digit year from author string trailing ', YYYY'."""
    m = re.search(r",\s*(20[0-2]\d|19[9]\d)\s*$", s or "")
    if m: return int(m.group(1))
    m = re.findall(r"\b(20[0-2]\d)\b", s or "")
    return int(m[-1]) if m else None


def parse_authors(raw: str) -> list[str]:
    raw = re.sub(r"-\s+.*$", "", raw or "")
    raw = re.sub(r",\s*20[0-2]\d\s*$", "", raw)
    return [p.strip().rstrip("…")
            for p in re.split(r"[,;]", raw)
            if 2 < len(p.strip()) < 60]


# ── main loader ───────────────────────────────────────────────────────────────
def load_dataset(path: str, name: str, verbose: bool = True) -> dict:
    """Parse a CiteCraft citation JSON file into a structured dataset dict.

    Returns
    -------
    dict with keys:
        papers      : list of paper dicts
        auth2id     : {author_name: int}
        cite_pairs  : list of (citing_pid, cited_pid) intra-corpus pairs
        train_idx   : list of paper indices for training (OOD domain excluded)
        val_idx     : list of paper indices for validation
        test_idx    : list of paper indices for test
        ood_idx     : list of paper indices with held-out domain labels
        ood_domains : set of held-out domain strings
        ancestor    : {domain_name: ancestor_name}  (for OOD-AR)
        name        : dataset name string
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    papers, auth2id = [], {}
    ood_domains = set(OOD_HOLDOUT.get(name, []))

    for idx, entry in enumerate(raw):
        title    = (entry.get("original_csv_title",    "") or "").strip()
        abstract = (entry.get("original_csv_abstract", "") or "").strip()

        # arXiv has extra scholar fields — prefer them if richer
        if name == "arXiv":
            t2 = (entry.get("scraped_main_title_from_scholar", "") or "").strip()
            a2 = (entry.get("scraped_main_abstract_from_scholar", "") or "").strip()
            if len(t2) > len(title): title = t2
            if len(a2) > len(abstract): abstract = a2

        raw_auth  = (entry.get("scraped_main_authors", "") or "").strip()
        keywords  = entry.get("scraped_main_keywords", []) or []
        authors   = parse_authors(raw_auth)
        year      = extract_year(raw_auth)
        domain    = infer_domain(title, abstract)
        citing    = entry.get("citing_articles", []) or []

        for a in authors:
            if a not in auth2id: auth2id[a] = len(auth2id)
        for ca in citing:
            for a in parse_authors(ca.get("authors", "") or ""):
                if a not in auth2id: auth2id[a] = len(auth2id)

        papers.append({
            "pid":       idx,
            "title":     title,
            "abstract":  abstract,
            "authors":   authors,
            "keywords":  keywords,
            "year":      year,
            "domain":    domain,
            "domain_id": D2I.get(domain, 6),
            "bucket":    citation_bucket(len(citing)),
            "n_cite":    len(citing),
            "citing":    citing,
        })

    # ── intra-corpus citation pairs ───────────────────────────────────────────
    def norm(t: str) -> str:
        return re.sub(r"\W+", " ", (t or "").lower()).strip()

    idx_map = {norm(p["title"]): p["pid"]
               for p in papers if p["title"]}
    cite_pairs, seen = [], set()
    for p in papers:
        for ca in p["citing"]:
            k = norm(ca.get("title", "") or "")
            if k in idx_map and idx_map[k] != p["pid"]:
                pair = (p["pid"], idx_map[k])
                if pair not in seen:
                    cite_pairs.append(pair); seen.add(pair)

    # ── temporal split ────────────────────────────────────────────────────────
    known = sorted(set(p["year"] for p in papers if p["year"]))
    if len(known) >= 3:
        vy = known[int(len(known) * 0.80)]
        ty = known[int(len(known) * 0.90)]
        tr = [p["pid"] for p in papers if p["year"] and p["year"] < vy]
        va = [p["pid"] for p in papers if p["year"] and p["year"] == vy]
        te = [p["pid"] for p in papers if not p["year"] or p["year"] >= ty]
    else:
        tr, va, te = [], [], []

    # Fallback: stratified random 70/15/15
    if len(tr) < 10 or len(va) < 3 or len(te) < 3:
        rng  = np.random.default_rng(42)
        perm = rng.permutation(len(papers)).tolist()
        n    = len(perm)
        tr   = perm[:int(n * 0.70)]
        va   = perm[int(n * 0.70):int(n * 0.85)]
        te   = perm[int(n * 0.85):]

    # ── OOD split: remove held-out domains from training ──────────────────────
    train_domains  = set(D2I[d] for d in D2I if d not in ood_domains)
    tr_clean = [i for i in tr if papers[i]["domain_id"] in train_domains]
    ood_idx  = [p["pid"] for p in papers if p["domain"] in ood_domains]

    if verbose:
        dom_cnt = Counter(p["domain"] for p in papers)
        bkt_cnt = Counter(p["bucket"] for p in papers)
        print(f"  [{name}] papers={len(papers)} authors={len(auth2id)} "
              f"cite_pairs={len(cite_pairs)} ood={len(ood_idx)}")
        print(f"  split: train={len(tr_clean)} val={len(va)} test={len(te)}")
        print(f"  domains: {dict(dom_cnt.most_common(4))}")
        print(f"  buckets: 0={bkt_cnt[0]} 1-4={bkt_cnt[1]} 5+={bkt_cnt[2]}")

    return dict(
        papers=papers, auth2id=auth2id, cite_pairs=cite_pairs,
        train_idx=tr_clean, val_idx=va, test_idx=te,
        ood_idx=ood_idx, ood_domains=ood_domains,
        train_domains=train_domains,
        ancestor=ANCESTOR_MAP, name=name,
    )
