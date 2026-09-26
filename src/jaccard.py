"""Step 5 - RQ2 self-consistency analysis (label-free).

Reads the temperature=0.7 K-run extractions written by consistency.py and
measures how stable each model's output is across runs, via mean pairwise
Jaccard overlap of the extracted triple sets. Reported overall and per
relation type. Consistency is a reproducibility signal, NOT an accuracy proxy.
"""

import json
import schema
from collections import defaultdict
from itertools import combinations
from statistics import mean
from collect_corpus import DATASET_DIR       # openai-free -> no import side effects
from pathlib import Path
import csv

CONSIST_DIR = DATASET_DIR / "extractions" / "consistency"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

def jaccard(a, b):
    """Jaccard overlap of two triple-sets: |A and B| / |A or B|."""
    union = a | b
    if len(union) == 0:
        return 1.0                            # both runs emitted nothing -> defined as fully consistent
    return len(a & b) / len(union)

def load_runs(model):
    """Load one model's consistency file into {arxiv_id: {run: set_of_triple_keys}}."""
    by_abstract = defaultdict(dict)
    path = CONSIST_DIR / f"{model}.jsonl"
    for line in open(path, encoding="utf-8"):
        rec = json.loads(line)
        triples = rec["triples"]
        # canonical key = (relation, normalized head, normalized tail);
        # entity types are dropped on purpose -- RQ2 compares content, not typing.
        s = {(t["relation"], t["head"].strip().lower(), t["tail"].strip().lower())
             for t in triples}
        by_abstract[rec["arxiv_id"]][rec["run"]] = s
    return by_abstract

def mean_pairwise(sets):
    """Mean Jaccard over every run-pair in a list of triple-sets (C(K,2) pairs)."""
    return mean([jaccard(a, b) for a, b in combinations(sets, 2)])

def abstract_jaccard(runs):
    """Mean pairwise Jaccard across a single abstract's K runs (C(3,2)=3 pairs for K=3)."""
    sets = list(runs.values())
    return mean_pairwise(sets)

def overall_jaccard(by_abstract):
    """Model-level consistency: mean of per-abstract Jaccard over all abstracts. Returns (mean, n)."""
    vals = [abstract_jaccard(runs) for runs in by_abstract.values()]
    return mean(vals), len(vals)

def relation_jaccard(by_abstract, R):
    """Per-relation consistency for relation R; abstracts where R never appears are skipped. Returns (mean, n)."""
    vals = []
    for runs in by_abstract.values():
        sets_R = [{k for k in s if k[0] == R} for s in runs.values()]
        if all(len(x) == 0 for x in sets_R):
            continue
        vals.append(mean_pairwise(sets_R))
    return mean(vals), len(vals)

def compute_rows():
    """Build the Table-2 rows: per model, one OVERALL row plus one row per relation."""
    rows = []
    for model in ["gpt-4.1-2025-04-14", "gpt-4.1-mini-2025-04-14"]:
        d = load_runs(model)

        j, n = overall_jaccard(d)                        # tuple-unpack (mean, n)
        rows.append({"model": model, "relation": "OVERALL",
                     "mean_jaccard": j, "n_abstracts": n})

        for R in schema.RELATION_NAMES:                  # the 6 frozen relations
            j, n = relation_jaccard(d, R)
            rows.append({"model": model, "relation": R,
                         "mean_jaccard": j, "n_abstracts": n})
    return rows

def write_rows(rows):
    """Write all rows to results/consistency.csv (overwrite; header + one line per row)."""
    RESULTS_DIR.mkdir(exist_ok=True)                     # create results/ if missing
    out = RESULTS_DIR / "consistency.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:   # newline="" -> no blank lines on Windows
        writer = csv.DictWriter(f,
                    fieldnames=["model", "relation", "mean_jaccard", "n_abstracts"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}")

if __name__ == "__main__":
    rows = compute_rows()
    write_rows(rows)