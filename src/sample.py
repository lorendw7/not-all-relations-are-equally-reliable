"""Step 4 - stratified sampling of triples for human verification.

typecheck.py flattened every extracted triple into per-model worklists
(dataset/labels/worklist_<model>.jsonl). Judging all ~1,800 triples by hand is
too much for one annotator, so this module draws a fixed-seed stratified sample
of N_PER_CELL triples per (model x relation) cell, joins the abstract text back
in, and writes dataset/labels/sample_to_label.csv -- the worklist for manual
precision judging (RQ1).

Design constraints (RESEARCH_PLAN 6.2):
  - Stratify by (model x relation): every relation type must get enough judged
    triples for a per-relation precision estimate. A plain random sample would
    starve rare relations (e.g. IMPROVES_OVER).
  - Reproducible: one seeded RNG, a fixed cell-iteration order, and each cell
    sorted before sampling. The sample is frozen once labeling starts.
  - Blind: the annotator sees only the abstract + the triple, never the type
    check, so type_ok/violation are deliberately kept out of the CSV.

Run:  uv run python src/sample.py
"""

import random, json, csv
import schema
from collect_corpus import DATASET_DIR
from collections import defaultdict

N_PER_CELL = 30                # triples drawn per (model x relation) cell
SEED = 42                      # same seed as the corpus sample; recorded in docs
MODELS = ["gpt-4.1-2025-04-14", "gpt-4.1-mini-2025-04-14"]


if __name__ == "__main__":
    # Read both worklists and bucket every triple by its (model, relation) cell.
    # defaultdict(list) auto-creates an empty list the first time a cell is seen,
    # so each row can be appended without an existence check.
    group = defaultdict(list)
    for model in MODELS:
        path = DATASET_DIR / "labels" / f"worklist_{model}.jsonl"
        for line in open(path, "r", encoding="utf-8"):
            row = json.loads(line)
            group[(model, row["relation"])].append(row)

    # arxiv_id -> abstract text, used to join the source text back onto each triple.
    abstracts = dict()
    for line in open(DATASET_DIR / "abstracts.jsonl", "r", encoding="utf-8"):
        row = json.loads(line)
        abstracts[row["arxiv_id"]] = row["abstract"]

    # Draw N_PER_CELL per cell with one seeded RNG. Iterating MODELS x RELATION_NAMES
    # (both fixed orders) and sorting each cell first makes the draw reproducible.
    rng = random.Random(SEED)
    sample = []
    for model in MODELS:
        for relation in schema.RELATION_NAMES:
            cell = group.get((model, relation), [])
            cell.sort(key=lambda t: t["triple_id"])  # stable within-cell order
            k = min(N_PER_CELL, len(cell))  # take all if the cell has < N
            sample.extend(rng.sample(cell, k))  # extend: splice in the batch
            print(f"{model} {relation} -> {k} / {len(cell)}")  # per-cell coverage

    # Build the annotation rows: a globally-unique id (triple_id collides across
    # models, so prefix with model), the triple, the joined abstract, and two blank
    # columns the annotator fills in. type_ok/violation are intentionally omitted.
    rows = []
    for r in sample:
        rows.append({
            "sample_id": f'{r["model"]}|{r["triple_id"]}',
            "model": r["model"],
            "arxiv_id": r["arxiv_id"],
            "relation": r["relation"],
            "head": r["head"],
            "tail": r["tail"],
            "abstract": abstracts[r["arxiv_id"]],
            "verdict": "",  # filled by the annotator: correct / incorrect
            "error_type": "",  # filled only when verdict is incorrect (taxonomy)
        })
    rng.shuffle(rows)  # mix both models together -> blind judging

    # Write the CSV. extrasaction="ignore" drops any keys outside fieldnames (a
    # safety net for stray worklist fields); newline="" stops blank lines on Windows.
    fieldnames = ["sample_id", "model", "arxiv_id", "relation",
                  "head", "tail", "verdict", "error_type", "abstract"]

    out_path = DATASET_DIR / "labels" / "sample_to_label.csv"
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
        print(f"wrote {len(rows)} rows -> {out_path}")