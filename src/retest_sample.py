"""Step 5 - lock the intra-annotator test-retest subset.

Draws a fixed-seed random subset of the 360 already-judged triples and writes a
BLIND re-judge worksheet (abstract + triple, blank verdict). The first-pass
verdict is deliberately dropped so the ~2-weeks-later re-judgement stays blind.
Frozen once committed; later compared vs sample_to_label.csv for test-retest κ.
"""

import csv
import random

from collect_corpus import DATASET_DIR


N = 40                           # size of the retest subset
SEED = 42                        # documented + frozen once committed -> reproducible draw
LABELS_DIR = DATASET_DIR / "labels"


if __name__ == "__main__":
    rows = []
    with open(LABELS_DIR / "sample_to_label.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    rows.sort(key=lambda r: r["sample_id"])   # fixed order -> seeded draw is reproducible
    rng = random.Random(SEED)
    subset = rng.sample(rows, N)               # N distinct rows, same every run

    out = []
    for r in subset:
        out.append({
            "sample_id": r["sample_id"],
            "model": r["model"],
            "arxiv_id": r["arxiv_id"],
            "relation": r["relation"],
            "head": r["head"],
            "tail": r["tail"],
            "abstract": r["abstract"],
            "verdict": "",             # BLIND: never copy r["verdict"] -> re-judged fresh in ~2 weeks
            "error_type": "",
        })

    fieldnames = ["sample_id", "model", "arxiv_id", "relation",
                  "head", "tail", "verdict", "error_type", "abstract"]
    out_path = LABELS_DIR / "retest_subset.csv"
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out)
    print(f"wrote {len(out)} rows to {out_path}")