"""Intra-annotator test-retest kappa (Table 3).

Compares the annotator's first-pass verdicts (sample_to_label.csv) against the
blind second pass on the frozen 40-triple subset (retest_subset.csv), matched by
sample_id, and reports Cohen's kappa plus raw agreement. A high value is the goal
here: it shows the judging criteria are stable over time. Reuses agreement.py's
kappa unchanged.
"""

import csv
from collect_corpus import DATASET_DIR
from agreement import kappa          # same Cohen's kappa as the human-judge check

LABELS_DIR = DATASET_DIR / "labels"

def load_verdicts(path):
    """sample_id -> verdict, for rows that carry a verdict (blank rows skipped)."""
    v = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["verdict"].strip():          # ignore any not-yet-judged row
                v[r["sample_id"]] = r["verdict"]

    return v

if __name__ == "__main__":
    first = load_verdicts(LABELS_DIR / "sample_to_label.csv")    # pass 1 (all 360)
    second = load_verdicts(LABELS_DIR / "retest_subset.csv")     # pass 2 (the 40)

    # safety check: every re-judged id must exist in the first pass, else the
    # files are mismatched -- fail loudly rather than compute a wrong kappa.
    missing = set(second) - set(first)
    assert not missing, f"reset ids not in first pass: {missing}"

    ids = sorted(second)                      # the 40 ids, in one fixed order
    p1 = [first[i] for i in ids]              # first-pass verdicts, aligned to ids
    p2 = [second[i] for i in ids]             # second-pass verdicts, same order

    k = kappa(p1, p2)
    raw = sum(a == b for a, b in zip(p1, p2)) / len(ids)         # simple % agreement
    print(f"test-retest kappa: {k:.3f} raw = {raw:.3f} n = {len(ids)}")