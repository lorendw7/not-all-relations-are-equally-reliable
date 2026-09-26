"""Step 5 - human vs LLM-judge agreement (RQ1 cross-check, Table 3 data).

Aligns verdicts on the same 360 triples and reports Cohen's kappa + raw
agreement per relation type, for three pairings: the human against each blind
judge (judge_gemini.csv, judge_claude.csv) and the two judges against each
other. The human is always the anchor; a judge is the object under test, never
ground truth.

Kappa is chance-corrected, so a judge's skewed positive rate (both judges
rarely say '0') deflates it -- report kappa AND raw agreement together (the
"kappa paradox").
"""

import csv
from pathlib import Path
from collect_corpus import DATASET_DIR
import schema


LABELS_DIR = DATASET_DIR / "labels"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

def load_pairs(judge_file="judge_gemini.csv"):
    """Join human and Gemini verdicts on sample_id -> list of {relation, h, j} records.

    h / j are the raw '1'/'0' strings from the CSVs; both stay strings so the two
    label lists compare cleanly in kappa().
    """
    human = {}
    with open(LABELS_DIR / "sample_to_label.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            human[r["sample_id"]] = {"verdict": r["verdict"], "relation": r["relation"]}

    judge = {}
    with open(LABELS_DIR / judge_file, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            judge[r["sample_id"]] = r["supported"]

    records = []
    for sid in human:
        if sid in judge:
            records.append({
                "relation": human[sid]["relation"],
                "h": human[sid]["verdict"],
                "j": judge[sid]
            })

    return records

def load_judge_pairs(file_a="judge_gemini.csv", file_b="judge_claude.csv"):
    """Join two judge files on sample_id -> records with 'h'=A's verdict, 'j'=B's.

    relation still comes from sample_to_label.csv; the human verdict is not read.
    """
    id_to_relation = {}
    with open(LABELS_DIR / "sample_to_label.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            id_to_relation[r["sample_id"]] = r["relation"]

    judge_a = {}
    judge_b = {}
    with open(LABELS_DIR / file_a, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            judge_a[r["sample_id"]] = r["supported"]

    with open(LABELS_DIR / file_b, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            judge_b[r["sample_id"]] = r["supported"]

    res = []
    for sample_id in id_to_relation:
        if sample_id not in judge_a:
            continue
        if sample_id not in judge_b:
            continue
        res.append({
            "relation": id_to_relation[sample_id],
            "h": judge_a[sample_id],
            "j": judge_b[sample_id]
        })

    return res


def kappa(humans, judges):
    """Cohen's kappa for two aligned binary label lists ('1'/'0'). None if undefined."""
    n = len(humans)
    po = sum(h == j for h, j in zip(humans, judges)) / n     # observed agreement
    ph1 = humans.count("1") / n                              # human positive rate
    pj1 = judges.count("1") / n                              # judge positive rate
    pe = ph1 * pj1 + (1 - ph1) * (1 - pj1)                   # chance-expected agreement
    if 1 - pe < 1e-12:                                       # both raters constant & identical -> undefined
        return None
    return (po - pe) / (1 - pe)

def agreement(records):
    """Kappa + raw agreement + n over a set of paired records. Returns (kappa, raw, n)."""
    humans = [r["h"] for r in records]
    judges = [r["j"] for r in records]
    k = kappa(humans, judges)
    raw = sum(h == j for h, j in zip(humans, judges)) / len(records)   # simple % agreement
    return k, raw, len(records)

def per_relation(records):
    """One agreement row per relation type (both models pooled)."""
    rows = []
    for R in schema.RELATION_NAMES:
        sub = [r for r in records if r["relation"] == R]
        k, raw, n = agreement(sub)
        rows.append({"relation": R, "kappa": k, "raw_agreement": raw, "n": n})
    return rows

def write_rows(rows):
    """Write agreement rows to results/agreement.csv (overwrite). None kappa -> blank cell."""
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "agreement.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f,
                                ["pair", "relation", "kappa", "raw_agreement", "n"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    # Three comparisons through the same agreement() / per_relation(): the human
    # anchor against each judge, then the two judges against each other. The
    # third one is what distinguishes shared bias from independent error -- if
    # the judges track each other far better than either tracks the human, they
    # are wrong in the same direction rather than at random.
    pairs = [
        ("human-gemini", load_pairs("judge_gemini.csv")),
        ("human-claude", load_pairs("judge_claude.csv")),
        ("gemini-claude", load_judge_pairs()),
    ]
    rows = []
    for name, records in pairs:
        k, raw, n = agreement(records)                # overall, all 360 pairs
        overall_row = {"pair": name, "relation": "OVERALL",
                       "kappa": k, "raw_agreement": raw, "n": n}
        rows.append(overall_row)                      # OVERALL first, then relations
        rows += [{"pair": name, **r} for r in per_relation(records)]

    write_rows(rows)