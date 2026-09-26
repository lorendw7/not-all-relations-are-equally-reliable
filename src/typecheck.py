"""Step 4 - post-hoc per-relation type check (deterministic, no API, no human).

Structured Outputs only constrains each field to the global entity/relation
enums; it cannot enforce per-relation head/tail typing (e.g. "PROPOSES must be
Paper -> Method"). This module checks that combination after extraction against
schema.RELATIONS.

Two outputs from the temp=0 accuracy run:
  - dataset/labels/worklist_<model>.jsonl: one row per extracted triple (the
    input to human verification), each tagged with type_ok + which side (if any)
    violated the schema typing.
  - results/type_violation.csv: type-violation rate per model x relation (a
    cheap structural-error metric supporting RQ3).

Run:  uv run python src/typecheck.py
"""

import csv
import json
import schema
from collect_corpus import DATASET_DIR
from collections import Counter
from pathlib import Path

# relation name -> (allowed head types, allowed tail types), built from the frozen schema
ALLOWED = {r.name: (set(r.head), set(r.tail)) for r in schema.RELATIONS}
EXTRACT_DIR = DATASET_DIR / "extractions"
LABELS_DIR = DATASET_DIR / "labels"
MODELS = ["gpt-4.1-mini-2025-04-14", "gpt-4.1-2025-04-14"]


def check_triple(t):
    """Return (type_ok, violation) for one triple dict.
    violation ∈ {None, "head", "tail", "both"} — which side broke the schema typing.
    """
    allowed_head, allowed_tail = ALLOWED[t["relation"]]
    bad_head = t["head_type"] not in allowed_head
    bad_tail = t["tail_type"] not in allowed_tail
    if bad_head and not bad_tail:
        return False, "head"
    elif not bad_head and bad_tail:
        return False, "tail"
    elif bad_head and bad_tail:
        return False, "both"
    return True, None


def build_worklist(model):
    """Flatten one model's extractions into a per-triple worklist with type tags.

    Reads dataset/extractions/<model>.jsonl and writes one row per triple to
    dataset/labels/worklist_<model>.jsonl. Opened in "w" mode (not append):
    this transform is deterministic and free, so it is simply regenerated on
    each run rather than resumed.
    """
    in_path = EXTRACT_DIR / f"{model}.jsonl"
    out_path = LABELS_DIR / f"worklist_{model}.jsonl"
    LABELS_DIR.mkdir(exist_ok=True)

    with open(in_path, "r", encoding="utf-8") as f, \
        open(out_path, "w", encoding="utf-8") as out:
        for line in f:
            rec = json.loads(line)
            for i, t in enumerate(rec["triples"]):
                type_ok, violation = check_triple(t)
                row = {"triple_id": f"{rec['arxiv_id']}#{i}",  # stable key: id + index within the abstract
                       "arxiv_id": rec["arxiv_id"],
                       "model": model,  # provenance / RQ3 grouping key
                       **t,  # spread the triple's head/head_type/relation/tail/tail_type
                       "type_ok": type_ok,
                       "violation": violation}
                out.write(json.dumps(row) + "\n")


def summarize():
    """Aggregate the worklists into a type-violation rate per model x relation.

    Reads both worklists and writes results/type_violation.csv. The rate is the
    fraction of extracted triples whose head/tail typing is invalid for their
    relation -- a structural-error signal expected to favour the stronger model.
    """
    total = Counter()    # (model, relation) -> triples seen
    bad = Counter()      # (model, relation) -> type-violating triples

    for model in MODELS:
        for relation in schema.RELATION_NAMES:
            total[(model, relation)] =0
            bad[(model, relation)] = 0

    for model in MODELS:
        path = LABELS_DIR / f"worklist_{model}.jsonl"
        for line in open(path, "r", encoding="utf-8"):
            row = json.loads(line)
            key = (row["model"], row["relation"])
            total[key] += 1
            if not row["type_ok"]:
                bad[key] += 1
    out_csv_dir = Path(__file__).parent.parent  / "results"
    out_csv_dir.mkdir(exist_ok=True)
    out_csv = out_csv_dir / "type_violation.csv"
    write_csv(bad, total, out_csv)


def write_csv(bad, total, out_csv):
    """Write one CSV row per (model, relation) with its type-violation rate.

    newline="" is required on Windows so the csv module does not insert a blank
    line between rows.
    """
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "relation", "total", "violations", "violation_rate"])
        for key in sorted(total):
            model, relation = key
            n = total[key]
            v = bad[key]
            rate = v / n if n > 0 else 0.0
            w.writerow([model, relation, n, v, rate])

        model_total, model_bad = Counter(), Counter()
        for (model, relation), count in total.items():
            model_total[model] += count
            model_bad[model] += bad[(model, relation)]
        for model in sorted(model_total):
            n, v = model_total[model], model_bad[model]
            w.writerow([model, "ALL", n, v, v / n if n > 0 else 0.0])

def breakdown():
    """Per (model, relation, head_type, tail_type) count, with type_ok.

    A full pivot of the worklists: shows exactly which head/tail type
    combinations the models produce for each relation (structural axis only;
    factual correctness comes later from human labels).
    """
    counts = Counter()
    for model in MODELS:
        path = LABELS_DIR / f"worklist_{model}.jsonl"
        for line in open(path, "r", encoding="utf-8"):
            row = json.loads(line)
            # composite key: the full (model, relation, head_type, tail_type, type_ok)
            # signature -- one bucket per distinct typing pattern
            key = (row["model"], row["relation"],
                   row["head_type"], row["tail_type"], row["type_ok"])
            counts[key] += 1

    out_csv = Path(__file__).parent.parent / "results" / "violation_breakdown.csv"
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "relation", "head_type", "tail_type", "type_ok", "count"])
        for key in sorted(counts):
            w.writerow([*key, counts[key]])   # *key spreads the 5-tuple, then append the count

if __name__ == "__main__":
    for m in MODELS:
        build_worklist(m)

    summarize()
    breakdown()