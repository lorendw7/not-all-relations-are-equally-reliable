"""Interactive labeling tool for the precision-verification step (Step 4).

Walks the annotator through `sample_to_label.csv` one triple at a time, showing
only the abstract + (head, relation, tail) -- never the structural type-check --
so the human verdict stays independent of the type axis (§6.2).

For each row the annotator enters a verdict: 1 (correct), 0 (wrong), s (skip and
leave blank for later), or q (quit). A 0 then prompts for an error_type tag from
the §6.4 taxonomy; a 1 leaves error_type empty.

Resumable and crash-safe: rows that already carry a verdict are skipped on load,
and the whole CSV is rewritten atomically after every single label -- so the
session can be stopped with q (or killed) at any point and resumed where it left
off, with no data loss and the worklist's columns/order preserved.
"""

import csv, os, textwrap
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"

# File being labeled. Repoint at a throwaway copy while developing the
# interactive loop; switch back to the real worklist for the actual labeling.
PATH = DATASET_DIR / "labels" / "sample_to_label.csv"

# Digit -> error_type tag, ordered by the §6.4 decision priority (pick the first
# that applies). Only consulted when a verdict is 0.
TAGS = {"1": "hallucination",
        "2": "inverted",
        "3": "wrong_relation",
        "4": "wrong_span",
        "5": "spurious"}


def load(path):
    """Read every row as a dict and return (rows, original_column_order).

    newline="" is mandatory on Windows so the csv reader handles the newlines
    embedded in the abstract field instead of the OS splitting on them;
    encoding is pinned to utf-8 to match how sample.py wrote the file.
    """
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames


def save(path, rows, fieldnames):
    """Write all rows back, preserving the exact columns and their order.

    Crash-safe: write to a temp file first, then os.replace() -- an atomic swap
    on Windows -- so a crash mid-write can never leave a half-written, corrupt
    sample_to_label.csv.
    """
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)

def render(r, done, total):
    print("\n" * 40)
    print(f"[{done + 1}/{total}] relation = {r['relation']}")
    print(textwrap.fill(r["abstract"], width=90))
    print("-" * 90)
    print(f"    head: {r['head']}")
    print(f"relation: {r['relation']}")
    print(f"    tail: {r['tail']}")

def ask_verdict():
    while True:
        v = input("verdict [1=correct 0=wrong s=skip q=quit]: ").strip()
        if v in ("0", "1", "s", "q"):
            return v
        else:
            print("  invalid - please enter 0, 1, s, or q")

def ask_error_type():
    menu = " ".join(f"{k}={v}" for k, v in TAGS.items())
    while True:
        t = input(f"error type [{menu}]: ").strip()
        if t in TAGS:
            return TAGS[t]
        else:
            print("  invalid - pick a number from the menu")

def main():
    rows, fieldnames = load(PATH)
    total = len(rows)
    done = sum(1 for r in rows if r["verdict"].strip())
    for r in rows:
        if r["verdict"].strip():
            continue
        render(r, done, total)
        v = ask_verdict()
        if v == "q":
            break
        if v == "s":
            continue

        r["verdict"] = v
        r["error_type"] = ask_error_type() if v == "0" else ""
        done += 1
        save(PATH, rows, fieldnames)
    print(f"\n{done}/{total} labeled")


if __name__ == "__main__":
    main()