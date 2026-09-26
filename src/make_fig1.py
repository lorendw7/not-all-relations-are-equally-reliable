"""Step 5 - Figure 1: error taxonomy by relation type (stacked bar).

Counts the human error_type tags among the false positives (verdict=0) in
sample_to_label.csv and draws one stacked bar per relation, split by error type
(hallucination / wrong_relation / wrong_span / spurious). Saves a PNG for the paper.
"""

import csv
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import schema
from collect_corpus import DATASET_DIR
from pathlib import Path
import pandas as pd

# {relation: Counter(error_type -> n)} over the wrong triples only
counts = defaultdict(Counter)
with open(DATASET_DIR / "labels" / "sample_to_label.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r["verdict"] == "0":                 # only false positives carry an error_type
            counts[r["relation"]][r["error_type"]] += 1

relations = schema.RELATION_NAMES               # x-axis: the 6 relations (one bar each)
error_types = ["hallucination", "wrong_relation", "wrong_span", "spurious"]  # stack order (bottom->top)


fig, ax = plt.subplots(figsize=(3.5, 3.0))          # fig = canvas, ax = the plot area
x = range(len(relations))                       # bar positions 0..5
bottom = [0] * len(relations)                   # running base each bar is stacked from

for et in error_types:                          # draw one layer (one error type) at a time
    heights = [counts[rel][et] for rel in relations]    # this layer's count per relation (0 if absent)
    ax.bar(x, heights, bottom=bottom, label=et)         # sits on top of everything drawn so far
    bottom = [b + h for b, h in zip(bottom, heights)]   # raise the base for the next layer

ax.set_xticks(x)
ax.set_xticklabels(relations, rotation=45, ha="right", fontsize=7)  # angled so long names don't overlap
ax.set_ylabel("number of errors")
ax.legend(title="error type")                   # maps colour -> error type
fig.tight_layout()                              # tighten margins so labels aren't clipped

RESULT_DIR = Path(__file__).resolve().parent.parent / "results"
fig.savefig(RESULT_DIR / "fig1_error_taxonomy.png", dpi=200)   # save to disk (no plt.show in a script)
print("saved fig1_error_taxonomy.png")
fig.savefig(RESULT_DIR / "fig1_error_taxonomy.pdf")
print("saved fig1_error_taxonomy.pdf")

df = pd.read_csv(DATASET_DIR / "labels" / "sample_to_label.csv")
wrong = df[df["verdict"] == 0]
print(len(wrong))
print(pd.crosstab(wrong["relation"], wrong["error_type"], margins=True))