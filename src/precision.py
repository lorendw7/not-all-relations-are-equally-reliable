"""Step 5 - per-relation precision (RQ1 headline, RQ3 model tier).

Reads the human verification labels and computes precision = correct / total
per (model x relation) cell -- the 12 cells that become Table 1. Precision is a
binomial proportion at small n (N=30/cell) with p often near 1.0, so each cell
also needs a 95% Wilson score interval (statsmodels proportion_confint,
method='wilson') -- the Wald interval degenerates or exceeds [0, 1] here.

Input:  dataset/labels/sample_to_label.csv (verdict 1/0 per triple, human-judged)
Output: results/precision.csv  (per model x relation: correct, total, precision, CI)

Run:  uv run python src/precision.py
"""
from pathlib import Path
import pandas as pd
from collect_corpus import DATASET_DIR
from statsmodels.stats.proportion import proportion_confint

LABELS_DIR = DATASET_DIR / "labels"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

path = LABELS_DIR / "sample_to_label.csv"
df = pd.read_csv(path)

# verdict must be numeric so sum = count of 1s (correct) and size = total.
# (.astype returns a new column, not in place -- assign it back.)
df["verdict"] = df["verdict"].astype(int)

# Split the 360 rows into 2x6 = 12 (model, relation) groups, then aggregate
# each group to one row: correct = sum of verdicts (1s), total = row count.
df_result = df.groupby(['model', 'relation']).agg(correct = ('verdict', 'sum'),
                                                  total = ('verdict', 'size'))
df_result['precision'] = df_result['correct'] / df_result['total']

# 95% Wilson CI per cell. proportion_confint is vectorized over the columns and
# returns (lower array, upper array). method="wilson" is essential: the default
# "normal" (Wald) collapses to a zero-width [1, 1] at p=1.0 and can exceed [0, 1].
low, high = proportion_confint(df_result['correct'], df_result['total'],
                               alpha=0.05, method="wilson")
df_result['ci_low'] = low
df_result['ci_high'] = high

df_result = df_result.reset_index()  # MultiIndex (model, relation) -> plain columns
df_result.to_csv(RESULTS_DIR / "precision.csv", index=False)
