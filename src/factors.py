"""Step 5b - which factor explains precision: relation type or model tier (RQ1a)?

Table 1 spreads 0.40-0.87 across twelve (model x relation) cells whose Wilson
intervals overlap heavily, and overlapping intervals are not a test. The paper's
central claim -- reliability is a property of the relation type, not of the model
-- therefore needs one. This script pools the same 360 verdicts along each factor
in turn, then tests both.

Pooling to n=60 per relation is legitimate because the sample is stratified at 30
triples per (model x relation) cell, so the two factors are balanced and neither
biases the other. But the pooled figures are NOT weighted by how often each
relation occurs in the corpus: they describe the six relation types, not the
average triple in the extracted graph.

Input:  dataset/labels/sample_to_label.csv (verdict 1/0 per triple, human-judged)
Output: results/factors.csv       (pooled precision + Wilson CI, one row per factor level)
        results/factor_tests.csv  (chi-square per factor: statistic, dof, p, n)

Both tests are computed WITHOUT Yates' continuity correction, so the 2x2 model
statistic stays identical to the square of the two-proportion z-statistic and
stays comparable with the uncorrected 6x2 relation test.

Run:  uv run python src/factors.py
"""
from pathlib import Path
import pandas as pd
from collect_corpus import DATASET_DIR
from statsmodels.stats.proportion import proportion_confint, proportions_ztest
from scipy.stats import chi2_contingency


LABELS_DIR = DATASET_DIR / "labels"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

path = LABELS_DIR / "sample_to_label.csv"
df = pd.read_csv(path)
# verdict must be numeric so sum = count of 1s (correct) and size = total.
# (.astype returns a new column, not in place -- assign it back.)
df["verdict"] = df["verdict"].astype(int)

def pooled(df, key):
    df_result = df.groupby(key).agg(correct=("verdict", "sum"),
                                             total=("verdict", "size"))

    df_result['precision'] = df_result['correct'] / df_result['total']

    # 95% Wilson CI per cell. proportion_confint is vectorized over the columns and
    # returns (lower array, upper array). method="wilson" is essential: the default
    # "normal" (Wald) collapses to a zero-width [1, 1] at p=1.0 and can exceed [0, 1].
    low, high = proportion_confint(df_result['correct'], df_result['total'],
                                   alpha=0.05, method="wilson")
    df_result['ci_low'] = low
    df_result['ci_high'] = high

    # The group key becomes the index, so restore it as a column -- then rename it
    # to a fixed "level" and tag the factor, so the relation table and the model
    # table share one shape and can be concatenated without half-empty columns.
    out = df_result.reset_index()
    out = out.rename(columns={key: 'level'})
    out.insert(0, "factor", key)

    return out

by_relation = pooled(df, "relation")
by_model = pooled(df, "model")
pd.concat([by_relation, by_model],
          ignore_index=True).to_csv(RESULTS_DIR / "factors.csv", index=False)

ct = pd.crosstab(df["relation"], df["verdict"])

chi2, p, dof, ex = chi2_contingency(ct)
print(chi2)
print(p)
print(dof)
print(ex)

assert ex.min() >= 5, ex

# Decompose the statistic by row: each relation's own (O-E)^2/E summed over its
# two verdict cells. These add back up to chi2, so they show how much of the
# effect each relation is responsible for -- which bounds how the result may be
# stated. A single dominant row means "this relation is the outlier", not "all
# six differ". See RESULTS.md Table 1b and the threats in section 5.
print((((ct - ex) ** 2) / ex).sum(axis=1))

ct_model = pd.crosstab(df["model"], df["verdict"])
chi2_m, p_m, dof_m, ex_m = chi2_contingency(ct_model, correction=False)
print(chi2_m)
print(p_m)
print(dof_m)
print(ex_m)

assert ex_m.min() >= 5, ex_m

tests = pd.DataFrame([
    {"factor": "relation", "chi2": chi2, "p": p, "dof": dof, "n": len(df)},
    {"factor": "model", "chi2": chi2_m, "p": p_m, "dof": dof_m, "n": len(df)},
])

tests.to_csv(RESULTS_DIR / "factor_tests.csv", index=False)
z, _ = proportions_ztest(by_model["correct"], by_model["total"])

assert abs(z ** 2 - chi2_m) < 1e-9, (z**2, chi2_m)
