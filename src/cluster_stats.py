"""Step 5c - clustering diagnostics: ICC, design effect, effective sample size.

The 360 verdicts come from 155 abstracts, not 360 independent draws. Triples
sharing an abstract share its phrasing and ambiguity, so precision.py's Wilson
intervals and factors.py's chi-square -- both of which assume independent rows --
overstate certainty. This script measures how much.

The chain is ICC -> DEFF -> n_eff:
  ICC   how strongly verdicts cluster within a paper (0 = not at all, 1 = a
        paper's triples are all-right or all-wrong, so it carries one bit)
  DEFF  how much that inflates the variance over the independence assumption
  n_eff N / DEFF -- the number of independent rows this sample is really worth

It then applies the design effect to the omnibus tests of factors.py as a
Rao-Scott FIRST-ORDER correction: chi2_adj = chi2 / DEFF, same degrees of
freedom. First-order means one global DEFF stands in for every cell of the
table, rather than a per-cell design effect -- standard practice at this scale,
but the paper must say "first-order" rather than "design-adjusted".

Input:  dataset/labels/sample_to_label.csv (verdict 1/0 per triple, human-judged)
        results/factor_tests.csv           (chi-square per factor, from factors.py)
Output: results/cluster_stats.csv  (one row: N, k, both mean cluster sizes, ICC,
        DEFF, n_eff). The corrected omnibus tests are printed, not written --
        factor_tests.csv is left untouched so the uncorrected statistics stay
        on record beside the corrected ones, the same way precision.csv and
        precision_bootstrap.csv coexist.

Run:  uv run python src/factors.py && uv run python src/cluster_stats.py
"""
import csv
import collections
from pathlib import Path

import pandas as pd
# Aliased: a bare `chi2` would be shadowed the moment a chi-square *statistic*
# is bound to that name, and the resulting AttributeError is opaque.
from scipy.stats import chi2 as chi2_dist

DATA_ROOT = Path(__file__).resolve().parent.parent
LABELS = DATA_ROOT / "dataset" / "labels" / "sample_to_label.csv"
RESULTS_DIR = DATA_ROOT / "results"

with open(LABELS, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

rows = [r for r in rows if r["verdict"] != ""]  # drop not-yet-judged rows

# arxiv_id -> list of that paper's verdicts, e.g. "2606.17182v1" -> [1, 0, 1].
# The cluster is the paper, NOT (paper x model): what the triples share is the
# abstract's ambiguity, which both models read. This matches how resample.py
# resamples, so the two scripts describe the same clustering.
# csv gives strings, so int() -- otherwise "1" + "0" concatenates to "10".
clusters = collections.defaultdict(list)
for r in rows:
    clusters[r["arxiv_id"]].append(int(r["verdict"]))


N = len(rows)      # 360 judged triples
k = len(clusters)  # 155 papers
p_bar = sum(sum(v) for v in clusters.values()) / N  # overall precision

# One (n_i, p_i) pair per paper: its size and its own precision. Kept as tuples
# rather than two parallel lists so the size and the proportion can never drift
# out of alignment.
cells = [(len(v), sum(v) / len(v)) for v in clusters.values()]

# One-way random-effects ANOVA estimator of the ICC.
#
# MSB: between-paper variance -- how far each paper's precision sits from the
# overall precision. Zero if every paper scores exactly p_bar. The /(k-1) is
# what makes it a *mean* square; dropping it leaves a raw sum ~150x too large.
msb = sum(n * (p - p_bar) ** 2 for n, p in cells) / (k - 1)

# MSW: within-paper variance. p*(1-p) is the Bernoulli variance, so it is 0 for
# a paper that is all-right or all-wrong and peaks at .25 for a 50/50 paper --
# exactly "how mixed is this paper internally". The n_i*p_i*(1-p_i) shorthand
# for the within-paper sum of squares holds only because verdicts are 0/1.
msw = sum(n * p * (1 - p) for n, p in cells) / (N - k)

# n0: the effective average cluster size, corrected for unequal cluster sizes
# (they run 1 to 8 here). Equals the plain mean only when all clusters match.
sum_n2 = sum(n ** 2 for n, p in cells)  # also feeds Kish's n_a below
n0 = (N - sum_n2 / N) / (k - 1)

# ICC. Note (n0 - 1) * msw is ONE term of the denominator: in Python the
# parentheses and the * are both mandatory, unlike the maths notation.
# Clipped at 0 because MSB < MSW (a negative ICC) is a small-sample artifact,
# and a negative ICC would push DEFF below 1 -- claiming that clustering made
# the sample *more* certain, which is nonsense.
icc = max(0.0, (msb - msw) / (msb + (n0 - 1) * msw))

assert 0.0 <= icc <= 1.0, icc

# Two different "average cluster size", both reported, neither interchangeable:
#
# mean_size  the arithmetic mean, 2.32 -- the figure the paper's footnote quotes
#            (RESEARCH_PLAN.md section 10).
# n_a        Kish's size-weighted mean, 3.24. The squaring weights big clusters
#            more heavily, because an 8-triple paper loses far more information
#            to clustering than eight 1-triple papers do. This is the one the
#            design effect needs; RESULTS.md must say so, since it is the larger
#            of the two and quoting 2.32 here would understate DEFF.
mean_size = N / k
n_a = sum_n2 / N

# Kish's design effect: the factor by which clustering inflates the variance
# over what the independence assumption gives. Floored at 1 -- clustering can
# only ever cost information, never add it. (Redundant while icc is clipped
# above, but it states the invariant where it belongs.)
deff = max(1.0, 1 + (n_a - 1) * icc)

# Effective sample size: 360 rows are worth this many independent ones. Bounded
# by k (ICC = 1, each paper carries one bit) and N (ICC = 0, no clustering).
n_eff = N / deff
assert k <= n_eff <= N, n_eff

print(msb, msw, n0, icc)
print(mean_size, n_a, deff, n_eff)

# Rao-Scott first-order correction of the omnibus tests. chi-square scales with
# the sample size -- duplicating every row doubles both O and E and so doubles
# every (O-E)^2/E term -- so dividing by DEFF hands back the sample size the
# independence assumption over-claimed. The dof describe the table's shape, not
# its size, and are left alone.
assert (RESULTS_DIR / "factor_tests.csv").exists(), "run src/factors.py first"
tests = pd.read_csv(RESULTS_DIR / "factor_tests.csv")

# BOTH factors are corrected, not just `relation`. The model row is already null
# and stays null (0.58 -> 0.62); reporting it corrected is what shows the
# correction was applied to the whole family rather than to the one result it
# could plausibly have killed.
tests["chi2_adj"] = tests["chi2"] / deff
tests["p_adj"] = chi2_dist.sf(tests["chi2_adj"], tests["dof"])

# Direction checks, not value checks: DEFF >= 1 means the statistic can only
# shrink and the p-value can only grow. These hold whatever the data becomes.
assert (tests["chi2_adj"] <= tests["chi2"]).all()
assert (tests["p_adj"] >= tests["p"]).all()

print(tests)

pd.DataFrame([{
    "n_rows": N,
    "n_clusters": k,
    "mean_cluster_size": mean_size,
    "kish_n": n_a,
    "icc": icc,
    "deff": deff,
    "n_eff": n_eff,
}]).to_csv(RESULTS_DIR / "cluster_stats.csv", index=False)