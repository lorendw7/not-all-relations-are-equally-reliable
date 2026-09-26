"""Step 5b - paper-level cluster bootstrap for the per-relation precision CIs.

precision.py's Wilson interval treats the 30 judged triples in a cell as 30
independent Bernoulli trials, but they are nested in abstracts (see
cluster_stats.py: ICC 0.112, DEFF 1.25). This script rebuilds the intervals by
resampling *abstracts* with replacement, which keeps each abstract's rows
together in every replicate and so preserves the within-abstract correlation.

Two intervals are written per cell, from the same 10,000 replicates:

  ci_low/ci_high              the 2.5th-97.5th percentile of the replicates.
                              Bounded by [0, 1] and blind to it: the boundary
                              cell returns an upper bound of exactly 1.000.
  ci_low_logit/ci_high_logit  a symmetric normal-approximation interval built
                              on the logit scale and transformed back, which is
                              what the paper quotes. Symmetric in logit space
                              becomes asymmetric in probability space, and the
                              bound stays strictly inside (0, 1).

A percentile interval is invariant under any monotone transform, so logit does
nothing for it -- the transform only earns its place once the interval is built
as centre +/- width.

Input:  dataset/labels/sample_to_label.csv (verdict 1/0 per triple, human-judged)
Output: results/precision_bootstrap.csv

Run:  uv run python src/resample.py
"""
import csv, collections
import random
import statistics
from pathlib import Path
from scipy.special import logit, expit
from scipy.stats import norm

Z = norm.ppf(0.975)    # 1.95996..., the 97.5th percentile of the standard normal

# Paths are derived from this file's location, not from the working directory,
# so the script runs the same from the repo root as from src/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABELS = PROJECT_ROOT / "dataset" / "labels" / "sample_to_label.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

# Load the labeled sample: one row per (paper, model, relation, triple) with a
# human verdict of '1'/'0', or '' if this row hasn't been judged yet.
with open(LABELS, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

# Group rows by source paper (arxiv_id). Triples pulled from the same paper are
# not independent, so the bootstrap below resamples at the paper level, not the
# individual-row level.
papers = collections.defaultdict(list)
for r in rows:
    papers[r["arxiv_id"]].append(r)

def all_precisions(subset, shrink=False, key=lambda r: (r["model"], r["relation"])):
    """Precision (k/n) per group over a set of labeled rows.

    key chooses the grouping level and defaults to the (model, relation) cell of
    Table 1, n=30. Passing key=lambda r: r["relation"] pools the two tiers into
    the per-relation level the paper's merged table reports, n=60. Everything
    downstream -- the paper-level resampling, the shrink correction, the logit
    transform -- is identical at either level; only the counters differ.

    shrink=True applies the Haldane-Anscombe correction, (k+0.5)/(n+1), which
    keeps every replicate strictly inside (0, 1) so that logit() below stays
    finite -- roughly 5% of replicates for the boundary cell are all-correct,
    and a single logit(1.0) = inf would poison that cell's standard deviation.
    It nudges toward 0.5 by at most 1/(2(n+1)), leaves 0.5 itself untouched,
    and is *only* for the replicates: the reported point estimate must stay the
    raw k/n that precision.csv and Table 1 report. Default False so that the
    point-estimate call below is unaffected.
    """
    k = collections.Counter()  # correct (verdict == '1') count per cell
    n = collections.Counter()  # judged (verdict != '') count per cell
    subset = [r for r in subset if r["verdict"] != '']  # drop not-yet-judged rows

    for r in subset:
        cell = key(r)
        n[cell] += 1
        k[cell] += int(r['verdict'])

    if shrink:
        return {c: (k[c] + 0.5) / (n[c] + 1) for c in n}

    return {c: k[c] / n[c] for c in n}

rng = random.Random(20260806)  # fixed seed so the bootstrap is reproducible
pids = list(papers.keys())
B = 10000  # number of bootstrap resamples

# (model, relation) -> list of B bootstrap precision estimates. Two sets, from
# the *same* replicates, because the two intervals need different inputs:
# the percentile interval must see the raw k/n (shrinking it would smooth away
# the 1.000 upper bound that motivates the logit interval in the first place),
# while the logit interval needs every replicate strictly inside (0, 1).
samples = collections.defaultdict(list)         # raw k/n      -> percentile CI
samples_shrunk = collections.defaultdict(list)  # Haldane-Ansc -> logit CI

# The same two intervals again, one level up: relations with the two tiers pooled
# (n=60), which is the level the paper's merged Table 1 actually reports. These
# are accumulated inside the *same* loop, off the *same* replicates, rather than
# by re-running the script with a different key -- that keeps the two levels
# mutually consistent (a replicate that happened to draw few ADDRESSES_TASK
# papers moves both alike) and keeps seed=20260806 the single thing that pins
# every interval in this file.
samples_pooled        = collections.defaultdict(list)   # key=relation, raw k/n
samples_pooled_shrunk = collections.defaultdict(list)   # key=relation, Haldane-Anscombe

# Paper-level (cluster) bootstrap: each round resamples papers *with replacement*
# (so the same paper can be drawn more than once, or not at all), then
# recomputes per-cell precision over every row belonging to the drawn papers.
# Resampling whole papers -- instead of individual rows -- keeps each paper's
# rows together in every replicate, which is what preserves the within-paper
# correlation between triples pulled from the same abstract. This is what
# makes it a *cluster* bootstrap rather than a plain row-level bootstrap.
for b in range(B):
    drawn = rng.choices(pids, k=len(pids))  # paper IDs drawn with replacement
    boot_rows = [row for pid in drawn for row in papers[pid]]  # flatten drawn papers' rows
    # Both calls see the same boot_rows, so the two sets are paired replicate by
    # replicate rather than being two independent bootstraps -- the RNG is drawn
    # from once per round and the seed's reproducibility is unaffected.
    for cell, p in all_precisions(boot_rows).items():
        samples[cell].append(p)
    for cell, p in all_precisions(boot_rows, shrink=True).items():
        samples_shrunk[cell].append(p)
    for cell, p in all_precisions(boot_rows, key=lambda r: r["relation"]).items():
        samples_pooled[cell].append(p)
    for cell, p in all_precisions(boot_rows, shrink=True, key=lambda r: r["relation"]).items():
        samples_pooled_shrunk[cell].append(p)

# The number actually reported for each cell: precision computed once on the
# real (non-resampled) labeled data. `samples[cell]` is never averaged into
# this -- it is only used below to size the uncertainty band around it.
rows_precisions = all_precisions(rows)

rows_precisions_pooled = all_precisions(rows, key=lambda r: r["relation"])

# Separate output file from results/precision.csv on purpose: that file holds
# the Wilson score interval computed by src/precision.py (per-row binomial CI,
# assumes independent rows). This is a different, cluster-aware CI for the
# same point estimates, kept side by side rather than overwriting the Wilson
# numbers RESULTS.md already documents.
with open(RESULTS_DIR / "precision_bootstrap.csv",
          'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['model', 'relation', 'precision', 'ci_low', 'ci_high',
                     'ci_low_logit', 'ci_high_logit', 'n_boot'])
    for cell, vals in samples.items():
        # samples[cell] is now B (or fewer, for sparse cells) precision values,
        # one per simulated "re-run" of the paper-sampling process. Sorting
        # them and reading off the 2.5th/97.5th percentile gives a 95%
        # confidence interval: the middle 95% of where precision landed
        # across replicates. n_boot (written below) records len(vals) per
        # cell, so a cell where a bootstrap draw sometimes missed every paper
        # carrying it -- and n_boot < B -- shows up in the CSV itself.
        cuts = statistics.quantiles(vals, n=40, method='inclusive')
        lo, hi = cuts[0], cuts[-1]
        point_estimate = rows_precisions[cell]

        # Logit-scale interval. The spread of the replicates on the logit scale
        # is the standard error there; the centre is the logit of the *reported*
        # point estimate, not the mean of the replicates. centre +/- Z*se is
        # symmetric in logit space, and expit turns it into an asymmetric
        # interval in probability space -- shorter above the estimate than
        # below, which is the right shape for a proportion near 1.
        logit_vals = [logit(p) for p in samples_shrunk[cell]]
        se = statistics.stdev(logit_vals)  # sample sd: replicates are a sample
        center = logit(point_estimate)
        lo_logit = expit(center - Z * se)
        hi_logit = expit(center + Z * se)

        # logit(0) and logit(1) are infinite; no cell is at either bound today
        # (the extremes are .400 and .867), so assert it rather than assume it.
        assert 0.0 < point_estimate < 1.0, (cell, point_estimate)
        assert lo_logit < point_estimate < hi_logit, (cell, lo_logit, hi_logit)
        # The point of the whole transform, stated where it is produced.
        assert hi_logit < 1.0, (cell, hi_logit)

        writer.writerow([cell[0], cell[1], point_estimate,
                         lo, hi, lo_logit, hi_logit, len(vals)])

with open(RESULTS_DIR / "precision_pooled_bootstrap.csv",
          'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['factor', 'level', 'precision', 'ci_low', 'ci_high',
                     'ci_low_logit', 'ci_high_logit', 'n_boot'])
    for cell, vals in samples_pooled.items():
        cuts = statistics.quantiles(vals, n=40, method='inclusive')
        lo, hi = cuts[0], cuts[-1]
        point_estimate = rows_precisions_pooled[cell]

        logit_vals = [logit(p) for p in samples_pooled_shrunk[cell]]
        se = statistics.stdev(logit_vals)
        center = logit(point_estimate)
        lo_logit = expit(center - Z * se)
        hi_logit = expit(center + Z * se)

        assert 0.0 < point_estimate < 1.0, (cell, point_estimate)
        assert lo_logit < point_estimate < hi_logit, (cell, lo_logit, hi_logit)
        assert hi_logit < 1.0, (cell, hi_logit)

        # The factor column is hard-coded because relation is the only level
        # pooled here. Add a second key (model) and this has to become a
        # per-accumulator label, or every row lands mislabelled as "relation".
        writer.writerow(["relation", cell, point_estimate,
                         lo, hi, lo_logit, hi_logit, len(vals)])
