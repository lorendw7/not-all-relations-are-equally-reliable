"""Step 8 - what does the per-relation profile buy? (RQ1c, Table 7)

Sections IV-A to IV-E establish that precision varies with the relation type.
This script asks what that profile is FOR: if human verification is scarce,
does routing it by relation type beat spreading it uniformly? Every sampled
triple already carries a relation, a human verdict and a blind Gemini verdict,
so any routing policy can be replayed after the fact -- no new annotation.

A policy is a predicate over one judged triple: "send this to the human?".
Replaying it yields three numbers:

    effort   = |R| / N          share of triples a human must read
    capture  = |R_bad| / E      share of the 125 errors the human sees and removes
    residual = C / (N - |R_bad|)  precision of what survives into the graph

The residual denominator is the one worth reading twice. It is N - |R_bad|,
NOT N - |R|: routing a triple to the human does not remove it, only a NEGATIVE
verdict does. Triples the human reads and approves stay in the graph. The
numerator is C, unchanged by every policy, because the human never deletes a
correct triple. Two endpoints check the formula: route everything and residual
must be exactly 1.0; route nothing and it must equal the baseline 235/360.

THIS SIMULATION IS POST-HOC AND IN-SAMPLE. The three relations in LOW were
chosen because they scored lowest on the very 360 verdicts the policies are
then scored against, so the capture figures are an upper bound on what the
same policy would reach on unseen abstracts. It demonstrates that the profile
is actionable; it is not a validated operating point. The 50% effort figure is
likewise an artifact of the balanced sample (30 triples per model x relation
cell); on the full worklist those three relations are ~53% of extracted
triples. See RESEARCH_PLAN.md 6.7 and the threat in the paper's section VII.

No confidence intervals: this is a deterministic replay over a fixed set of
360 rows, not an estimate from a sampling process.

Input:  dataset/labels/sample_to_label.csv (human verdicts)
        dataset/labels/judge_gemini.csv    (blind LLM-judge verdicts)
        both joined on sample_id by agreement.load_pairs()
Output: results/triage.csv (one row per policy)

Run:  uv run python src/triage.py
"""
import csv
from pathlib import Path
from agreement import load_pairs

# The three lowest-precision relations in Table 1b (.450, .583, .667). Hard-coded
# rather than derived from factors.csv: the policy is part of the study design,
# not a function of the data, and freezing it keeps the replay reproducible. The
# cut is also arbitrary at the margin -- EVALUATED_ON (.667) and PROPOSES (.717)
# are five points apart. That arbitrariness is the in-sample problem, stated in
# the module docstring and in the paper.
LOW = {"ADDRESSES_TASK", "REPORTS_METRIC", "EVALUATED_ON"}


def route_all(r):
    return True


def route_low_relations(r):
    return r["relation"] in LOW


def route_judge_rejects(r):
    return r["j"] == 0


def route_union(r):
    return route_low_relations(r) or route_judge_rejects(r)


POLICIES = [
    ("P0", "everything", route_all),
    ("P1", "low-precision relations", route_low_relations),
    ("P2", "judge rejects only", route_judge_rejects),
    ("P3", "P1 union P2", route_union),
]


def evaluate(rows, routed):
    """Replay one routing policy over the judged rows.

    `routed` is the policy predicate. Returns the raw counts alongside the three
    rates, so the table can be re-checked against integers (180, 78) rather than
    reverse-engineered from ratios.
    """
    N = len(rows)
    E = sum([1 for r in rows if r["h"] == 0])   # human said 0: the 125 errors
    C = N - E                                   # human said 1: the 235 correct

    R = [r for r in rows if routed(r)]          # sent to the human
    R_bad = [r for r in R if r["h"] == 0]       # ...and deleted by them

    return {
        "routed":   len(R),
        "effort":   len(R) / N,
        "captured": len(R_bad),
        # Denominator is E, not N: capture is the share of the 125 ERRORS the
        # policy catches, not their share of all 360 extracted triples.
        "capture_rate": len(R_bad) / E,
        # N - len(R_bad), not N - len(R): see the module docstring.
        "residual_precision": C / (N - len(R_bad)),
    }

if __name__ == "__main__":

    # load_pairs() returns the raw '1'/'0' CSV strings so that agreement.py's kappa
    # can compare the two label lists directly. Cast to int at the door: "0" is
    # truthy and "0" == 0 is False, so leaving them as strings makes every verdict
    # test silently wrong rather than raising.
    rows = [{"relation": r["relation"], "h": int(r["h"]), "j": int(r["j"])} for r in load_pairs()]

    # load_pairs() inner-joins on sample_id, so a missing judge verdict would drop a
    # row silently. Pin the sample size instead.
    N = len(rows)
    E = sum(1 for r in rows if r["h"] == 0)
    assert (N, E, N - E) == (360, 125, 235), (N, E, N - E)

    results = []
    for name, desc, routed in POLICIES:
        row = {"policy": name, "description": desc}
        row.update(evaluate(rows, routed))
        results.append(row)

    by_name = {row["policy"]: row for row in results}

    # Two layers of check, so a failure says which layer broke. The assertion above
    # covers the input; these cover the arithmetic and then the whole chain. They run
    # before anything is written, so a bad triage.csv never reaches disk.
    #
    # Formula. Routing everything means every error is seen and removed, so the
    # residual must be exactly 1.0 -- it is C/(N-E) = 235/235. Any other value means
    # one of the three denominators is wrong.
    assert abs(by_name["P0"]["residual_precision"] - 1.0) < 1e-9, by_name["P0"]

    # End to end. P1 is the one policy whose three numbers can be derived by hand from
    # the pooled precision table (RESULTS.md Table 1b), so it pins the join, the
    # predicate and the aggregation at once: the three LOW relations hold 60 triples
    # each (180/360); their errors are (60-27)+(60-35)+(60-40) = 78 of 125; and the
    # residual is 235/(360-78) = 235/282.
    assert abs(by_name["P1"]["effort"] - 0.500) < 1e-9, by_name["P1"]
    assert abs(by_name["P1"]["capture_rate"] - 0.624) < 1e-3, by_name["P1"]
    assert abs(by_name["P1"]["residual_precision"] - 0.833) < 1e-3, by_name["P1"]

    RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "triage.csv"

    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["policy", "description", "routed", "effort",
                                               "captured", "capture_rate", "residual_precision"])
        writer.writeheader()
        writer.writerows(results)
    print(f"wrote {len(results)} results to {out}")