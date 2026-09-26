# Per-Relation Reliability of LLM-Based Knowledge Graph Extraction

Corpus, human judgments, LLM judgments, and evaluation code for the paper
*"Per-Relation Reliability of LLM-Based Knowledge Graph Extraction: A Verification-First
Study on Scientific Abstracts"* by Shandong He and Kenji Ono. The paper was
accepted as a short paper at the CANDAR 2026 WANC workshop (EasyChair #142);
the final proceedings version is pending.

We extract `(head, relation, tail)` triples from 200 arXiv CS abstracts with two pinned
GPT-4.1 snapshots under a frozen six-relation schema, and measure reliability per
individual relation type: verification-first human precision over 360 judged triples,
self-consistency across repeated runs, structural type adherence over all 1,831
extracted triples, two blind LLM judges checked against the human annotator, and a
post-hoc simulation of where human verification effort should go.

Everything the paper reports is recomputable from what is in this repository. The
extraction and judging outputs are committed rather than left to be regenerated,
because they come from model APIs and a re-run is not guaranteed to return the same
text.

## Layout

| Path | Content |
|---|---|
| `dataset/abstracts.jsonl` | The frozen 200-abstract corpus (arXiv API; seeded sample, `seed=42`) |
| `dataset/corpus_meta.json` | Collection provenance: query, window, seed, counts |
| `dataset/extractions/` | All extracted triples, `temperature=0` accuracy run, both model tiers |
| `dataset/extractions/consistency/` | The `temperature=0.7`, K=3 consistency runs (first 40 abstracts) |
| `dataset/labels/sample_to_label.csv` | The stratified 360-triple sample with every human verdict and error tag |
| `dataset/labels/judge_gemini.csv` | Blind Gemini verdicts on the same 360 triples |
| `dataset/labels/judge_claude.csv` | Blind Claude verdicts on the same 360 triples, same rubric |
| `dataset/labels/retest_subset.csv` | The blind 40-triple test–retest re-judgments |
| `dataset/labels/worklist_*.jsonl` | The full extraction worklists the sample was drawn from |
| `results/` | Every computed table, plus the figures |
| `src/` | Collection, extraction, judging, and evaluation scripts |
| `survey/volume_sweep.md` | The workshop-volume sweep behind the paper's novelty claim: the volumes walked, the screening criterion, and the full-text candidates with the reason each was cleared |

## Reproduction map

| Result in the paper | Script | Output |
|---|---|---|
| Precision per relation, per tier and pooled (Table I, point estimates) | `precision.py`, `factors.py` | `precision.csv`, `factors.csv` |
| **The 95% intervals printed in Table I** | `resample.py` | `precision_pooled_bootstrap.csv` — columns `ci_low_logit`, `ci_high_logit` |
| Per-cell intervals at n=30 (computed, not quoted in the paper) | `resample.py` | `precision_bootstrap.csv` |
| Clustering diagnostics: ICC, design effect, effective n (Sec. III-C) | `cluster_stats.py` | `cluster_stats.csv` |
| Omnibus factor tests and their Rao–Scott correction (Sec. IV-A) | `factors.py`, then `cluster_stats.py` | `factor_tests.csv`; corrected χ² printed |
| Error-type shares quoted in prose (Sec. IV-B) | `make_fig1.py` | cross-tab printed; `fig1_error_taxonomy.pdf` |
| Type violations (Table II) | `typecheck.py` | `type_violation.csv`, `violation_breakdown.csv` |
| Self-consistency (Sec. IV-D) | `jaccard.py` | `consistency.csv` |
| Human–judge and judge–judge agreement (Sec. IV-E) | `judge.py`, `judge_claude.py`, then `agreement.py` | `agreement.csv` |
| Verification policies replayed over the 360 verdicts (Table III) | `triage.py` | `triage.csv` |
| Intra-annotator test–retest kappa | `retest.py` | printed |

Scripts live in `src/` and outputs in `results/`; both are dropped from the table above
for width. Run any of them as `uv run python src/<name>.py` — paths are derived from the
script's own location, so the working directory does not matter.

**Which interval goes with which table.** Two files carry intervals for the same six
relations, computed by different methods, and they are not interchangeable:

- `results/factors.csv` — Wilson (binomial) intervals, which assume the 360 verdicts are
  independent draws.
- `results/precision_pooled_bootstrap.csv` — paper-level cluster bootstrap on the logit
  scale, resampling abstracts with replacement so that an abstract's triples stay
  together in every replicate.

**Table I quotes the bootstrap.** The 360 verdicts come from 155 distinct abstracts, so
the independence Wilson assumes does not hold; the cluster bootstrap is what the paper
reports, and for these six relations the two methods disagree at every endpoint.

## Reproducibility

Every randomized step is seeded: corpus and sample `seed=42`, cluster bootstrap
`seed=20260806` with B=10,000 replicates. The evaluation scripts are therefore
deterministic given the committed labels — re-running one rewrites its CSV with
identical contents, so a non-empty `git diff` after a re-run means something other than
the seed changed.

Model snapshots are pinned: `gpt-4.1-2025-04-14` and `gpt-4.1-mini-2025-04-14` for
extraction, through OpenAI Structured Outputs (strict `json_schema`). Both are
non-reasoning chat models, which is what makes the `temperature=0` accuracy run and the
`temperature=0.7` consistency sub-study possible.

The two judges are `gemini-2.5-flash` and `claude-haiku-4-5-20251001`, both at
`temperature=0` with thinking disabled, both seeing only the abstract and the triple,
and both reading the same rubric byte for byte — `src/judge_common.py` holds the rubric,
the prompt and the CSV loop, and each judge script is a thin provider adapter over it.
They are deliberately from different model families than the extractors, and from each
other, so that agreement between them cannot be attributed to one vendor. Gemini
publishes no dated snapshot, so the served `modelVersion` is recorded per row in the
output CSV.

## Running it

The environment is managed with [uv](https://docs.astral.sh/uv/); versions are pinned in
`pyproject.toml` and `uv.lock`.

```
uv run python src/precision.py      # or any other evaluation script
```

The evaluation scripts run offline from the committed data and need no keys. Re-running
extraction or judging does, in a `.env` file at the repository root:

```
OPENAI_API_KEY=sk-...          # extraction
GEMINI_API_KEY=...             # first judge
ANTHROPIC_API_KEY=sk-ant-...   # second judge
```

`src/load_neo4j.py` loads the extracted graph into a local Neo4j instance and produces
the `results/fig2*` panels. It is the only script that needs a database, it needs
`NEO4J_PASSWORD` in the same `.env`, and nothing in the paper's tables depends on it.

## Scope

This release supports the paper's measurements on one sample of computer-science
abstracts. Its human verdicts are a reference standard, not ground truth. The
verification design estimates support among extracted triples; it does not measure
recall or establish that either model tier is equivalent to the other. The policy
replay counts triples selected for review, not elapsed human review time, and its
relation choices were evaluated on the same labeled sample.

## License

Original code, documentation, annotations, and computed results in this repository
are released under the [MIT License](LICENSE). The arXiv abstracts are included for
reproducibility and remain subject to the rights of their original authors; this
license does not grant rights to third-party text.
