# Workshop-Volume Sweep

This file is the evidence behind one sentence in the paper:

> We are not aware of an evaluation that reports verification-first precision — judged against the
> source text, with no gold set — per individual relation type for end-to-end LLM triple
> extraction.

That sentence is a bound on what we surveyed, not a claim about the world. A universal negative
("no prior work does X") is refuted by a single paper we failed to find, so what follows is the
list we walked, the criterion we applied, and the papers we read in full and cleared. A reader who
knows of a counterexample should be able to check it against this list directly.

## Screening criterion

A paper would have counted as a hit if it reported **all four** of the following together:

1. **Precision, or an equivalent correctness rate**, for LLM-extracted triples;
2. **Per individual relation type** — not per corpus, per graph, per ontology, per template, or
   per task category;
3. **Judged against the source text**, rather than matched against a pre-existing gold set; and
4. For **end-to-end extraction**, in which the model chooses the entities as well as the relation
   — not classification of a given entity pair over a closed relation set.

Each of the four is load-bearing. Dropping (3) admits the per-class F1 that supervised relation
extraction reports as a matter of course; dropping (4) admits relation-classification benchmarks.
A paper meeting three of four was read at full text before being cleared.

## Method

Every published volume of the two most on-topic workshop series was walked title by title, plus
the ESWC 2026 knowledge-graph-quality workshop, which was treated as the highest-risk venue
because its entire subject is KG quality. Titles that could plausibly meet three of the four
criteria were opened; those that still looked plausible after the abstract were read at full text.

Sweep performed **2026-07-29**. 69 papers across seven volumes. Three read at full text.

## Volumes swept

| volume | workshop | papers | outcome |
|---|---|---|---|
| [Vol-3184](https://ceur-ws.org/Vol-3184/) | Text2KG 2022 @ ESWC | 10 | Pre-LLM era; OpenIE and pipeline papers, no evaluation study. |
| [Vol-3447](https://ceur-ws.org/Vol-3447/) | Text2KG 2023 @ ESWC | 13 | Closest is Khorashadizadeh et al. on in-context KG generation — a capability study, not a reliability one. |
| [Vol-3747](https://ceur-ws.org/Vol-3747/) | Text2KG 2024 @ ESWC | 13 | Two candidates read at full text, both cleared — see below. |
| [Vol-4020](https://ceur-ws.org/Vol-4020/) | Text2KG 2025 @ ESWC | 15 | Construction and RAG systems; the one judging paper targets SPARQL query correctness, a different task. |
| [Vol-3967](https://ceur-ws.org/Vol-3967/) | ELMKE 2024 @ EKAW | 4 | Ontology learning; no triple-level evaluation. |
| [Vol-3977](https://ceur-ws.org/Vol-3977/) | ELMKE 2025 @ ESWC | 4 | One candidate read at full text, cleared — see below. |
| [Vol-4205](https://ceur-ws.org/Vol-4205/) | QKG 2026 @ ESWC | 10 | KG *quality* workshop, so the highest-risk volume. All papers concern SHACL, FAIR compliance, or KG-level quality dimensions; none concerns LLM extraction reliability. |
| | **total** | **69** | |

## Candidates read at full text

Each was verified against the full PDF rather than the title or abstract.

**Bertolini et al., "On Constructing Biomedical Text-to-Graph Systems with LLMs"** (Text2KG 2024).
A large-scale comparison of architectures and of fine-tuning against in-context learning on
biomedical text-to-graph benchmarks. Automatic benchmark metrics only: the full text contains no
per-relation breakdown and no human verification of any kind. Diverges on criteria (2) and (3).

**Boylan et al., "KGValidator"** (Text2KG 2024). Automatic validation of KG construction, with no
human anchor at any point. Diverges on criterion (3). Cited in the paper as a point of comparison
for automatic-only validation.

**Hannah et al., "Large Language Models as Knowledge Evaluation Agents"** (ELMKE 2025). The
closest paper found. It uses Gemini as a proxy for domain experts and compares its assessments
against theirs — the same judge family and broadly the same question we ask. But the objects
judged are *relationship labels* derived from an XML schema in ontology engineering, not triples
extracted from running text, and results are reported per label rather than per relation type.
Diverges on criteria (2) and (4). It does not refute the claim; it is independent evidence from
the Semantic Web community that an LLM proxy evaluator is promising but unreliable, which is
consistent with our own judge result.

## Residual risk

One item, stated rather than hidden. The Text2KG 2026 edition (5th, ESWC 2026) had **no CEUR
volume published** as of the sweep date. It is the single venue where a competing result could
still surface, and it cannot be checked for this submission. It will be re-checked before the
camera-ready.

## What this does and does not establish

It establishes that a deliberate, dated, enumerable search of the venues where such a paper would
most likely appear did not find one, and it lets a reader audit that search rather than take it on
trust.

It does not establish that no such paper exists. The sweep covers two workshop series and one
quality workshop; it does not cover the main Semantic Web conference tracks, the NLP venues, or
the preprint literature exhaustively. The paper's claim is worded to match that scope.
