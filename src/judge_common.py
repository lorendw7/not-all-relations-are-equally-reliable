"""Step 4 - frozen material shared by every LLM judge.

Both judges (Gemini and Claude) see the same rubric, the same prompt template
and the same output schema, and write the same four CSV columns. Only the
model family differs -- that is the whole point of running a second judge, so
everything in this module is part of the experiment's fixed configuration.

Changing any of it silently invalidates the verdicts already collected. Verify
RUBRIC_SHA against the value recorded in docs/RESULTS.md before committing.

Nothing provider-specific belongs here: no SDK imports, no API keys, no model
identifiers. Those live in judge.py (Gemini) and judge_claude.py (Claude).
"""

import csv

import pandas as pd
from pathlib import Path
from pydantic import BaseModel
import hashlib


DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"


PATH = DATASET_DIR / "labels" / "sample_to_label.csv"

# Blindness is enforced here. sample_to_label.csv also carries the human
# verdict and error_type columns; no judge may ever see them, so every read of
# that file goes through load_blind_rows() rather than a bare read_csv().
BLIND_COLS = ["sample_id", "abstract", "relation", "head", "tail"]

# Both judges write these four columns, in this order, so that agreement.py
# can join either output against the human labels without special-casing.
FIELDS = ["sample_id", "supported", "reason", "model_version"]

# Frozen byte for byte. The Gemini judge produced all 360 verdicts under
# exactly this text; the Claude judge must see exactly the same one, or the
# comparison stops being a single-variable manipulation. RUBRIC_SHA (bottom of
# this file) is the tripwire.
RUBRIC = """You are a blind verifier for knowledge-graph triples extracted from the abstract \
of a computer-science paper. You are given ONLY the abstract and a single (head, relation, \
tail) triple. Decide whether the triple is supported by — or reasonably inferable from — the \
abstract.

Verdict:
- supported = true  — The triple is correct: the abstract supports the relation between head
  and tail. Minor paraphrase or abbreviation is acceptable.
- supported = false — The triple is wrong: a hallucination, an unsupported (spurious) claim, a
  wrong relation type, a severely distorted span, or inverted head and tail.

Per-relation question — apply the one matching the triple's relation:
- PROPOSES       — Does the paper introduce or present `tail` as its own contribution?
- USES_METHOD    — Does `head` apply or employ `tail` as a technique or tool?
- ADDRESSES_TASK — Does `head` target or tackle `tail` as a problem or task?
- EVALUATED_ON   — Is `head` benchmarked or tested on `tail` (a dataset or benchmark)?
- IMPROVES_OVER  — Does the abstract claim `head` outperforms `tail` (a baseline or prior work)?
- REPORTS_METRIC — Does the abstract cite a concrete numeric result for `tail` attributed to `head`?

Two strict policies:
- P1 — EVALUATED_ON requires an evaluation that actually happened. Future or planned phrasing
  ("we present … and planned experimental validation", "targeting benchmark performance on X")
  does NOT count, even when a benchmark is named → supported = false.
- P2 — REPORTS_METRIC requires a concrete number. A bare metric name with only a qualitative
  description ("only very modest detection accuracy", "high recall") and no numeric value does
  NOT count → supported = false.

Head-type convention: Judge the CONTENT of the relation, not the entity type of the head. When
the head is a named entity rather than the literal placeholder "Paper"
(e.g. (SL-S4Wave, PROPOSES, self-supervised learning framework)), judge whether the relation
content holds — here, "the paper proposes a self-supervised learning framework" is supported →
true. Do NOT mark a triple wrong merely because the head should structurally be "Paper". A wrong
head still yields false only when it breaks the content of the relation (e.g. inverted head/tail,
or a severely distorted span).

Output: set `supported` (true/false) and give a one-sentence `reason`."""

class Verdict(BaseModel):
    """The judge's output schema, enforced by both providers' structured outputs."""
    supported: bool
    reason: str


def load_blind_rows():
    """The 360 sampled triples, blind: abstract + triple only, no human labels."""
    df = pd.read_csv(PATH, usecols=BLIND_COLS, encoding="utf-8")
    return df


def build_contents(row):
    """Render one row into the user-turn text the judge is asked to verify.

    The rubric goes in the system slot (see judge.py / judge_claude.py); this
    is only the evidence.
    """
    abstract = row.abstract
    # NOTE: the line after the closing triple quote below is eight spaces, not
    # empty. It is an accident of the original f-string indentation, but it is
    # part of the prompt the Gemini judge actually saw for all 360 verdicts.
    # Do not "clean it up" -- an editor that strips trailing whitespace on save
    # will silently change the prompt.
    contents = f"""Abstract:
\"\"\"{abstract}\"\"\"
        
Triple to judge:
    head     = {row.head}
    relation = {row.relation}
    tail     = {row.tail}"""
    return contents

def load_done(out_path):
    """sample_ids already judged in out_path; empty set if it does not exist yet."""
    if out_path.exists():
        done = set(pd.read_csv(out_path, usecols=["sample_id"], encoding="utf-8")["sample_id"])
    else:
        done = set()
    return done


def run_judge(rows, judge_fn, out_path):
    """Judge every row not already in out_path, appending one CSV row each.

    judge_fn(contents) -> (Verdict, model_version) is the provider adapter:
    each judge script wraps its own SDK call behind that one signature, so this
    loop never learns which model it is talking to.

    Rows already present in out_path are skipped and each verdict is flushed as
    it is written, so an interrupted run is resumed by simply running it again.
    """
    done = load_done(out_path)
    new_file = not out_path.exists()
    count = len(done) + 1
    with open(out_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()

        for row in rows.itertuples(index=False):
            if row.sample_id in done:
                continue
            sid = row.sample_id
            contents = build_contents(row)

            print(f"start: {count}")
            # The adapter hands back a validated Verdict and whatever version
            # string its provider reports -- Gemini has no dated snapshot, so
            # the served version is recorded per row rather than assumed.
            v, version = judge_fn(contents)
            writer.writerow({
                "sample_id": sid,
                "supported": int(v.supported),
                "reason": v.reason,
                "model_version": version,
            })
            count += 1
            f.flush()

# Fingerprint of the frozen rubric. Print it before and after any refactor
# that moves this text: an editor reformat or a stray whitespace change is
# invisible in a diff, but not here. Expected: 2362 chars, 8ade652ac267.
RUBRIC_SHA = hashlib.sha256(RUBRIC.encode("utf-8")).hexdigest()[:12]

if __name__ == "__main__":
    print(len(RUBRIC), RUBRIC_SHA)