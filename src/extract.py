"""Step 3 - LLM triple extraction.

Extract (head, relation, tail) triples from an abstract using the OpenAI API
with Structured Outputs (strict json_schema -> guaranteed schema-valid JSON).

The schema, relation definitions, and annotation rules all come from
`schema.py`, so the extraction prompt and the JSON schema stay in sync with the
single source of truth. The extractor models are the two non-reasoning tiers
GPT-4.1 (strong) and GPT-4.1-mini (cheap), run at temperature=0 for the
reproducible accuracy run.

`run_model` extracts the whole frozen corpus with one model and appends results
to dataset/extractions/<model>.jsonl, one record per line with provenance
(model / temperature / seed). Runs are resumable: already-extracted abstracts
are skipped and per-abstract failures are logged and retried on the next run.

Run the full accuracy extraction (both model tiers):  uv run python src/extract.py
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
import schema
import json

from collect_corpus import DATASET_DIR

load_dotenv()          # load OPENAI_API_KEY (and others) from .env into the env
client = OpenAI(max_retries=5)      # reads OPENAI_API_KEY from env; auto-retries 429 / timeouts / 5xx with backoff

print("key loaded:", bool(os.getenv("OPENAI_API_KEY")))   # quick sanity check


def build_schema():
    """Build the strict JSON schema for one extraction call.

    The output is an object {"triples": [...]} (Structured Outputs requires the
    top level to be an object, not an array). Each triple is constrained to the
    entity/relation enums from schema.py. Strict mode also requires every field
    in "required" and additionalProperties=False at each level.

    Note: the global enums cannot enforce per-relation head/tail typing (e.g.
    "PROPOSES must be Paper->Method"); that is checked post-hoc against
    schema.RELATIONS.
    """
    triple = {
        "type": "object",
        "properties": {
            "head": {"type": "string"},
            "head_type": {"type": "string", "enum": schema.ENTITY_TYPES},    # entity-type enum
            "relation": {"type": "string", "enum": schema.RELATION_NAMES},   # relation-name enum
            "tail": {"type": "string"},
            "tail_type": {"type": "string", "enum": schema.ENTITY_TYPES},    # entity-type enum
        },
        "required": ["head", "head_type", "relation", "tail", "tail_type"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"triples": {"type": "array", "items": triple}},
        "required": ["triples"],
        "additionalProperties": False,
    }


def build_system_prompt():
    """Assemble the system prompt from schema.py (definitions + rules).

    Listing the relation definitions and the annotation rules from the same
    source the human verifier uses keeps the model and the rubric aligned.
    """
    lines = ["Extract (head, relation, tail) triples from the abstract.",
             "Allowed relations:"]
    for r in schema.RELATIONS:
        lines.append(f"- {r.name} ({r.head[0]} -> {r.tail[0]}): {r.definition} e.g. {r.example}")
    lines.append("Rules:")
    for rule in schema.ANNOTATION_RULES:
        lines.append(f"- {rule}")
    lines.append("Return ONLY triples grounded in the abstract; if none, return an empty list.")
    return "\n".join(lines)


def extract_one(abstract_text, model="gpt-4.1-mini-2025-04-14", temperature=0, seed=42):
    """Extract triples from one abstract; return a list of triple dicts.

    temperature=0 and a fixed seed keep the accuracy run reproducible. Returns
    an empty list if the model refuses (rare). An empty list is also a valid
    result for entity-sparse abstracts.
    """
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        seed=seed,
        messages=[
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": abstract_text},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "triple_extraction",
                "strict": True,                 # guarantee schema-valid JSON
                "schema": build_schema()
            },
        },
    )
    msg = resp.choices[0].message
    if msg.refusal:                             # model declined to answer
        return []
    return json.loads(msg.content)["triples"]   # content is a JSON string -> parse it

# One output file per model under dataset/extractions/ (the dataset/ dir is
# created in Step 1 by collect_corpus).
OUT_DIR = DATASET_DIR / "extractions"

def load_done(path):
    """Return the set of arxiv_ids already extracted in `path` (resume support).

    Each line of the output .jsonl is one completed record; collecting their ids
    lets run_model skip work already done, so an interrupted run can be restarted
    without re-spending API calls or writing duplicate rows.
    """
    done = set()
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                done.add(json.loads(line)["arxiv_id"])
    return done

def run_model(model, temperature=0, seed=42):
    """Extract triples for every abstract with one model; append results to disk.

    Writes dataset/extractions/<model>.jsonl, one self-describing record per line
    (triples + provenance: model / temperature / seed). The file is opened in
    append mode and the run is resumable: abstracts already present (see
    load_done) are skipped, so a crashed run continues where it left off.
    """
    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / f"{model}.jsonl"
    done = load_done(out_path)                       # ids already extracted -> skip

    with open(DATASET_DIR / "abstracts.jsonl", "r", encoding="utf-8") as f:
        records = [json.loads(l) for l in f]         # frozen corpus from Step 1

    with open(out_path, "a", encoding="utf-8") as out:   # "a" = append, never truncate
        for r in records:
            if r["arxiv_id"] in done:
                continue                             # resume: this abstract is done
            try:
                triples = extract_one(r["abstract"],
                                      model=model,
                                      temperature=temperature,
                                      seed=seed)
            except Exception as e:                   # failed even after SDK retries
                print(f"FAILED {r['arxiv_id']}: {e}")
                continue                             # not written -> retried on next run
            rec = {"arxiv_id": r["arxiv_id"],        # extraction + provenance
                   "model":model,
                   "temperature":temperature,
                   "seed":seed,
                   "triples": triples}
            out.write(json.dumps(rec) + "\n")
            out.flush()                              # flush per record -> crash-safe
            print(f"{r['arxiv_id']} -> {len(triples)} triples")

if __name__ == "__main__":
    # Accuracy run: both extractor tiers at temperature=0 (reproducible).
    models = ["gpt-4.1-mini-2025-04-14", "gpt-4.1-2025-04-14"]
    for m in models:
        run_model(m, temperature=0, seed=42)