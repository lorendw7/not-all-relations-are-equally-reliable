"""Step 6 - load the extracted triples into Neo4j (source for Figure 2).

Builds the property graph from the temperature=0 accuracy run of the strong
extractor: every triple becomes two MERGEd nodes plus one typed relationship.
Nodes are keyed on the raw entity string (no entity linking), which is why the
resulting graph is a set of per-paper stars rather than a connected KG -- see
RESEARCH_PLAN 6.6. The run is idempotent: the graph is wiped first, so
re-running always reproduces the same graph.

Needs a running Neo4j Desktop instance and NEO4J_PASSWORD in .env.
"""

import json

from neo4j import GraphDatabase
from dotenv import load_dotenv
import os
from collect_corpus import DATASET_DIR

load_dotenv()                                   # NEO4J_PASSWORD from .env, never hard-coded
URI = "neo4j://127.0.0.1:7687"
AUTH = ("neo4j", os.getenv("NEO4J_PASSWORD"))
MODEL = "gpt-4.1-2025-04-14"                    # the strong tier's accuracy run


def node_ref(entity_type, name, arxiv_id):
    """Identity of one endpoint: (label, key property, key value).

    A "Paper" head is the literal placeholder the extractor emits, so it is keyed
    on this record's arxiv_id -- otherwise all 200 papers would collapse into one
    node. Every other entity is keyed on its name.
    """
    if entity_type == "Paper":
        return "Paper", "id", arxiv_id
    return entity_type, "name", name

def load_triple(session, arxiv_id, t):
    """MERGE one triple into the graph as (head)-[:RELATION]->(tail)."""
    hl, hk, hv = node_ref(t["head_type"], t["head"], arxiv_id)
    tl, tk, tv = t["tail_type"], "name", t["tail"]      # tails are always entities
    rel = t["relation"]
    # Labels and relationship types cannot be parameterised in Cypher, so they are
    # interpolated (safe: they come from the frozen schema, not user input) and
    # backtick-quoted; the actual values go through $params so Neo4j escapes them.
    # MERGE, not CREATE: a repeated entity reuses its node, which is what links the graph.
    query = (
        f"MERGE (h:`{hl}` {{`{hk}`: $hv}}) "
        f"MERGE (t:`{tl}` {{`{tk}`: $tv}}) "
        f"MERGE (h)-[:{rel}]->(t) "
    )
    session.run(query, hv=hv, tv=tv)

if __name__ == "__main__":
    path = DATASET_DIR / "extractions" / f"{MODEL}.jsonl"

    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")    # wipe first -> re-runs stay reproducible

            n = 0
            for line in open(path, encoding="utf-8"):
                rec = json.loads(line)
                for t in rec["triples"]:
                    load_triple(session, rec["arxiv_id"], t)
                    n += 1
            print(f"loaded {n} triples")