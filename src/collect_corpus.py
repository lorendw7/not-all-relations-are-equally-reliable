"""Step 1 - Corpus collection.

Fetch CS paper abstracts from the public arXiv API and freeze a reproducible
sample to `dataset/abstracts.jsonl`.

Design choices that make the corpus reproducible:
  - The query (categories + submittedDate window) is hard-coded and frozen.
  - Results are sorted by submittedDate/ascending, so paging is deterministic.
  - The 200-paper sample is drawn with a fixed random seed.
  - All provenance (query, window, seed, counts) is written to
    `dataset/corpus_meta.json`.

Output paths are anchored to this file's location, so the script can be run
from any working directory:  uv run python src/collect_corpus.py
"""

import urllib.parse
import feedparser
import time
import re
import json
import random
from pathlib import Path

# Output directory, resolved relative to this file (project_root/dataset),
# so the script works regardless of the current working directory.
DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"

# arXiv API endpoint (returns Atom XML, parsed below with feedparser).
BASE = "http://export.arxiv.org/api/query"

# Frozen query: papers in cs.AI / cs.CL / cs.LG submitted in the
# 2026-06-15 . 2026-06-21 (UTC) window. submittedDate format is YYYYMMDDhhmm.
SEARCH = "(cat:cs.AI OR cat:cs.CL OR cat:cs.LG) AND submittedDate:[202606150000 TO 202606212359]"


def build_url(start, max_results):
    """Build one paged, URL-encoded arXiv API request URL.

    sortBy/sortOrder are fixed so the same query always returns the same
    ordering (required for reproducible paging and sampling).
    """
    params = {
        "search_query": SEARCH,
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "ascending",
    }
    return f"{BASE}?{urllib.parse.urlencode(params)}"


def fetch(url):
    """Fetch and parse one arXiv API page, with a polite research User-Agent."""
    return feedparser.parse(url, agent="llm-kg-reliability/0.1 " +
                                       "(research; contact anonymized for review)")


def fetch_retry(url, tries=4):
    """Fetch a page, retrying on empty responses with linear backoff.

    The arXiv API occasionally returns an empty result set transiently, so an
    empty page is retried (3s, 6s, 9s) before giving up.
    """
    for i in range(tries):
        d = fetch(url)
        if d.entries:
            return d
        time.sleep(3 * (i + 1))
    return d


def parse_id(entry_id):
    """Split an arXiv id URL into (versioned id, base id).

    "http://arxiv.org/abs/2606.16072v2" -> ("2606.16072v2", "2606.16072").
    The base id (version stripped) is used for de-duplication.
    """
    raw = entry_id.rsplit("/", 1)[-1]
    base = re.sub(r"v\d+$", "", raw)
    return raw, base


def to_record(entry):
    """Convert one feedparser entry into a flat record dict.

    Note: entry.tags and entry.authors are lists of dicts, so we pull out the
    "term"/"name" fields. The abstract is kept as-is (raw text, not cleaned).
    """
    arxiv_id, base = parse_id(entry.id)
    return {
        "arxiv_id": arxiv_id,
        "base_id": base,
        "primary_category": entry.arxiv_primary_category["term"],
        "categories": [t["term"] for t in entry.tags],
        "title": " ".join(entry.title.split()),
        "abstract": entry.summary,                       # raw text, kept unmodified
        "authors": [a["name"] for a in entry.authors],
        "submitted": entry.published,
        "updated": entry.updated,
    }


def main():
    # Keep only papers whose PRIMARY category is one of the three target classes
    # (a paper merely cross-listed in cs.AI is dropped).
    KEEP = {"cs.AI", "cs.CL", "cs.LG"}
    PAGE = 200

    seen = set()      # base ids already collected (cross-listing de-dup)
    records = []
    start = 0

    # Page through the whole window until an empty page signals the end.
    while True:
        d = fetch_retry(build_url(start, PAGE))
        if not d.entries:
            break

        for e in d.entries:
            rec = to_record(e)
            if rec["primary_category"] not in KEEP:   # drop off-topic primary
                continue
            if rec["base_id"] in seen:                # drop duplicates
                continue
            seen.add(rec["base_id"])
            records.append(rec)

        start += PAGE
        time.sleep(3)   # arXiv rate limit: >= 3s between requests

    # Need at least 200 to sample from; fail loudly otherwise.
    if len(records) < 200:
        raise SystemExit(f"Found {len(records)} entries, Not Enough")

    # Reproducible random sample: fixed seed -> same 200 papers every run.
    SEED = 42
    random.seed(SEED)
    sample = random.sample(records, 200)
    sample.sort(key=lambda r: r["submitted"])   # order by submission time (readability)

    # Write the frozen corpus as JSON Lines (one JSON object per line).
    with open(DATASET_DIR / "abstracts.jsonl", "w", encoding="utf-8") as f:
        for r in sample:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Provenance: everything needed to reproduce this exact sample.
    meta = {
        "query": SEARCH,
        "date_window_utc": ["202606150000", "202606212359"],
        "sort": "submittedDate/ascending",
        "primary_filter": sorted(KEEP),
        "sample_seed": SEED,
        "n_in_window": len(records),   # papers matching the filter, before sampling
        "n_sampled": len(sample),      # == 200
        "fetched_on": "2026-06-23",
    }

    with open(DATASET_DIR / "corpus_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=4)

    print(f"n_in_window={len(records)}  n_sampled={len(sample)}  -> dataset/abstracts.jsonl")


if __name__ == "__main__":
    main()
