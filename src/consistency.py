# src/consistency.py — RQ2 self-consistency extraction.
#
# Re-extracts the first N abstracts K times each at temperature=0.7, every run
# with a different seed, so Step 5 can measure output stability (label-free
# Jaccard across runs). Reuses extract.extract_one unchanged; output is kept
# separate from the temp=0 accuracy run, under dataset/extractions/consistency/.
import json
from extract import extract_one              # reuse the temp=0 extractor, unchanged
from collect_corpus import DATASET_DIR

K = 3                                          # runs per abstract
N = 40                                         # subset size (first N of the frozen corpus)
SEEDS = [42, 43, 44]                           # one distinct seed per run -> independent, reproducible samples
CONSIST_DIR = DATASET_DIR / "extractions" / "consistency"

def load_done_runs(path):
    """Return the set of (arxiv_id, run) pairs already extracted in `path`.

    Each abstract yields K lines here, so resume must key on the (abstract, run)
    pair -- keying on arxiv_id alone would skip runs 1..K-1 after the first.
    """
    done_run = set()
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                done_run.add((rec["arxiv_id"], rec["run"]))
    return done_run

def run_consistency(model, temperature=0.7):
    """Extract the first N abstracts K times each; append results to disk.

    Writes dataset/extractions/consistency/<model>.jsonl, one record per
    (abstract, run) with provenance (model / temperature / seed / run). Append
    mode + resume make the run restartable.
    """
    CONSIST_DIR.mkdir(parents=True, exist_ok=True)   # consistency/ is a new subdir
    out_path = CONSIST_DIR / f"{model}.jsonl"
    done = load_done_runs(out_path)                  # (arxiv_id, run) pairs to skip

    with open(DATASET_DIR / "abstracts.jsonl", "r", encoding="utf-8") as f:
        records = [json.loads(l) for l in f][:N]    # first N of the frozen corpus

    with open(out_path, "a", encoding="utf-8") as out:   # "a" = append, never truncate
        for r in records:
            for k in range(K):                           # K runs of the same abstract
                if (r["arxiv_id"], k) in done:
                    continue                             # resume: this (abstract, run) is done
                try:
                    triples = extract_one(r["abstract"], model=model, temperature=temperature,
                                          seed=SEEDS[k])  # distinct seed per run
                except Exception as e:
                    print(f"FAILED {r['arxiv_id']} run{k}: {e}")
                    continue                             # not written -> retried on next run

                rec = {"arxiv_id": r["arxiv_id"],  # extraction + provenance
                       "model": model,
                       "temperature": temperature,
                       "seed": SEEDS[k],
                       "triples": triples,
                       "run": k}

                print(f"run{k}: {r['arxiv_id']} -> {len(triples)} triples")
                out.write(json.dumps(rec) + "\n")
                out.flush()                              # flush per record -> crash-safe

if __name__ == "__main__":
    for m in ["gpt-4.1-mini-2025-04-14", "gpt-4.1-2025-04-14"]:
        run_consistency(m)
