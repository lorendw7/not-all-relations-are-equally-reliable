"""Step 4 - the Gemini judge (RQ1 cross-check).

Provider adapter for judge_common.run_judge: everything Gemini-specific --
client, model id, sampling config, retry policy, response unpacking -- lives
here; the rubric, prompt and CSV loop come from judge_common.
"""
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types, errors
from judge_common import (RUBRIC, Verdict, DATASET_DIR,
                          load_blind_rows, run_judge)

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

JUDGE_PATH = DATASET_DIR / "labels" / "judge_gemini.csv"
MODEL = "gemini-2.5-flash"

def judge_one(client, contents):
    """-> (Verdict, model_version). Retries 5xx and 429 with exponential backoff."""
    for attempt in range(6):
        try:
            # Frozen run configuration -- this block *is* the experiment.
            # temperature 0 and a zero thinking budget make the verdicts
            # reproducible; the rubric goes in the system slot, mirroring
            # judge_claude.py; response_schema forces schema-valid JSON.
            resp = client.models.generate_content(
                model=MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=RUBRIC,
                    temperature=0,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                    response_mime_type="application/json",
                    response_schema=Verdict
                )
            )
            # google-genai leaves .parsed as None on some responses; parse the
            # raw JSON rather than dropping the row. Dropped rows would not be
            # missing at random -- harder triples fail more often, which would
            # bias the per-relation comparison.
            v = resp.parsed or Verdict.model_validate_json(resp.text)
            # Gemini publishes no dated snapshot, so the served version is
            # recorded per row instead of assumed (see CLAUDE.md, Judge).
            return v, resp.model_version
        except errors.ServerError as e:
            wait = 2 ** attempt
            print(f" server busy ({e.code}), retry in {wait} seconds")
            time.sleep(wait)
        except errors.ClientError as e:
            wait = 2 ** attempt
            if e.code == 429:
                print(f" rate limited ({e.code}), retry in {wait} seconds")
                time.sleep(wait)
            else:
                # Only 429 is transient. A 400 or 401 is our bug; retrying it
                # six times just hides the real error behind the message below.
                raise
    raise RuntimeError("give up after retries")


def main():
    client = genai.Client()
    run_judge(load_blind_rows(), lambda c: judge_one(client, c), JUDGE_PATH)


if __name__ == "__main__":
    main()