"""Step 4 - the Claude judge (second judge, robustness check on RQ1).

Provider adapter for judge_common.run_judge. Claude is a third model family,
distinct from both the extractors (OpenAI) and the first judge (Google), so a
shared over-acceptance pattern can be attributed to LLM judges in general
rather than to Gemini in particular.

claude-haiku-4-5 is the tier match for gemini-2.5-flash, and the only current
Claude model that still accepts a temperature and runs without thinking -- the
same constraint that excludes reasoning models from the extractor pair.
"""
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from judge_common import (RUBRIC, Verdict, DATASET_DIR,
                          load_blind_rows, run_judge)

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

JUDGE_PATH = DATASET_DIR / "labels" / "judge_claude.csv"
MODEL = "claude-haiku-4-5-20251001"


def judge_one(client, contents):
    """-> (Verdict, model_version).

    Retries are the SDK's job here: anthropic.Anthropic(max_retries=...) backs
    off on 429, 5xx and timeouts, so this adapter has no retry loop of its own.
    """
    # Frozen run configuration -- this block *is* the experiment, and it mirrors
    # judge.py parameter for parameter. temperature 0 for reproducibility; no
    # thinking parameter at all, because Haiku 4.5 does not think unless asked;
    # the rubric in the system slot, matching Gemini's system_instruction; and
    # output_format to force a schema-valid Verdict instead of free text.
    resp = client.messages.parse(
        model=MODEL,
        max_tokens=1024,
        temperature=0,
        system=RUBRIC,
        messages=[{"role": "user", "content": contents}],
        output_format=Verdict,
    )

    # A refusal or a truncated response is not guaranteed to match the schema,
    # and the rows that trigger them are not a random subset -- sensitive
    # wording refuses more often. Failing loudly keeps the stratified sample
    # intact; resume makes the cost of stopping close to zero.
    if resp.stop_reason != "end_turn":
        raise RuntimeError(f"unexpected stop reason {resp.stop_reason!r}")

    # Unlike Gemini, Anthropic serves a dated snapshot, so resp.model just
    # echoes MODEL back. It is still recorded per row to keep this CSV
    # structurally identical to judge_gemini.csv.
    return resp.parsed_output, resp.model


def main():
    client = anthropic.Anthropic(max_retries=8)
    run_judge(load_blind_rows(), lambda c: judge_one(client, c), JUDGE_PATH)


if __name__ == "__main__":
    main()