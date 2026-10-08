# PromptGate

Prompt eval and regression gate. Every pull request that changes a prompt is scored against a 50-case golden dataset, and the merge is blocked if the pass rate drops more than 5 percentage points.

Test use case: a log-error classifier. One log line goes in, and JSON with `category` and `severity` comes out. It runs on `openai/gpt-oss-20b` through Groq's free tier.

## Proof it works

The current prompt (v2), re-run against its baseline, passes ([run log](https://github.com/aryanchoudharitech-ai/promptGate/actions/runs/37613037500)):

```
vs baseline: +0.0 points
GATE: PASS (+0.0 points, limit -5.0)
```

A well-meant "when in doubt, escalate" prompt (v3) hard-codes keyword rules for severity. In a 10-case pre-flight every format check still passed and only severity accuracy dropped (4/10), so JSON, schema and label checks alone would not have caught it. The correctness rule did, and the gate blocked the pull request ([blocked PR](https://github.com/aryanchoudharitech-ai/promptGate/pull/1), [run log](https://github.com/aryanchoudharitech-ai/promptGate/actions/runs/37616993593)):

```
vs baseline: -38.0 points
  regressed (23): a05, a07, a08, a10, a12, e03, e05, e07, ...
  fixed (4): a03, e04, n23, n24
GATE: FAIL (-38.0 points on both attempts, limit -5.0)
Error: Process completed with exit code 1.
```

Log excerpts are pasted here because GitHub Actions logs expire and require sign-in to view.

## Results

| Prompt | Pass rate | vs baseline | Gate |
| ------ | --------- | ----------- | ---- |
| v1     | 64.0%     | -12.0       | fail |
| v2     | 76.0%     | baseline    | pass |
| v3     | 38.0%     | -38.0       | fail |

## How it works

```
prompts/vN.txt -> run 50 golden cases -> rule scorers (JSON, schema, labels, exact match)
                         |                        |
                  SQLite run history      pass rate vs data/baseline.json
                                                  |
                                  drop > 5 points on both attempts?
                                                  |
                          exit 0 pass / 1 regression / 2 could not evaluate
```

Gate rules:

- A candidate fails only if it scores more than 5 percentage points below the committed baseline on two attempts in a row. Run-to-run noise is about 2 cases, so one bad run alone never fails the build.
- Unit tests run first, with no API calls and no secrets.
- The API key lives in a GitHub Secret and is visible only to the gate step.
- The gate refuses to run if the golden dataset or the tested model changed since the baseline was recorded.
- The baseline is a committed file, so changing it shows up in code review.

## Run it locally

```
pip install -r requirements.txt     # then put GROQ_API_KEY in a .env file
python -m src.run_eval v2           # run the dataset against a prompt version
python -m src.db                    # run history
python -m src.gate check v2         # gate a prompt against the baseline
python -m pytest                    # unit tests, no API calls
```

## Limitations

- 50 cases means one case is 2 points. Single runs are noisy, so the gate retries once before failing.
- Changes smaller than about 3 cases are invisible to a 50-case gate. A mild first draft of v3 scored higher than v2 on a 10-case sample, so 10-case runs are smoke checks only.
- The baseline is one run, not an average.
- Labels are exact-match. Severity is subjective, and some labels are debatable.
- The LLM judge agreed with my hand labels on 8 of 10 cases. It is advisory and never gates.
- Free-tier model quotas limit how often the eval can run.

## Layout

```
src/        provider, prompt loader, dataset, scorers, runner, db, gate
prompts/    versioned prompts (v1, v2, ...) and the judge rubric
data/       golden dataset, calibration set, baseline.json
tests/      pytest suite
.github/    CI workflow
```
