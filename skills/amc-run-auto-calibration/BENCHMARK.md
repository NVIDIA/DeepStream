# Skill Benchmark: amc-run-auto-calibration

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `amc-run-auto-calibration`
- Evaluation date: 2026-10-16
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 14 evaluation tasks (14 positive)
- Dataset digest: `sha256:fb500370483c27111636c6067cdf0986c142194d119209b2fb4b69c7fa2f9015` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## What This Report Answers

The three-tier evaluation checks whether the skill:

- is safe to use;
- produces correct answers;
- is discovered and activated when needed;
- helps the agent complete the user's goal and expected workflow; and
- avoids wasted skill and tool usage.

## Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 90.7% — baseline ran, but no comparable score was available; uplift unavailable | 88.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 82.1% → 96.4% (+14.3 points) | 89.3% → 92.9% (+3.6 points) |
| Correctness | 38.6% → 100.0% (+61.4 points) | 58.6% → 95.7% (+37.1 points) |
| Discoverability | 82.9% — baseline ran, but no comparable score was available; uplift unavailable | 77.9% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 41.9% → 84.6% (+42.7 points) | 51.9% → 82.0% (+30.1 points) |
| Efficiency | 89.6% — baseline ran, but no comparable score was available; uplift unavailable | 91.7% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 3,300,339 | 3,399,629 | -99,290 | -2.92% | skill 14/14; base 14/14 |
| claude-code | 1 | 267,188 | 469,978 | -202,790 | -43.15% | skill 1/1; base 1/1 |
| claude-code | 10 | 109,833 | 183,372 | -73,539 | -40.10% | skill 1/1; base 1/1 |
| claude-code | 11 | 119,021 | 124,198 | -5,177 | -4.17% | skill 1/1; base 1/1 |
| claude-code | 12 | 267,782 | 217,169 | +50,613 | +23.31% | skill 1/1; base 1/1 |
| claude-code | 13 | 109,139 | 266,967 | -157,828 | -59.12% | skill 1/1; base 1/1 |
| claude-code | 14 | 468,901 | 289,450 | +179,451 | +62.00% | skill 1/1; base 1/1 |
| claude-code | 2 | 230,320 | 1,009,672 | -779,352 | -77.19% | skill 1/1; base 1/1 |
| claude-code | 3 | 271,873 | 153,342 | +118,531 | +77.30% | skill 1/1; base 1/1 |
| claude-code | 4 | 143,885 | 121,265 | +22,620 | +18.65% | skill 1/1; base 1/1 |
| claude-code | 5 | 189,002 | 152,375 | +36,627 | +24.04% | skill 1/1; base 1/1 |
| claude-code | 6 | 30,149 | 29,966 | +183 | +0.61% | skill 1/1; base 1/1 |
| claude-code | 7 | 189,493 | 30,128 | +159,365 | +528.96% | skill 1/1; base 1/1 |
| claude-code | 8 | 694,907 | 263,264 | +431,643 | +163.96% | skill 1/1; base 1/1 |
| claude-code | 9 | 208,846 | 88,483 | +120,363 | +136.03% | skill 1/1; base 1/1 |
| codex | All cases | 2,182,055 | 2,759,775 | -577,720 | -20.93% | skill 14/14; base 14/14 |
| codex | 1 | 147,386 | 61,162 | +86,224 | +140.98% | skill 1/1; base 1/1 |
| codex | 10 | 113,211 | 40,977 | +72,234 | +176.28% | skill 1/1; base 1/1 |
| codex | 11 | 112,568 | 45,975 | +66,593 | +144.85% | skill 1/1; base 1/1 |
| codex | 12 | 215,307 | 96,888 | +118,419 | +122.22% | skill 1/1; base 1/1 |
| codex | 13 | 176,270 | 115,506 | +60,764 | +52.61% | skill 1/1; base 1/1 |
| codex | 14 | 374,829 | 1,902,262 | -1,527,433 | -80.30% | skill 1/1; base 1/1 |
| codex | 2 | 67,715 | 83,431 | -15,716 | -18.84% | skill 1/1; base 1/1 |
| codex | 3 | 181,244 | 122,581 | +58,663 | +47.86% | skill 1/1; base 1/1 |
| codex | 4 | 255,595 | 25,089 | +230,506 | +918.75% | skill 1/1; base 1/1 |
| codex | 5 | 185,669 | 85,012 | +100,657 | +118.40% | skill 1/1; base 1/1 |
| codex | 6 | 13,748 | 13,411 | +337 | +2.51% | skill 1/1; base 1/1 |
| codex | 7 | 68,589 | 70,584 | -1,995 | -2.83% | skill 1/1; base 1/1 |
| codex | 8 | 161,427 | 68,796 | +92,631 | +134.65% | skill 1/1; base 1/1 |
| codex | 9 | 108,497 | 28,101 | +80,396 | +286.10% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 5,482,394 | 6,159,404 | -677,010 | -10.99% | skill 28/28; base 28/28 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 35 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 14 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: Instructions don't mention 'run_script' (`skills/amc-run-auto-calibration/SKILL.md`)
- **MEDIUM** QUALITY/quality_discoverability: Description contains vague words (`skills/amc-run-auto-calibration/SKILL.md`)
- **MEDIUM** QUALITY/quality_efficiency: Deeply nested references in rtsp.md (`skills/amc-run-auto-calibration/SKILL.md`)
- **MEDIUM** SECURITY/Unknown (LP3): MCP Least Privilege: The skill declares capabilities that span environment variable access, file reads/writes, network calls, and shell execu (`SKILL.md:1`)
- **MEDIUM** SECURITY/External Transmission (E1): Data Exfiltration: curl -fsS "$BASE_URL/ready"
curl -fsS "${VIOS_BASE_URL}/vst/api/v1/sensor/list"
```

If the AMC microservice is missing `VIOS_BASE_URL`, do not edit checked-in compose files. Relaunch the microservice (`references/rtsp.md:50`)
- 30 additional finding(s) are available in the full evaluation artifacts.

</details>

## Scoring Methodology

<details>
<summary>Show dimension definitions, source signals, and thresholds</summary>

| Dimension | Question | Scored signals |
|---|---|---|
| Security | Is it safe to use? | `security` (100%) |
| Correctness | Is the answer correct? | `accuracy` (100%) |
| Discoverability | Was the right skill loaded when needed? | `skill_execution` (100%) |
| Effectiveness | Did the skill help complete the task? | `goal_accuracy` (50%) + `behavior_check` (50%) |
| Efficiency | Did it avoid wasted tool calls and token usage? | `skill_efficiency` (50%) + `token_efficiency` (50%) |

- Dimension bands: PASS at 50% or above; NEUTRAL from 40% to below 50%; FAIL below 40%.
- Overall Tier 3 lift: PASS at +5 points or more; FAIL at -10 points or less; values between those bands are NEUTRAL.
- Overall verdict: PASS only when every configured dimension passes for at least one supported agent. Lift is reported as diagnostic evidence and does not override this gate.
- The 50% attempt pass threshold is a separate per-task gate; it is not the dimension pass threshold.
- Effectiveness is the equal-weight mean of goal completion (`goal_accuracy`) and expected workflow adherence (`behavior_check`).
- Efficiency is 50% tool-call productivity (the backward-compatible `skill_efficiency` wire id) and 50% `token_efficiency`. Positive-case skill routing is scored under Discoverability, not Efficiency; a negative case without a routing target is N/A. N/A sources are omitted, remaining weights are renormalized, and the dimension is marked partial.

Signals present in this run:

- `security` (Security): unsafe operations, secret leakage, and unauthorized access.
- `skill_execution` (Skill Execution): whether the expected skill was selected, decoys were avoided, and the workflow executed.
- `skill_efficiency` (Tool Productivity): tool-call productivity (legacy wire id; routing is scored under Discoverability).
- `accuracy` (Accuracy): final-answer correctness against the reference answer.
- `goal_accuracy` (Goal Accuracy): whether the user's goal was achieved.
- `behavior_check` (Behavior Check): whether the expected workflow behavior was followed.
- `token_efficiency` (Token Efficiency): actual uncached prompt plus completion usage (50% of Efficiency).

</details>

## Freshness

Regenerate this benchmark when the skill, evaluation dataset, target agent/model, evaluator version, environment, or scoring policy changes.
