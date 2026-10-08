# Agent Evaluation Lab

A small, hands-on lab for evaluating an LLM agent the way you would in production: a
LangGraph agent for an engineering platform, a golden dataset, deterministic and
LLM-as-a-judge evaluators, Langfuse tracing and experiments, and a GitHub Actions gate
that blocks pull requests when quality regresses.

## Contents

- [What the agent does](#what-the-agent-does)
- [Repository layout](#repository-layout)
- [Setup](#setup)
- [Evaluation layers](#evaluation-layers)
- [Running things](#running-things)
- [CI gate](#ci-gate)
- [Failure simulation](#failure-simulation)
- [Common tasks](#common-tasks)

## What the agent does

A **supervisor** reads each user request and routes it to exactly one destination,
returning a structured `RoutingDecision` (`route` + `reason`):

| Route | Used for |
|---|---|
| `cad_agent` | Creating or modifying geometry, dimensions, parameters, CAD models |
| `simulation_agent` | Running or analysing simulations, e.g. stress analysis |
| `plm_agent` | Revisions, lifecycle state, version history |
| `clarify` | Not enough information to decide |

The full agent is a LangGraph state machine (`app/graph.py`):

```
START → supervisor ─┬─ cad_agent ────────→ END
                    ├─ simulation_agent ─→ END
                    ├─ clarify ──────────→ END
                    └─ plm_agent ⇄ plm_tools
                           └─(no tool calls)→ END
```

- `cad_agent`, `simulation_agent` and `clarify` are stubs that return a fixed response.
- `plm_agent` is a real tool-calling agent. It calls `get_part_metadata`
  (`app/tools.py`), which returns fake PLM data (revision `C`, state `Released`,
  version `3.2`) in place of a real PLM system.

All model calls use OpenAI `gpt-4.1-mini` at temperature 0.

## Repository layout

```
app/
  models.py         RoutingDecision and AnswerEvaluation (Pydantic structured outputs)
  supervisor.py     route_request(): the supervisor on its own, used by the routing evals
  graph.py          The full LangGraph agent (supervisor + agents + PLM tool loop)
  state.py          AgentState shared by the graph's nodes
  tools.py          get_part_metadata tool (fake PLM backend)

evals/
  dataset.json              Routing golden dataset (input → expected_route, by slice)
  config.py                 Routing accuracy thresholds (overall and per slice)
  evaluators.py             Deterministic checks (route, tools, tool args, tool-call cap)
  run_evals.py              Routing eval + threshold gate (runs in CI)
  graph_dataset.json        End-to-end graph cases with behavioural constraints
  run_graph_evals.py        Offline graph runner with deterministic checks
  run_langfuse_experiment.py  Langfuse experiment: graph + six scores + gate (runs in CI)
  simulate_failure.py       Runs the graph with corrupted tool data (revision D)
  llm_judge.py              Rubric judge for correctness, relevance and groundedness (1–5)
  test_judge.py             Demo of llm_judge on a deliberately wrong answer

scripts/
  upload_graph_dataset.py   Syncs graph_dataset.json to the Langfuse dataset
  test_graph.py             Runs one PLM question through the graph, traced to Langfuse

.github/workflows/evals.yml  CI: runs both gated evals on every PR to main
```

## Setup

Requirements: Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Create a `.env` file in the repo root (it is git-ignored):

```bash
OPENAI_API_KEY=sk-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

The Langfuse experiment also expects a dataset named `agent-evaluation-lab` to exist in
your Langfuse project. Create it once in the Langfuse UI, then upload the cases (see
[Common tasks](#common-tasks)).

## Evaluation layers

The repo evaluates the agent at two levels.

### 1. Routing evals: supervisor only

`evals/run_evals.py` sends each case in `evals/dataset.json` to `route_request()` and
compares the returned route with `expected_route` using an exact string match.

Each case looks like this:

```json
{ "id": "cad_001", "input": "Change the thickness of bracket A to 3 mm.",
  "expected_route": "cad_agent", "slice": "cad" }
```

It reports overall accuracy and accuracy for each slice (`cad`, `simulation`, `plm`,
`ambiguous`). Thresholds are set in `evals/config.py`:

```python
MIN_OVERALL_ACCURACY = 0.90
MIN_SLICE_ACCURACY = {"cad": 0.80, "simulation": 0.80, "plm": 0.80, "ambiguous": 0.80}
```

If any threshold is missed, the script exits with code `1`.

> The dataset is small (7 cases), so a single wrong route fails the 90% overall
> threshold.

### 2. Graph evals: the full agent, end to end

These run the whole LangGraph agent and check what it actually did. Cases in
`evals/graph_dataset.json` list **behavioural constraints** instead of an exact tool
sequence:

```json
{
  "id": "plm_e2e_001",
  "slice": "plm",
  "input": "What is the current revision of part B-104?",
  "expected_route": "plm_agent",
  "constraints": {
    "required_tools": ["get_part_metadata"],
    "forbidden_tools": [],
    "required_tool_args": { "get_part_metadata": { "part_id": "B-104" } },
    "max_tool_calls": 2,
    "answer_must_mention": ["B-104", "C"]
  },
  "reference_answer": "Part B-104 is currently at revision C."
}
```

The Langfuse experiment (`evals/run_langfuse_experiment.py`) gives each case **six
scores**, each 1 (pass) or 0 (fail):

| Score | Type | Checks |
|---|---|---|
| `route` | deterministic | The supervisor picked `expected_route` |
| `required_tools` | deterministic | Every tool in `required_tools` was called |
| `required_tool_args` | deterministic | At least one call to each tool used the required arguments (extra arguments are allowed) |
| `max_tool_calls` | deterministic | Total tool calls ≤ `max_tool_calls` |
| `answer_correctness` | LLM judge | The final answer states the same facts as `reference_answer`; different wording is fine |
| `groundedness` | LLM judge | Every factual claim in the answer is supported by the tool results from **the same run** (it never sees `reference_answer`) |

`forbidden_tools` and `answer_must_mention` are in the dataset but are not scored yet.

The two judges answer different questions:

- **`answer_correctness`**: is the answer right?
- **`groundedness`**: did the agent stay faithful to its evidence?

Reading them together tells you where a failure came from:

| answer_correctness | groundedness | Meaning |
|---|---|---|
| 1 | 1 | Correct and faithful |
| 0 | 1 | Agent reported its data faithfully, but the **data was wrong** |
| 0 | 0 | Agent **made something up** |

The experiment fails (exit code `1`) if any dataset item is missing from the results,
or if any of the six scores is missing or not 1. This check matters because the
Langfuse SDK only logs failed tasks and evaluators and leaves them out of the results,
which would otherwise let an empty run pass.

## Running things

```bash
# Routing evals with the threshold gate (supervisor calls are traced to Langfuse)
uv run python -m evals.run_evals

# Graph evals offline: reads graph_dataset.json, prints route, tool calls,
# answer and the four deterministic checks
uv run python -m evals.run_graph_evals

# Langfuse experiment: reads the Langfuse dataset, gives six scores, gates on all of them
uv run python -m evals.run_langfuse_experiment

# Single PLM question through the graph, traced to Langfuse
uv run python -m scripts.test_graph
```

Every graph run is traced to Langfuse. Experiment runs appear under
**Datasets → agent-evaluation-lab → Runs**, where you can compare runs side by side and
open each item's trace (supervisor → PLM agent → tool → answer) with its scores attached.

## CI gate

`.github/workflows/evals.yml` runs on every pull request to `main` (and manually via
`workflow_dispatch`). The `run-evaluations` job:

1. Installs dependencies with `uv sync`
2. Runs `evals.run_evals` (routing thresholds)
3. Runs `evals.run_langfuse_experiment` (all six scores must be 1)

The **Protect main** ruleset requires `run-evaluations` to pass, so a PR that regresses
either layer cannot be merged.

The job needs these repository secrets (**Settings → Secrets and variables → Actions**):
`OPENAI_API_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL`.

Things to know:

- **CI reads the Langfuse dataset, not `graph_dataset.json`.** Edits to the JSON reach
  CI only after you re-run the upload script.
- **Each CI run creates a new dataset run in Langfuse.**
- **LLM judges can vary between runs.** `answer_correctness` and `groundedness` can
  occasionally disagree with themselves. Re-run the job before investigating a failure
  that doesn't reproduce.

## Failure simulation

`evals/simulate_failure.py` shows a realistic failure: the PLM backend returns bad data.
For one run, it temporarily patches `get_part_metadata` so it returns revision `D`
instead of `C`. The graph and the tool code are not modified.

```bash
uv run python -m evals.simulate_failure
# Final answer: The current revision of part B-104 is revision D.
```

The trace is sent to Langfuse with the name `plm-simulated-failure` and the tag
`failure-mining-demo`.

When this scenario was run through the experiment, these were the scores:

| Scores | Result |
|---|---|
| `route`, `required_tools`, `required_tool_args`, `max_tool_calls` | 1: routing and tool use were correct |
| `answer_correctness` | **0**: says D, the reference says C |
| `groundedness` | 1: the answer matches the (bad) tool evidence |

The deterministic checks alone would have let this through. `answer_correctness`
catches it, and `groundedness` points to the data source rather than the model.

## Common tasks

**Add a routing case.** Append to `evals/dataset.json` with a `slice` that has a
threshold in `evals/config.py`. A slice without a threshold is not gated.

**Add a graph case**

1. Append the case to `evals/graph_dataset.json`.
2. Run `uv run python -m evals.run_graph_evals` to see how the agent behaves.
3. Sync it to Langfuse with `uv run python scripts/upload_graph_dataset.py`.
   - Items are upserted using the ID `agent-evaluation-lab-<case id>`, so re-running
     updates the existing item instead of duplicating it.
   - Removing a case from the JSON does not delete it from Langfuse; archive it there
     instead.

> `groundedness` scores 0 when a run produced no tool results, because the current
> dataset is PLM-only. Revisit this before adding cases for routes that don't call
> tools.

**Change the supervisor prompt.** The prompt currently lives in two places:
`app/supervisor.py` (used by the routing evals) and `app/graph.py` (used by the graph
and the experiment). Update both, or a regression in one may not be caught by the other
eval.
