# Jev + LangChain: support-ticket triage agent

Use case: an agentic support-ticket triage system with two workflows
that share the same shape, differing only in *which engine classifies
the ticket*:

```
request -> [classifier] -> early exit? -> auto-file
                         -> otherwise   -> [department router]
                                           -> billing agent
                                           -> technical agent
                                           -> account agent
                                           -> other agent
                                        -> reply
```

- **"Only LLM"** — a Gemini call classifies the ticket (department,
  urgency, refund intent) and can decide to auto-resolve it right
  there; otherwise it's routed to the matching department agent.
- **"Jev + LLM"** — **Jev** (TypeSafe AI's "System One" model — fast,
  cheap, typed decisions instead of generated text) classifies the
  ticket instead, with the same early-exit authority; otherwise it's
  routed to the same department agents.

Because the resolver stage (department agents, tools, thresholds) is
identical in both workflows, any latency or cost difference between
them comes entirely from the classifier — see [Benchmark](#benchmark)
below.

- `triage_agent.py` — both workflows.
  - `jev_triage()` classifies via the official `langchain_typesafe`
    package (`TypeSafeClassifier` + `Noul`/`Score`/`Choice`);
    `llm_triage()` classifies via a single Gemini call with structured
    output, asked the same three questions on the same rubric. Both
    return the same signal shape, and the same `should_early_exit()`
    threshold decides the early exit either way.
  - If a ticket doesn't exit early, its classified `department` routes
    it to one of four specialized LangChain ReAct agents
    (`DEPARTMENT_AGENTS`), each with its own role framing and toolset:
    - **billing** — `lookup_order`, `create_support_ticket`, `issue_refund`
      (gated on the classifier's refund-probability signal).
    - **technical** — `lookup_order`, `create_support_ticket`,
      `escalate_to_engineering`.
    - **account** — `create_support_ticket`, `send_password_reset_link`.
    - **other** — `create_support_ticket`.
- `main.py` — runs one sample ticket through the **Jev + LLM** workflow.
- `benchmark.py` — runs 10 sample tickets through **both** workflows
  and reports real latency and cost for each; see
  [Benchmark](#benchmark) below.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
copy .env.example .env   # fill in TYPESAFE_API_KEY and GOOGLE_API_KEY
.\.venv\Scripts\python main.py
```

## Benchmark

`benchmark.py` runs the same 10 sample tickets through both workflows
and reports real wall-clock latency and token-based cost for each:

```powershell
.\.venv\Scripts\python benchmark.py
```

Cost is computed from published pricing: Jev ($0.042 / 1M input
tokens, output free) and Gemini 2.5 Flash ($0.30 / 1M input tokens,
$2.50 / 1M output tokens) — the latter used for both the LLM classifier
and every department agent. A representative run over 10 tickets (5 of
which were low-stakes enough for both classifiers to auto-resolve):

| Workflow | Total latency | Avg/ticket | Total cost | Avg/ticket | Auto-resolved |
|---|---|---|---|---|---|
| `Only LLM` | 38797 ms | 3880 ms | $0.013844 | $0.001384 | 5/10 |
| `Jev + LLM` | 18496 ms | 1850 ms | $0.004613 | $0.000461 | 5/10 |
| **Delta** | **−52.3%** | | **−66.7%** | | |

Both classifiers agreed on which tickets were safe to auto-resolve, so
the gap isn't about accuracy — it's that a general-purpose LLM call
pays for its own reasoning/"thinking" tokens even on a simple
classification task (observed ~300+ output tokens per classification
call), while Jev returns a typed, calibrated answer with no generated
text at all.

This is a live network benchmark, not a controlled one: provider load,
network variance, and model non-determinism all affect single-run
numbers, so expect them to shift between runs. Run it a few times if
you want a stable average.
