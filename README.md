# Jev + LangChain: support-ticket triage agent

Use case: an agentic support-ticket triage system with two workflows
that share the same shape, differing only in *which engine classifies
the ticket*:

```
request -> [classifier] -> early exit? -> auto-file
                         -> otherwise   -> [resolver LLM + tools] -> reply
```

- **"Only LLM"** — a Gemini call classifies the ticket (department,
  urgency, refund intent) and can decide to auto-resolve it right
  there; otherwise the same LLM resolves it via tools.
- **"Jev + LLM"** — **Jev** (TypeSafe AI's "System One" model — fast,
  cheap, typed decisions instead of generated text) classifies the
  ticket instead, with the same early-exit authority; otherwise the
  same resolver LLM handles it.

Because the resolver stage is identical in both workflows, any latency
or cost difference between them comes entirely from the classifier —
see [Benchmark](#benchmark) below.

- `triage_agent.py` — both workflows. `jev_triage()` classifies via the
  official `langchain_typesafe` package (`TypeSafeClassifier` +
  `Noul`/`Score`/`Choice`); `llm_triage()` classifies via a single Gemini
  call with structured output, asked the same three questions on the
  same rubric. Both return the same signal shape, and the same
  `should_early_exit()` threshold decides the early exit either way.
  The shared resolver stage (`build_tools`, `SYSTEM_PROMPT`) then uses
  tools (`lookup_order`, `create_support_ticket`, `issue_refund`) to
  resolve tickets that didn't exit early. `issue_refund` is gated on
  the classifier's own refund-probability signal, so the classifier
  acts as a safety guardrail on the resolver LLM, not just a router.
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

## Why split the work this way

Jev is designed to be ~40-200x faster and far cheaper than a frontier
LLM for exactly this kind of structured decision. That advantage only
shows up when the classifier's signals let the pipeline *skip*
resolver LLM work, not just run alongside it — both workflows use
`should_early_exit()` to auto-resolve low-stakes tickets (low urgency,
no refund intent, confident routing) without ever calling the resolver
LLM, and only fall through to it for tickets that actually need
judgment. Keeping the resolver stage identical in both workflows
isolates the comparison to the one thing that actually differs: the
classifier engine.

## Benchmark

`benchmark.py` runs the same 10 sample tickets through both workflows
and reports real wall-clock latency and token-based cost for each:

```powershell
.\.venv\Scripts\python benchmark.py
```

Cost is computed from published pricing: Jev ($0.042 / 1M input
tokens, output free) and Gemini 2.5 Flash ($0.30 / 1M input tokens,
$2.50 / 1M output tokens) — the latter used for both the LLM classifier
and the resolver agent. A representative run over 10 tickets (5 of
which were low-stakes enough for both classifiers to auto-resolve):

| Workflow | Total latency | Avg/ticket | Total cost | Avg/ticket | Auto-resolved |
|---|---|---|---|---|---|
| `Only LLM` | 50518 ms | 5052 ms | $0.013413 | $0.001341 | 5/10 |
| `Jev + LLM` | 18779 ms | 1878 ms | $0.004942 | $0.000494 | 5/10 |
| **Delta** | **−62.8%** | | **−63.2%** | | |

Both classifiers agreed on which tickets were safe to auto-resolve, so
the gap isn't about accuracy — it's that a general-purpose LLM call
pays for its own reasoning/"thinking" tokens even on a simple
classification task (observed ~300+ output tokens per classification
call), while Jev returns a typed, calibrated answer with no generated
text at all. That gap compounds on every ticket that reaches the
resolver stage, and disappears on none.

This is a live network benchmark, not a controlled one: provider load,
network variance, and model non-determinism all affect single-run
numbers (one "Only LLM" resolver call took ~20s on its own), so expect
them to shift between runs. Run it a few times if you want a stable
average.
