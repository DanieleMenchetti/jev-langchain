# Jev + LangChain: support-ticket triage agent

Use case: an agentic support-ticket triage pipeline that combines
**Jev** (TypeSafe AI's "System One" model — fast, cheap, typed decisions
instead of generated text) with a **LangChain** ReAct agent (Gemini 2.5
Flash) for the actual reasoning and tool use.

- `triage_agent.py` — the pipeline: Jev classifies each ticket via the
  official `langchain_typesafe` package (`TypeSafeClassifier` +
  `Noul`/`Score`/`Choice`, department via `Choice`, urgency via `Score`,
  refund intent via `Noul`), then a LangChain agent uses those signals plus tools
  (`lookup_order`, `create_support_ticket`, `issue_refund`) to resolve
  it. `issue_refund` is gated on Jev's own refund-probability signal, so
  the fast model acts as a safety guardrail on the LLM agent, not just
  a classifier.
- `main.py` — runs three sample tickets through the pipeline.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
copy .env.example .env   # fill in TYPESAFE_API_KEY and GOOGLE_API_KEY
.\.venv\Scripts\python main.py
```


## Why split the work this way

Jev is designed to be ~40-200x faster and far cheaper than a frontier
LLM for exactly this kind of structured decision. Running it first lets
the pipeline route, prioritize, and safety-gate tickets before ever
paying for an LLM call, and the LLM agent only has to focus on
reasoning and drafting — not re-deriving classifications it's already
been handed.
