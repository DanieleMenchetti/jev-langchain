"""Measured comparison of the two triage workflows on the same 10 tickets:

1. "Only LLM"   -- an LLM call classifies the ticket (and can early-exit);
   otherwise the resolver LLM (same model) handles it.
2. "Jev + LLM"  -- Jev classifies the ticket (and can early-exit);
   otherwise the same resolver LLM handles it.

Both workflows share the exact same resolver stage and the exact same
early-exit thresholds (`should_early_exit` in `triage_agent.py`), so any
difference in latency or cost comes entirely from the classifier stage:
a general-purpose LLM call vs. a purpose-built Jev call.

Pricing used (published rates):

- Jev (jev-latest):  $0.042 / 1M input tokens, output free.
  https://www.jevai.org/jev-api , https://dev.to/valyuai/how-to-use-jev
- Gemini 2.5 Flash:  $0.30 / 1M input tokens, $2.50 / 1M output tokens
  (used for both the LLM classifier and the resolver agent).
  https://openrouter.ai/google/gemini-2.5-flash

This is not a controlled benchmark (network variance, provider load, and
model non-determinism all affect single-run numbers) -- run it a few
times if you want stable averages. It exists to make the tradeoff
concrete instead of asserting it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_typesafe import TypeSafeClassifier

from triage_agent import handle_ticket_jev_classifier, handle_ticket_llm_classifier

load_dotenv()

JEV_INPUT_PRICE_PER_M = 0.042
JEV_OUTPUT_PRICE_PER_M = 0.0
LLM_INPUT_PRICE_PER_M = 0.30
LLM_OUTPUT_PRICE_PER_M = 2.50

SAMPLE_TICKETS = [
    "My order ORD-1001 arrived broken and I want my money back, this is ridiculous.",
    "Hi, I can't log into my account, it keeps saying my password is wrong even after I reset it.",
    "Quick question: does the Pro subscription (ORD-1002) include priority email support?",
    "What are your customer support hours on weekends?",
    "Do you offer a student discount on the Pro plan?",
    "Our integration has been throwing 500 errors for the last hour, production is down, please help NOW.",
    "I'd like a refund for ORD-1002, the Pro subscription doesn't work at all.",
    "How do I export my account data to CSV?",
    "The mouse I ordered (ORD-1001) has a double-click issue, is this covered under warranty?",
    "What payment methods do you accept?",
]

BASELINE_LABEL = "Only LLM"


def _cost(input_tokens: int, output_tokens: int, input_price: float, output_price: float) -> float:
    return input_tokens / 1e6 * input_price + output_tokens / 1e6 * output_price


@dataclass
class Run:
    label: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    early_exit: bool


@dataclass
class Totals:
    runs: list = field(default_factory=list)

    def add(self, run: Run) -> None:
        self.runs.append(run)

    @property
    def latency_ms(self) -> float:
        return sum(r.latency_ms for r in self.runs)

    @property
    def cost_usd(self) -> float:
        return sum(r.cost_usd for r in self.runs)

    @property
    def early_exits(self) -> int:
        return sum(1 for r in self.runs if r.early_exit)


def run_llm_classifier_workflow(message: str, llm: ChatGoogleGenerativeAI) -> Run:
    result = handle_ticket_llm_classifier(message, llm=llm)
    signals = result["classifier_signals"]
    classifier_cost = _cost(signals["input_tokens"], signals["output_tokens"], LLM_INPUT_PRICE_PER_M, LLM_OUTPUT_PRICE_PER_M)
    resolver_cost = _cost(
        result["resolver_input_tokens"], result["resolver_output_tokens"], LLM_INPUT_PRICE_PER_M, LLM_OUTPUT_PRICE_PER_M
    )
    return Run(
        label=BASELINE_LABEL,
        latency_ms=result["total_latency_ms"],
        input_tokens=signals["input_tokens"] + result["resolver_input_tokens"],
        output_tokens=signals["output_tokens"] + result["resolver_output_tokens"],
        cost_usd=classifier_cost + resolver_cost,
        early_exit=result["early_exit"],
    )


def run_jev_classifier_workflow(message: str, jev: TypeSafeClassifier, llm: ChatGoogleGenerativeAI) -> Run:
    result = handle_ticket_jev_classifier(message, jev=jev, llm=llm)
    signals = result["classifier_signals"]
    classifier_cost = _cost(signals["input_tokens"], signals["output_tokens"], JEV_INPUT_PRICE_PER_M, JEV_OUTPUT_PRICE_PER_M)
    resolver_cost = _cost(
        result["resolver_input_tokens"], result["resolver_output_tokens"], LLM_INPUT_PRICE_PER_M, LLM_OUTPUT_PRICE_PER_M
    )
    return Run(
        label="Jev + LLM",
        latency_ms=result["total_latency_ms"],
        input_tokens=signals["input_tokens"] + result["resolver_input_tokens"],
        output_tokens=signals["output_tokens"] + result["resolver_output_tokens"],
        cost_usd=classifier_cost + resolver_cost,
        early_exit=result["early_exit"],
    )


def _print_run(run: Run) -> None:
    exit_note = "  [early-exit, no resolver call]" if run.early_exit else "  [resolver LLM invoked]"
    print(
        f"  {run.label:<10}: {run.latency_ms:7.0f} ms, "
        f"{run.input_tokens + run.output_tokens:5d} tokens, ${run.cost_usd:.6f}{exit_note}"
    )


def main() -> None:
    jev = TypeSafeClassifier()
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    totals: dict[str, Totals] = {}

    for i, message in enumerate(SAMPLE_TICKETS, start=1):
        print("=" * 88)
        print(f"TICKET {i}: {message}")

        for run in (
            run_llm_classifier_workflow(message, llm),
            run_jev_classifier_workflow(message, jev, llm),
        ):
            totals.setdefault(run.label, Totals()).add(run)
            _print_run(run)

    n = len(SAMPLE_TICKETS)
    baseline = totals[BASELINE_LABEL]

    print("=" * 88)
    print(f"SUMMARY over {n} ticket(s)")
    for label, t in totals.items():
        print(
            f"  {label:<10}: total {t.latency_ms:7.0f} ms (avg {t.latency_ms / n:6.0f} ms/ticket), "
            f"total ${t.cost_usd:.6f} (avg ${t.cost_usd / n:.6f}/ticket), "
            f"{t.early_exits}/{n} auto-resolved without a resolver LLM call"
        )

    print(f"\n  Comparison:")
    for label, t in totals.items():
        if label == BASELINE_LABEL:
            continue
        latency_delta_pct = (t.latency_ms - baseline.latency_ms) / baseline.latency_ms * 100
        cost_delta_pct = (t.cost_usd - baseline.cost_usd) / baseline.cost_usd * 100
        print(f"    {label:<10} vs. {BASELINE_LABEL}: {latency_delta_pct:+.1f}% latency, {cost_delta_pct:+.1f}% cost")


if __name__ == "__main__":
    main()
