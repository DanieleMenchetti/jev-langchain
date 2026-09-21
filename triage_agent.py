"""Agentic AI use case: support-ticket triage, Jev classifier vs. LLM classifier.

Uses the official `langchain_typesafe` integration (`TypeSafeClassifier`,
`Noul`, `Score`, `Choice`) instead of a hand-rolled HTTP client.

Architecture
------------
Both workflows share the exact same two-stage shape, so the only thing
that differs between them is *which engine does the classifying*:

    request -> [classifier] -> early exit? -> auto-file
                             -> otherwise   -> [resolver LLM + tools] -> reply

1. Classifier stage - answers the same three questions about the raw
   message (department, urgency, refund intent) and, from those
   signals, decides whether the ticket is safe to auto-resolve without
   ever reaching the resolver LLM:
     - `jev_triage()` - one Jev call (`Choice` + `Score` + `Noul`).
     - `llm_triage()` - one Gemini call with structured output, asked
       to answer the same three questions on the same rubric.
   Both return the same signal shape, and the same `should_early_exit()`
   threshold function decides the early exit either way.

2. Resolver stage (only reached when the classifier doesn't early-exit)
   - a LangChain ReAct agent (Gemini 2.5 Flash) reads the classifier's
   signals plus the ticket text and calls tools (look up the order,
   file a ticket, issue a refund) to produce a final resolution.
   `issue_refund` is gated on the classifier's own refund-probability
   signal, independent of whatever the resolver LLM decides mid-conversation.

Two workflows are exposed for comparison (see `benchmark.py`):
- `handle_ticket_llm_classifier` -- "only LLM": an LLM call classifies
  and can early-exit; otherwise the resolver LLM handles it.
- `handle_ticket_jev_classifier` -- "Jev + LLM": Jev classifies and can
  early-exit; otherwise the same resolver LLM handles it.

Because the resolver stage is identical in both workflows, any latency
or cost difference between them comes entirely from the classifier.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_typesafe import Choice, Noul, Score, TypeSafeClassifier
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# Shared classification vocabulary (both classifiers answer against this)
# --------------------------------------------------------------------------

DEPARTMENTS = {
    "billing": "Payments, invoices, refunds or subscription charges",
    "technical": "Bugs, errors or the product not working as expected",
    "account": "Login, password or account access problems",
    "other": "Anything that doesn't fit the categories above",
}

URGENCY_LEVELS = [
    "can wait, no real deadline",
    "should be handled within a few days",
    "needs attention today",
    "critical - customer is blocked or escalating right now",
]

# Thresholds for auto-resolving a ticket straight from the classifier's
# signals, skipping the resolver LLM entirely. Deliberately conservative:
# only low-stakes, unambiguous, non-refund tickets qualify. Applied the
# same way regardless of which classifier produced the signals.
EARLY_EXIT_MAX_URGENCY_SCORE = 1.0
EARLY_EXIT_MAX_REFUND_PROBABILITY = 0.15
EARLY_EXIT_MIN_DEPARTMENT_CONFIDENCE = 0.55


def should_early_exit(signals: Dict[str, Any]) -> bool:
    """Decide, from classifier signals alone, whether this ticket is safe
    to auto-route without ever calling the resolver LLM.
    """
    return (
        signals["urgency_score"] < EARLY_EXIT_MAX_URGENCY_SCORE
        and signals["refund_probability"] < EARLY_EXIT_MAX_REFUND_PROBABILITY
        and signals["department_confidence"] >= EARLY_EXIT_MIN_DEPARTMENT_CONFIDENCE
    )


def _urgency_label(score: float) -> str:
    level = round(min(max(score, 0), len(URGENCY_LEVELS) - 1))
    return URGENCY_LEVELS[level]


# --------------------------------------------------------------------------
# Classifier A: Jev
# --------------------------------------------------------------------------


def jev_triage(classifier: TypeSafeClassifier, message: str) -> Dict[str, Any]:
    """Classify a ticket with Jev's three question types in one call."""
    started = time.monotonic()
    response = classifier.invoke(
        {
            "state": message,
            "questions": {
                "department": Choice(
                    instructions="Which team should handle this customer message?",
                    criteria=DEPARTMENTS,
                ),
                "urgency": Score(
                    instructions="How urgent is this customer message?",
                    criteria=URGENCY_LEVELS,
                ),
                "wants_refund": Noul(
                    instructions="Is the customer explicitly asking for a refund or their money back?",
                ),
            },
        }
    )
    latency_ms = (time.monotonic() - started) * 1000

    department = response.choices["department"]
    urgency = response.scores["urgency"]
    wants_refund = response.nouls["wants_refund"]

    return {
        "engine": "jev",
        "department": department.choice,
        "department_confidence": department.confidence,
        "urgency_score": urgency.score,
        "urgency_label": _urgency_label(urgency.score),
        "urgency_confidence": urgency.confidence,
        "refund_probability": wants_refund.noul,
        "likely_refund_request": wants_refund.noul >= 0.5,
        "latency_ms": latency_ms,
        "input_tokens": response.usage.input_tokens or 0,
        "output_tokens": response.usage.output_tokens or 0,
    }


# --------------------------------------------------------------------------
# Classifier B: a plain LLM call with structured output
# --------------------------------------------------------------------------


class LLMTriageSignals(BaseModel):
    department: Literal["billing", "technical", "account", "other"]
    department_confidence: float = Field(ge=0, le=1)
    urgency_score: float = Field(ge=0, le=len(URGENCY_LEVELS) - 1)
    urgency_confidence: float = Field(ge=0, le=1)
    refund_probability: float = Field(ge=0, le=1)


_LLM_TRIAGE_PROMPT = """Classify this customer support message on three dimensions.

Department options:
{departments}

Urgency levels (0-indexed, pick the expected position on this scale):
{urgency_levels}

Also estimate the probability (0-1) that the customer is explicitly
asking for a refund or their money back.

Message: {message}
"""


def llm_triage(llm: BaseChatModel, message: str) -> Dict[str, Any]:
    """Classify a ticket with a single LLM call asked the same three
    questions Jev answers, using structured output for a comparable schema.
    """
    prompt = _LLM_TRIAGE_PROMPT.format(
        departments="\n".join(f"- {key}: {desc}" for key, desc in DEPARTMENTS.items()),
        urgency_levels="\n".join(f"{i}: {level}" for i, level in enumerate(URGENCY_LEVELS)),
        message=message,
    )
    classifier = llm.with_structured_output(LLMTriageSignals, include_raw=True)

    started = time.monotonic()
    result = classifier.invoke(prompt)
    latency_ms = (time.monotonic() - started) * 1000

    parsed: LLMTriageSignals = result["parsed"]
    usage = getattr(result["raw"], "usage_metadata", None) or {}

    return {
        "engine": "llm",
        "department": parsed.department,
        "department_confidence": parsed.department_confidence,
        "urgency_score": parsed.urgency_score,
        "urgency_label": _urgency_label(parsed.urgency_score),
        "urgency_confidence": parsed.urgency_confidence,
        "refund_probability": parsed.refund_probability,
        "likely_refund_request": parsed.refund_probability >= 0.5,
        "latency_ms": latency_ms,
        "input_tokens": usage.get("input_tokens", 0) or 0,
        "output_tokens": usage.get("output_tokens", 0) or 0,
    }


# --------------------------------------------------------------------------
# Resolver stage: shared by both workflows
# --------------------------------------------------------------------------

# Toy in-memory "order database" for the demo.
_ORDERS = {
    "ORD-1001": {"item": "Wireless mouse", "amount": 29.90, "status": "delivered"},
    "ORD-1002": {"item": "Annual Pro subscription", "amount": 119.00, "status": "active"},
}


def build_tools(signals: Dict[str, Any]):
    """Build the resolver agent's toolset, closing over this ticket's
    classifier signals.

    `issue_refund` is deliberately gated on the classifier's own
    `refund_probability` signal: even if the resolver LLM decides to
    call it, the tool refuses unless the classifier's independent read
    of the message actually supports a refund intent above a safety
    threshold. This is the "classifier guards resolver" pattern, and it
    applies identically whether the classifier was Jev or an LLM.
    """

    @tool
    def lookup_order(order_id: str) -> str:
        """Look up an order by its ID and return its item, amount and status."""
        order = _ORDERS.get(order_id)
        if not order:
            return f"No order found with id {order_id}"
        return f"{order_id}: {order['item']}, ${order['amount']:.2f}, status={order['status']}"

    @tool
    def create_support_ticket(summary: str) -> str:
        """File a support ticket for a human agent, tagged with the department/urgency already classified."""
        return (
            f"Ticket filed -> department={signals['department']}, "
            f"urgency={signals['urgency_label']!r}, summary={summary!r}"
        )

    @tool
    def issue_refund(order_id: str, amount: float) -> str:
        """Issue a refund for an order. Only allowed when the customer's refund intent is well established."""
        if signals["refund_probability"] < 0.65:
            return (
                "REFUND BLOCKED: the classifier's refund-intent probability is only "
                f"{signals['refund_probability']:.2f} (< 0.65). Ask the customer to "
                "confirm they want a refund before calling this tool again."
            )
        order = _ORDERS.get(order_id)
        if not order:
            return f"No order found with id {order_id}"
        return f"Refund of ${amount:.2f} issued for {order_id} ({order['item']})."

    return [lookup_order, create_support_ticket, issue_refund]


SYSTEM_PROMPT = """You are a customer support agent. A pre-classifier has already
scored this ticket; treat its numbers as reliable priors, not
suggestions to re-derive from scratch:

- department: {department} (confidence {department_confidence:.2f})
- urgency: {urgency_label} (score {urgency_score:.2f}, confidence {urgency_confidence:.2f})
- refund intent probability: {refund_probability:.2f}

Resolve the ticket: look up any order mentioned, and either file a
support ticket or issue a refund as appropriate. Be concise.
"""


def _sum_usage(messages) -> tuple[int, int]:
    """Sum input/output tokens across every AIMessage in an agent result."""
    input_tokens = output_tokens = 0
    for message in messages:
        usage = getattr(message, "usage_metadata", None)
        if usage:
            input_tokens += usage.get("input_tokens", 0) or 0
            output_tokens += usage.get("output_tokens", 0) or 0
    return input_tokens, output_tokens


def _resolve_or_exit(message: str, signals: Dict[str, Any], llm: BaseChatModel) -> Dict[str, Any]:
    """Auto-resolve from the classifier's signals, or hand off to the resolver agent."""
    if should_early_exit(signals):
        reply = (
            f"Ticket auto-filed -> department={signals['department']}, "
            f"urgency={signals['urgency_label']!r} (no resolver LLM call needed)."
        )
        return {
            "agent_reply": reply,
            "resolver_latency_ms": 0.0,
            "resolver_input_tokens": 0,
            "resolver_output_tokens": 0,
            "early_exit": True,
        }

    tools = build_tools(signals)
    agent = create_react_agent(model=llm, tools=tools)

    started = time.monotonic()
    result = agent.invoke(
        {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT.format(**signals)),
                {"role": "user", "content": message},
            ]
        }
    )
    latency_ms = (time.monotonic() - started) * 1000
    input_tokens, output_tokens = _sum_usage(result["messages"])

    return {
        "agent_reply": _as_text(result["messages"][-1].content),
        "resolver_latency_ms": latency_ms,
        "resolver_input_tokens": input_tokens,
        "resolver_output_tokens": output_tokens,
        "early_exit": False,
    }


# --------------------------------------------------------------------------
# Putting it together: the two workflows
# --------------------------------------------------------------------------


def handle_ticket_llm_classifier(message: str, llm: BaseChatModel) -> Dict[str, Any]:
    """'Only LLM' workflow: an LLM call classifies the ticket (and can
    early-exit); otherwise the resolver LLM handles it.
    """
    signals = llm_triage(llm, message)
    resolution = _resolve_or_exit(message, signals, llm)
    return {
        "classifier_signals": signals,
        "agent_reply": resolution["agent_reply"],
        "resolver_latency_ms": resolution["resolver_latency_ms"],
        "resolver_input_tokens": resolution["resolver_input_tokens"],
        "resolver_output_tokens": resolution["resolver_output_tokens"],
        "total_latency_ms": signals["latency_ms"] + resolution["resolver_latency_ms"],
        "early_exit": resolution["early_exit"],
    }


def handle_ticket_jev_classifier(message: str, jev: TypeSafeClassifier, llm: BaseChatModel) -> Dict[str, Any]:
    """'Jev + LLM' workflow: Jev classifies the ticket (and can
    early-exit); otherwise the same resolver LLM handles it.
    """
    signals = jev_triage(jev, message)
    resolution = _resolve_or_exit(message, signals, llm)
    return {
        "classifier_signals": signals,
        "agent_reply": resolution["agent_reply"],
        "resolver_latency_ms": resolution["resolver_latency_ms"],
        "resolver_input_tokens": resolution["resolver_input_tokens"],
        "resolver_output_tokens": resolution["resolver_output_tokens"],
        "total_latency_ms": signals["latency_ms"] + resolution["resolver_latency_ms"],
        "early_exit": resolution["early_exit"],
    }


def _as_text(content: Any) -> str:
    """Flatten a message's content into plain text.

    Some providers (e.g. Gemini) return content as a list of blocks
    (text, thought signatures, ...) instead of a plain string.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content)
