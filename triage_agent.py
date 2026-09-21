"""Agentic AI use case: support-ticket triage with Jev + LangChain.

Architecture
------------
Every incoming ticket goes through two stages:

1. System 1 (Jev) - one fast, cheap, parallel call that answers three
   typed questions about the raw message:
     - department  (Choice) -> which team should own this ticket
     - urgency     (Score)  -> how urgent it is, on an ordered rubric
     - wants_refund(Noul)   -> does the customer want money back

   These signals are used both as *context* handed to the LLM agent and
   as a *guardrail*: the high-stakes `issue_refund` tool refuses to run
   unless Jev's own refund signal actually supports it, independent of
   whatever the LLM decided.

2. System 2 (a LangChain ReAct agent backed by Gemini 2.5 Flash) - reads the Jev
   signals plus the ticket text, reasons about what to do, and calls
   tools (look up the order, open a ticket, draft a reply, or issue a
   refund) to produce a final resolution.

This mirrors the "fast structured decision gates an expensive LLM agent"
pattern Jev is designed for: cheap classification/safety checks up
front, LLM reasoning only where it's actually needed.
"""

from __future__ import annotations

from typing import Any, Dict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from jev_client import Choice, JevClient, Noul, Score

# --------------------------------------------------------------------------
# Stage 1: Jev triage
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


def jev_triage(client: JevClient, message: str) -> Dict[str, Any]:
    """Run the three Jev question types against a raw ticket message."""
    answers = client.ask(
        state=message,
        questions={
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
    )

    department = answers["department"]
    urgency = answers["urgency"]
    wants_refund = answers["wants_refund"]

    return {
        "department": department.choice,
        "department_confidence": department.confidence,
        "urgency_score": urgency.score,
        "urgency_label": URGENCY_LEVELS[round(min(max(urgency.score, 0), len(URGENCY_LEVELS) - 1))],
        "urgency_confidence": urgency.confidence,
        "refund_probability": wants_refund.noul,
        "likely_refund_request": wants_refund.is_yes,
        "jev_latency_ms": client.last_latency_ms,
    }


# --------------------------------------------------------------------------
# Stage 2: tools available to the LangChain agent
# --------------------------------------------------------------------------

# Toy in-memory "order database" for the demo.
_ORDERS = {
    "ORD-1001": {"item": "Wireless mouse", "amount": 29.90, "status": "delivered"},
    "ORD-1002": {"item": "Annual Pro subscription", "amount": 119.00, "status": "active"},
}


def build_tools(jev_signals: Dict[str, Any]):
    """Build the agent's toolset, closing over this ticket's Jev signals.

    `issue_refund` is deliberately gated on Jev's own `wants_refund`
    signal: even if the LLM decides to call it, the tool refuses unless
    Jev's independent read of the message actually supports a refund
    intent above a safety threshold. This is the "System 1 guards
    System 2" pattern.
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
        """File a support ticket for a human agent, tagged with the department/urgency Jev already computed."""
        return (
            f"Ticket filed -> department={jev_signals['department']}, "
            f"urgency={jev_signals['urgency_label']!r}, summary={summary!r}"
        )

    @tool
    def issue_refund(order_id: str, amount: float) -> str:
        """Issue a refund for an order. Only allowed when the customer's refund intent is well established."""
        if jev_signals["refund_probability"] < 0.65:
            return (
                "REFUND BLOCKED: Jev's refund-intent probability is only "
                f"{jev_signals['refund_probability']:.2f} (< 0.65). Ask the customer to "
                "confirm they want a refund before calling this tool again."
            )
        order = _ORDERS.get(order_id)
        if not order:
            return f"No order found with id {order_id}"
        return f"Refund of ${amount:.2f} issued for {order_id} ({order['item']})."

    return [lookup_order, create_support_ticket, issue_refund]


# --------------------------------------------------------------------------
# Putting it together
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a customer support agent. A fast pre-classifier (Jev) has
already scored this ticket; treat its numbers as reliable priors, not
suggestions to re-derive from scratch:

- department: {department} (confidence {department_confidence:.2f})
- urgency: {urgency_label} (score {urgency_score:.2f}, confidence {urgency_confidence:.2f})
- refund intent probability: {refund_probability:.2f}

Resolve the ticket: look up any order mentioned, and either file a
support ticket or issue a refund as appropriate. Be concise.
"""


def handle_ticket(message: str, jev: JevClient, llm: BaseChatModel) -> Dict[str, Any]:
    """Run the full triage -> agent pipeline for one incoming ticket."""
    signals = jev_triage(jev, message)

    tools = build_tools(signals)
    agent = create_react_agent(model=llm, tools=tools)

    result = agent.invoke(
        {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT.format(**signals)),
                {"role": "user", "content": message},
            ]
        }
    )
    final_reply = result["messages"][-1].content

    return {"jev_signals": signals, "agent_reply": final_reply}
