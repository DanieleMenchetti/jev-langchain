"""Demo runner: one sample ticket through the Jev + LLM triage workflow."""

from __future__ import annotations

import json

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_typesafe import TypeSafeClassifier

from triage_agent import handle_ticket_jev_classifier

load_dotenv()

SAMPLE_TICKET = "My order ORD-1001 arrived broken and I want my money back, this is ridiculous."


def main() -> None:
    jev = TypeSafeClassifier()  # reads TYPESAFE_API_KEY from the environment
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)  # reads GOOGLE_API_KEY

    print("TICKET:", SAMPLE_TICKET)
    result = handle_ticket_jev_classifier(SAMPLE_TICKET, jev=jev, llm=llm)

    print("\nJev signals (~%.0fms):" % result["classifier_signals"]["latency_ms"])
    print(json.dumps(result["classifier_signals"], indent=2))

    print("\nResolution%s:" % (" (early-exit, no resolver LLM call)" if result["early_exit"] else ""))
    print(result["agent_reply"])


if __name__ == "__main__":
    main()
