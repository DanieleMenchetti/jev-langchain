"""Demo runner: a few sample tickets through the Jev + LangChain triage agent."""

from __future__ import annotations

import json

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from jev_client import JevClient
from triage_agent import handle_ticket

load_dotenv()

SAMPLE_TICKETS = [
    "My order ORD-1001 arrived broken and I want my money back, this is ridiculous.",
    "Hi, I can't log into my account, it keeps saying my password is wrong even after I reset it.",
    "Quick question: does the Pro subscription (ORD-1002) include priority email support?",
]


def main() -> None:
    jev = JevClient()  # reads TYPESAFE_API_KEY from the environment
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)  # reads GOOGLE_API_KEY

    for message in SAMPLE_TICKETS:
        print("=" * 80)
        print("TICKET:", message)
        result = handle_ticket(message, jev=jev, llm=llm)

        print("\nJev signals (~%.0fms):" % result["jev_signals"]["jev_latency_ms"])
        print(json.dumps(result["jev_signals"], indent=2))

        print("\nAgent resolution:")
        print(result["agent_reply"])
        print()


if __name__ == "__main__":
    main()
