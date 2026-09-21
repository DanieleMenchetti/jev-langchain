"""Minimal Python client for TypeSafe AI's Jev "System One" model.

Jev (https://typesafe.ai) doesn't generate text: it takes a piece of state
and answers typed questions about it in parallel, returning calibrated
probabilities instead of prose. It exposes three question primitives:

- Noul   -> a yes/no proposition, returns P(yes)
- Score  -> position on an ordered rubric, returns expected score +
            per-level probabilities + confidence
- Choice -> pick one of up to 255 labeled options, returns probabilities
            per option + confidence

This client talks directly to the documented HTTP endpoint so it has no
dependency on the (still fast-moving) official SDK.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import requests

JEV_API_URL = "https://api.typesafe.ai/v1/systemone"
JEV_DEFAULT_MODEL = "jev-latest"


class JevError(RuntimeError):
    """Raised when the Jev API returns an error or a malformed response."""


# --------------------------------------------------------------------------
# Question types
# --------------------------------------------------------------------------

@dataclass
class Noul:
    instructions: str

    def to_payload(self) -> Dict[str, Any]:
        return {"type": "noul", "instructions": self.instructions}


@dataclass
class Score:
    instructions: str
    criteria: List[str]  # 2-10 ordered rubric levels, low -> high

    def to_payload(self) -> Dict[str, Any]:
        if not (2 <= len(self.criteria) <= 10):
            raise ValueError("Score.criteria must have between 2 and 10 ordered levels")
        return {"type": "score", "instructions": self.instructions, "criteria": self.criteria}


@dataclass
class Choice:
    instructions: str
    criteria: Dict[str, str]  # option key -> description, up to 255 options

    def to_payload(self) -> Dict[str, Any]:
        if not (1 <= len(self.criteria) <= 255):
            raise ValueError("Choice.criteria must have between 1 and 255 options")
        return {"type": "choice", "instructions": self.instructions, "criteria": self.criteria}


Question = Union[Noul, Score, Choice]


# --------------------------------------------------------------------------
# Answer types
# --------------------------------------------------------------------------

@dataclass
class NoulAnswer:
    noul: float  # P(yes), in [0, 1] -- no separate confidence field

    @property
    def is_yes(self) -> bool:
        return self.noul >= 0.5


@dataclass
class ScoreAnswer:
    score: float  # can land between levels, e.g. 1.035
    probabilities: List[float]
    confidence: float


@dataclass
class ChoiceAnswer:
    choice: str
    probabilities: Dict[str, float]
    confidence: float


Answer = Union[NoulAnswer, ScoreAnswer, ChoiceAnswer]


def _parse_answer(question: Question, raw: Dict[str, Any]) -> Answer:
    if isinstance(question, Noul):
        return NoulAnswer(noul=float(raw["noul"]))
    if isinstance(question, Score):
        return ScoreAnswer(
            score=float(raw["score"]),
            probabilities=list(raw.get("probabilities", [])),
            confidence=float(raw.get("confidence", 0.0)),
        )
    if isinstance(question, Choice):
        return ChoiceAnswer(
            choice=raw["choice"],
            probabilities=dict(raw.get("probabilities", {})),
            confidence=float(raw.get("confidence", 0.0)),
        )
    raise JevError(f"Unknown question type: {question!r}")


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------

class JevClient:
    """Thin, synchronous client around the Jev /v1/systemone endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = JEV_DEFAULT_MODEL,
        timeout: float = 5.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        if not self.api_key:
            raise JevError("Set TYPESAFE_API_KEY or pass api_key= explicitly")
        self.model = model
        self.timeout = timeout
        self.last_latency_ms: float = 0.0
        self._session = requests.Session()

    def ask(
        self,
        state: Union[str, Dict[str, Any], List[Any]],
        questions: Dict[str, Question],
    ) -> Dict[str, Answer]:
        """Evaluate every question in `questions` against `state` in one call.

        All questions run in parallel server-side, so batching several
        questions costs almost the same as asking one.
        """
        body = {
            "model": self.model,
            "state": state,
            "questions": {key: q.to_payload() for key, q in questions.items()},
        }
        started = time.monotonic()
        response = self._session.post(
            JEV_API_URL,
            json=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.timeout,
        )
        self.last_latency_ms = (time.monotonic() - started) * 1000

        if response.status_code != 200:
            raise JevError(f"Jev API error {response.status_code}: {response.text}")

        payload = response.json()
        answers_raw = payload.get("answers", {})
        return {key: _parse_answer(q, answers_raw[key]) for key, q in questions.items()}
