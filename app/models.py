from typing import Literal

from pydantic import BaseModel,Field


class RoutingDecision(BaseModel):
    route: Literal[
        "cad_agent",
        "simulation_agent",
        "plm_agent",
        "clarify",
    ]

    reason: str

class AnswerEvaluation(BaseModel):
    correctness: int = Field(ge=1, le=5)
    relevance: int = Field(ge=1, le=5)
    groundedness: int = Field(ge=1, le=5)
    explanation: str

