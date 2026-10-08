from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from app.models import RoutingDecision


class AgentState(TypedDict):
    user_input: str

    messages: Annotated[
        list[BaseMessage],
        add_messages,
    ]

    routing_decision: RoutingDecision | None

    final_response: str | None