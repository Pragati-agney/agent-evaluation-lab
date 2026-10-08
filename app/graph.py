import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from app.models import RoutingDecision
from app.state import AgentState
from app.tools import get_part_metadata


load_dotenv()


model = ChatOpenAI(
    model="gpt-4.1-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0,
)


# Give the PLM LLM access to the tool
plm_model = model.bind_tools(
    [get_part_metadata]
)


# LangGraph node responsible for executing the tool
plm_tool_node = ToolNode(
    [get_part_metadata]
)


# Supervisor must return our RoutingDecision structure
structured_model = model.with_structured_output(
    RoutingDecision
)


SYSTEM_PROMPT = """
You are a routing supervisor for an engineering platform.

Route each user request to exactly one destination:

cad_agent:
Use when the user wants to create or modify geometry,
dimensions, parameters, or CAD models.

simulation_agent:
Use when the user wants to run or analyze an engineering
simulation, stress analysis, or similar analysis.

plm_agent:
Use when the user asks about revisions, lifecycle state,
version history, or product lifecycle information.

clarify:
Use when there is not enough information to determine
the correct destination.

Return a routing decision and a short reason.
"""


PLM_SYSTEM_PROMPT = """
You are a PLM engineering assistant.

Answer questions about engineering part metadata,
including revision, lifecycle state, and version.

Use the get_part_metadata tool when you need
information about a specific part.

Do not invent part metadata.
"""


# --------------------------------------------------
# Nodes
# --------------------------------------------------


def supervisor_node(state: AgentState) -> dict:
    decision = structured_model.invoke(
        [
            ("system", SYSTEM_PROMPT),
            ("human", state["user_input"]),
        ]
    )

    return {
        "routing_decision": decision
    }


def cad_agent_node(state: AgentState) -> dict:
    return {
        "final_response": (
            "CAD agent handled the request: "
            f"{state['user_input']}"
        )
    }


def simulation_agent_node(state: AgentState) -> dict:
    return {
        "final_response": (
            "Simulation agent handled the request: "
            f"{state['user_input']}"
        )
    }


def plm_agent_node(state: AgentState) -> dict:
    response = plm_model.invoke(
        [
            ("system", PLM_SYSTEM_PROMPT),
            *state["messages"],
        ]
    )

    return {
        "messages": [response]
    }


def clarify_node(state: AgentState) -> dict:
    return {
        "final_response": (
            "More information is required."
        )
    }


# --------------------------------------------------
# Routing functions
# --------------------------------------------------


def route_after_supervisor(
    state: AgentState,
) -> str:

    decision = state["routing_decision"]

    if decision is None:
        return "clarify"

    return decision.route


def route_after_plm_agent(
    state: AgentState,
) -> str:

    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "plm_tools"

    return "end"


# --------------------------------------------------
# Build graph
# --------------------------------------------------


builder = StateGraph(AgentState)


builder.add_node(
    "supervisor",
    supervisor_node,
)

builder.add_node(
    "cad_agent",
    cad_agent_node,
)

builder.add_node(
    "simulation_agent",
    simulation_agent_node,
)

builder.add_node(
    "plm_agent",
    plm_agent_node,
)

builder.add_node(
    "plm_tools",
    plm_tool_node,
)

builder.add_node(
    "clarify",
    clarify_node,
)


# --------------------------------------------------
# Edges
# --------------------------------------------------


builder.add_edge(
    START,
    "supervisor",
)


builder.add_conditional_edges(
    "supervisor",
    route_after_supervisor,
    {
        "cad_agent": "cad_agent",
        "simulation_agent": "simulation_agent",
        "plm_agent": "plm_agent",
        "clarify": "clarify",
    },
)


# PLM agent can either call a tool or finish
builder.add_conditional_edges(
    "plm_agent",
    route_after_plm_agent,
    {
        "plm_tools": "plm_tools",
        "end": END,
    },
)


# After the tool executes, return to the PLM agent
builder.add_edge(
    "plm_tools",
    "plm_agent",
)


builder.add_edge(
    "cad_agent",
    END,
)

builder.add_edge(
    "simulation_agent",
    END,
)

builder.add_edge(
    "clarify",
    END,
)


# --------------------------------------------------
# Compile
# --------------------------------------------------


graph = builder.compile()