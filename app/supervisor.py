import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langfuse.langchain import CallbackHandler

from app.models import RoutingDecision

load_dotenv()

langfuse_handler=CallbackHandler()

model= ChatOpenAI(
    model="gpt-4.1-mini",
    api_key= os.getenv("OPENAI_API_KEY"),
    temperature=0
)

structured_model = model.with_structured_output(RoutingDecision)

SYSTEM_PROMPT = """
You are a routing supervisor for an engineering platform.

Route each user request to exactly one destination:

cad_agent:
Use when the user wants to create or modify geometry,
dimensions, parameters, or CAD models.

simulation_agent:
Use when the user wants to run or analyze an engineering
simulation,stress analysis, or similar analysis.

plm_agent:
Use when the user asks about revisions, lifecycle state,
version history, or product lifecycle information.

clarify:
Use when there is not enough information to determine
the correct destination.

Return a routing decision and a short reason.
"""

def route_request(user_input:str) -> RoutingDecision:
    return structured_model.invoke(
       [
            ("system", SYSTEM_PROMPT),
            ("human", user_input),
        ] ,
         config={
            "callbacks": [langfuse_handler],
            "run_name": "supervisor-routing",
        },

    )