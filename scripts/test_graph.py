from langchain_core.messages import HumanMessage
from langfuse.langchain import CallbackHandler

from app.graph import graph


langfuse_handler = CallbackHandler()


user_input = (
    "What is the current revision of part B-104?"
)


result = graph.invoke(
    {
        "user_input": user_input,

        "messages": [
            HumanMessage(
                content=user_input
            )
        ],

        "routing_decision": None,
        "final_response": None,
    },

    config={
        "callbacks": [
            langfuse_handler
        ],
        "run_name": "plm-agent-graph",
        "tags": [
            "eval-lab",
            "plm",
        ],
    },
)


print(
    "Final response:",
    result["messages"][-1].content
)