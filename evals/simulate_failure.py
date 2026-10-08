from unittest.mock import patch

from langchain_core.messages import HumanMessage
from langfuse import get_client
from langfuse.langchain import CallbackHandler

from app.graph import graph
from app.tools import get_part_metadata


USER_INPUT = "What is the current revision of part B-104?"

original_func = get_part_metadata.func


def corrupted_part_metadata(part_id: str) -> dict:
    # Simulated fault: the PLM lookup reports revision D
    # instead of C, so the agent answers from bad data.
    return {**original_func(part_id), "revision": "D"}


def simulate_failure() -> None:
    # Patch only for this run; app/graph.py and app/tools.py
    # are unchanged and the real tool is restored on exit.
    with patch.object(get_part_metadata, "func", corrupted_part_metadata):
        result = graph.invoke(
            {
                "user_input": USER_INPUT,
                "messages": [HumanMessage(content=USER_INPUT)],
                "routing_decision": None,
                "final_response": None,
            },
            config={
                "callbacks": [CallbackHandler()],
                "run_name": "plm-simulated-failure",
                "tags": ["failure-mining-demo"],
            },
        )

    print("Final answer:", result["messages"][-1].content)


# ---------------------------------------
# CLI entry point
# ---------------------------------------

if __name__ == "__main__":
    simulate_failure()

    # Short-lived script: send buffered traces
    # to Langfuse before the process exits.
    get_client().flush()
