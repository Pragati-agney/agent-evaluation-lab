import json
import sys
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from langfuse import get_client
from langfuse.langchain import CallbackHandler

from app.graph import graph
from evals.evaluators import (
    required_tool_args_present,
    required_tools_called,
    route_correct,
    within_max_tool_calls,
)


DATASET_PATH = Path(__file__).parent / "graph_dataset.json"

langfuse_handler = CallbackHandler()


def load_dataset() -> list[dict]:
    with DATASET_PATH.open() as file:
        return json.load(file)


def run_graph(case: dict) -> dict:
    return graph.invoke(
        {
            "user_input": case["input"],
            "messages": [HumanMessage(content=case["input"])],
            "routing_decision": None,
            "final_response": None,
        },
        config={
            "callbacks": [langfuse_handler],
            "run_name": f"graph-eval-{case['id']}",
            "tags": ["eval-lab", "graph-eval", case["slice"]],
            "metadata": {"case_id": case["id"]},
        },
    )


def extract_tool_calls(result: dict) -> list[dict]:
    """
    Collect every tool call the agents requested,
    in the order they were made.
    """

    tool_calls = []

    for message in result["messages"]:
        if isinstance(message, AIMessage):
            for tool_call in message.tool_calls:
                tool_calls.append(
                    {
                        "name": tool_call["name"],
                        "args": tool_call["args"],
                    }
                )

    return tool_calls


def extract_final_answer(result: dict) -> str:
    # The PLM path answers via messages; the other
    # paths set final_response instead.
    if result["final_response"] is not None:
        return result["final_response"]

    return result["messages"][-1].content


def score_case(
    case: dict,
    route: str | None,
    tool_calls: list[dict],
) -> dict[str, bool]:
    constraints = case["constraints"]

    return {
        "route": route_correct(
            expected_route=case["expected_route"],
            actual_route=route,
        ),
        "required_tools": required_tools_called(
            required_tools=constraints["required_tools"],
            tool_calls=tool_calls,
        ),
        "required_tool_args": required_tool_args_present(
            required_tool_args=constraints["required_tool_args"],
            tool_calls=tool_calls,
        ),
        "max_tool_calls": within_max_tool_calls(
            max_tool_calls=constraints["max_tool_calls"],
            tool_calls=tool_calls,
        ),
    }


def run_graph_evaluations() -> bool:
    dataset = load_dataset()

    all_passed = True

    for case in dataset:

        result = run_graph(case)

        decision = result["routing_decision"]
        route = decision.route if decision else None

        print(f"=== {case['id']} ===")
        print()
        print(f"Input: {case['input']}")
        print(f"Route: {route}")
        print()

        tool_calls = extract_tool_calls(result)

        print(f"Tool calls ({len(tool_calls)}):")

        for tool_call in tool_calls:
            print(
                f"  - {tool_call['name']} "
                f"{json.dumps(tool_call['args'])}"
            )

        print()
        print("Final answer:")
        print(extract_final_answer(result))
        print()

        # -----------------------------------
        # Deterministic scoring
        # -----------------------------------

        scores = score_case(case, route, tool_calls)

        print("Checks:")

        for check_name, check_passed in scores.items():
            print(
                f"  {'PASS' if check_passed else 'FAIL'} "
                f"{check_name}"
            )

        case_passed = all(scores.values())
        all_passed = all_passed and case_passed

        print()
        print(
            f"CASE {case['id']}: "
            f"{'PASS' if case_passed else 'FAIL'}"
        )
        print()

    print(
        "GRAPH EVALUATION: "
        f"{'PASS' if all_passed else 'FAIL'}"
    )

    return all_passed


# ---------------------------------------
# CLI entry point
# ---------------------------------------

if __name__ == "__main__":
    passed = run_graph_evaluations()

    # Short-lived script: send buffered traces
    # to Langfuse before the process exits.
    get_client().flush()

    if passed:
        sys.exit(0)

    sys.exit(1)
