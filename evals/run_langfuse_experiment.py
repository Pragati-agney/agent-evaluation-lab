import os
import sys

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langfuse import Evaluation, get_client
from langfuse.langchain import CallbackHandler
from pydantic import BaseModel

from app.graph import graph
from evals.evaluators import (
    required_tool_args_present,
    required_tools_called,
    route_correct,
    within_max_tool_calls,
)
from evals.run_graph_evals import (
    extract_final_answer,
    extract_tool_calls,
)


LANGFUSE_DATASET_NAME = "agent-evaluation-lab"

CHECK_NAMES = [
    "route",
    "required_tools",
    "required_tool_args",
    "max_tool_calls",
    "answer_correctness",
    "groundedness",
]


# ---------------------------------------
# Task
# ---------------------------------------

def graph_task(*, item, **kwargs) -> dict:
    user_input = item.input["input"]

    # Created inside the task so the graph's spans nest
    # under the experiment item's trace.
    result = graph.invoke(
        {
            "user_input": user_input,
            "messages": [HumanMessage(content=user_input)],
            "routing_decision": None,
            "final_response": None,
        },
        config={"callbacks": [CallbackHandler()]},
    )

    decision = result["routing_decision"]

    return {
        "route": decision.route if decision else None,
        "tool_calls": extract_tool_calls(result),
        "final_answer": extract_final_answer(result),
        # Evidence the groundedness judge checks the answer against.
        "tool_results": [
            message.content
            for message in result["messages"]
            if isinstance(message, ToolMessage)
        ],
    }


# ---------------------------------------
# Deterministic evaluators
# ---------------------------------------

def route_evaluator(*, output, expected_output, **kwargs) -> Evaluation:
    passed = route_correct(
        expected_route=expected_output["expected_route"],
        actual_route=output["route"],
    )

    return Evaluation(
        name="route",
        value=int(passed),
        comment=(
            f"expected={expected_output['expected_route']} "
            f"actual={output['route']}"
        ),
    )


def required_tools_evaluator(
    *, output, expected_output, **kwargs
) -> Evaluation:
    passed = required_tools_called(
        required_tools=expected_output["constraints"]["required_tools"],
        tool_calls=output["tool_calls"],
    )

    return Evaluation(name="required_tools", value=int(passed))


def required_tool_args_evaluator(
    *, output, expected_output, **kwargs
) -> Evaluation:
    passed = required_tool_args_present(
        required_tool_args=expected_output["constraints"][
            "required_tool_args"
        ],
        tool_calls=output["tool_calls"],
    )

    return Evaluation(name="required_tool_args", value=int(passed))


def max_tool_calls_evaluator(
    *, output, expected_output, **kwargs
) -> Evaluation:
    max_tool_calls = expected_output["constraints"]["max_tool_calls"]

    passed = within_max_tool_calls(
        max_tool_calls=max_tool_calls,
        tool_calls=output["tool_calls"],
    )

    return Evaluation(
        name="max_tool_calls",
        value=int(passed),
        comment=(
            f"{len(output['tool_calls'])} call(s), "
            f"max {max_tool_calls}"
        ),
    )


# ---------------------------------------
# LLM-based evaluator
# ---------------------------------------

class AnswerCorrectness(BaseModel):
    correct: bool
    reason: str


correctness_judge = ChatOpenAI(
    model="gpt-4.1-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0,
).with_structured_output(AnswerCorrectness)


CORRECTNESS_PROMPT = """
You are grading an AI assistant's answer against a reference answer.

Mark the answer correct only if it states the same facts as the
reference answer. Different wording is fine. Any contradicted,
missing, or changed fact (for example a different revision,
part number, or value) makes the answer incorrect.

Return whether the answer is correct and a one-sentence reason.
"""


def answer_correctness_evaluator(
    *, input, output, expected_output, **kwargs
) -> Evaluation:
    judgement = correctness_judge.invoke(
        [
            ("system", CORRECTNESS_PROMPT),
            (
                "human",
                f"QUESTION:\n{input['input']}\n\n"
                f"REFERENCE ANSWER:\n{expected_output['reference_answer']}\n\n"
                f"ACTUAL ANSWER:\n{output['final_answer']}",
            ),
        ]
    )

    return Evaluation(
        name="answer_correctness",
        value=int(judgement.correct),
        comment=judgement.reason,
    )


class Groundedness(BaseModel):
    grounded: bool
    reason: str


groundedness_judge = ChatOpenAI(
    model="gpt-4.1-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0,
).with_structured_output(Groundedness)


GROUNDEDNESS_PROMPT = """
You are checking whether an AI assistant's answer is grounded in
the tool evidence it received.

Judge only against the TOOL EVIDENCE. Ignore outside knowledge and
do not assess whether the evidence itself is true.

Mark the answer grounded only if every factual claim in it is
supported by the tool evidence. Any claim that is missing from,
or contradicts, the tool evidence makes the answer ungrounded.

Return whether the answer is grounded and a one-sentence reason.
"""


def groundedness_evaluator(*, input, output, **kwargs) -> Evaluation:
    # This experiment is PLM-only, so every answer should rest
    # on tool evidence; without any, fail instead of guessing.
    if not output["tool_results"]:
        return Evaluation(
            name="groundedness",
            value=0,
            comment="No tool results in this execution to ground the answer.",
        )

    tool_evidence = "\n".join(
        str(tool_result) for tool_result in output["tool_results"]
    )

    judgement = groundedness_judge.invoke(
        [
            ("system", GROUNDEDNESS_PROMPT),
            (
                "human",
                f"QUESTION:\n{input['input']}\n\n"
                f"TOOL EVIDENCE:\n{tool_evidence}\n\n"
                f"ANSWER:\n{output['final_answer']}",
            ),
        ]
    )

    return Evaluation(
        name="groundedness",
        value=int(judgement.grounded),
        comment=judgement.reason,
    )


# ---------------------------------------
# Run experiment
# ---------------------------------------

def run_experiment() -> bool:
    langfuse = get_client()

    dataset = langfuse.get_dataset(LANGFUSE_DATASET_NAME)

    result = dataset.run_experiment(
        name="graph-eval",
        description="Full graph run with deterministic checks",
        task=graph_task,
        evaluators=[
            route_evaluator,
            required_tools_evaluator,
            required_tool_args_evaluator,
            max_tool_calls_evaluator,
            answer_correctness_evaluator,
            groundedness_evaluator,
        ],
    )

    print(result.format(include_item_results=True))
    print()

    # The SDK drops items whose task failed and skips
    # evaluators that raised, so check nothing is missing.
    if len(result.item_results) != len(dataset.items):
        print(
            f"FAIL: {len(result.item_results)} of "
            f"{len(dataset.items)} item(s) completed"
        )
        return False

    all_passed = True

    for item_result in result.item_results:
        scores = {
            evaluation.name: evaluation.value
            for evaluation in item_result.evaluations
        }

        for check_name in CHECK_NAMES:
            if scores.get(check_name) != 1:
                print(
                    f"FAIL {item_result.item.id} {check_name}: "
                    f"{scores.get(check_name, 'missing')}"
                )
                all_passed = False

    print(
        "LANGFUSE EXPERIMENT: "
        f"{'PASS' if all_passed else 'FAIL'}"
    )

    return all_passed


# ---------------------------------------
# CLI entry point
# ---------------------------------------

if __name__ == "__main__":
    passed = run_experiment()

    if passed:
        sys.exit(0)

    sys.exit(1)
