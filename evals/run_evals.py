import json
import sys
from collections import defaultdict
from pathlib import Path

from app.supervisor import route_request
from evals.evaluators import route_correct
from evals.config import (
    MIN_OVERALL_ACCURACY,
    MIN_SLICE_ACCURACY,
)


DATASET_PATH = Path(__file__).parent / "dataset.json"


def load_dataset() -> list[dict]:
    with DATASET_PATH.open() as file:
        return json.load(file)


def evaluate_gate(
    overall_accuracy: float,
    slice_results: dict,
) -> bool:
    """
    Check whether evaluation results satisfy
    the configured release thresholds.
    """

    gate_passed = True

    # Check overall accuracy
    if overall_accuracy < MIN_OVERALL_ACCURACY:
        print(
            f"FAIL overall: "
            f"{overall_accuracy:.2%} "
            f"< required {MIN_OVERALL_ACCURACY:.2%}"
        )
        gate_passed = False

    # Check accuracy for each slice
    for slice_name, result in slice_results.items():

        slice_accuracy = (
            result["passed"] / result["total"]
        )

        required_accuracy = MIN_SLICE_ACCURACY.get(
            slice_name
        )

        if (
            required_accuracy is not None
            and slice_accuracy < required_accuracy
        ):
            print(
                f"FAIL {slice_name}: "
                f"{slice_accuracy:.2%} "
                f"< required {required_accuracy:.2%}"
            )

            gate_passed = False

    return gate_passed


def run_evaluations() -> bool:
    dataset = load_dataset()

    total = 0
    passed = 0

    slice_results = defaultdict(
        lambda: {"passed": 0, "total": 0}
    )

    # -----------------------------------
    # Run evaluation dataset
    # -----------------------------------

    for case in dataset:

        result = route_request(case["input"])

        is_correct = route_correct(
            expected_route=case["expected_route"],
            actual_route=result.route,
        )

        total += 1
        passed += int(is_correct)

        slice_name = case["slice"]

        slice_results[slice_name]["total"] += 1
        slice_results[slice_name]["passed"] += int(
            is_correct
        )

        print(
            f"{'PASS' if is_correct else 'FAIL'} "
            f"{case['id']} "
            f"expected={case['expected_route']} "
            f"actual={result.route}"
        )

    # -----------------------------------
    # Calculate metrics
    # -----------------------------------

    overall_accuracy = passed / total

    print()
    print("=== Evaluation Results ===")
    print()

    print(
        f"Overall accuracy: "
        f"{overall_accuracy:.2%}"
    )

    for slice_name, result in slice_results.items():

        slice_accuracy = (
            result["passed"] / result["total"]
        )

        print(
            f"{slice_name}: "
            f"{slice_accuracy:.2%}"
        )

    # -----------------------------------
    # Apply release gate
    # -----------------------------------

    print()
    print("=== Evaluation Gate ===")
    print()

    gate_passed = evaluate_gate(
        overall_accuracy=overall_accuracy,
        slice_results=slice_results,
    )

    print()

    if gate_passed:
        print("EVALUATION GATE: PASS")
        return True

    print("EVALUATION GATE: FAIL")
    return False


# ---------------------------------------
# CLI entry point
# ---------------------------------------

if __name__ == "__main__":

    passed = run_evaluations()

    if passed:
        sys.exit(0)

    sys.exit(1)