import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from langfuse import get_client


load_dotenv()


DATASET_PATH = (
    Path(__file__).parent.parent / "evals" / "graph_dataset.json"
)

# Existing dataset in Langfuse; this script never creates it.
LANGFUSE_DATASET_NAME = "agent-evaluation-lab"


def load_dataset() -> list[dict]:
    with DATASET_PATH.open() as file:
        return json.load(file)


def upload_dataset() -> None:
    langfuse = get_client()

    try:
        langfuse.get_dataset(LANGFUSE_DATASET_NAME)
    except Exception as error:
        print(
            f"Langfuse dataset '{LANGFUSE_DATASET_NAME}' "
            f"not found: {error}"
        )
        sys.exit(1)

    dataset = load_dataset()

    for case in dataset:

        # Item ids must be unique across all Langfuse datasets,
        # and reusing one upserts the item instead of duplicating.
        item_id = f"{LANGFUSE_DATASET_NAME}-{case['id']}"

        langfuse.create_dataset_item(
            dataset_name=LANGFUSE_DATASET_NAME,
            id=item_id,
            input={"input": case["input"]},
            expected_output={
                "expected_route": case["expected_route"],
                "constraints": case["constraints"],
                "reference_answer": case["reference_answer"],
            },
            metadata={
                "case_id": case["id"],
                "slice": case["slice"],
            },
        )

        print(f"Uploaded {item_id}")

    langfuse.flush()

    print()
    print(
        f"Uploaded {len(dataset)} item(s) to "
        f"'{LANGFUSE_DATASET_NAME}'"
    )


# ---------------------------------------
# CLI entry point
# ---------------------------------------

if __name__ == "__main__":
    upload_dataset()
