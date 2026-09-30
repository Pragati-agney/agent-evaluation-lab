from evals.llm_judge import evaluate_answer


user_input = """
The maximum stress is 280 MPa.
The allowable stress is 250 MPa.

Is the design acceptable?
"""

reference_answer = """
No. The design is not acceptable because the maximum
stress of 280 MPa exceeds the allowable stress of 250 MPa.
"""

actual_answer = """
Yes. The design is perfectly safe because 280 MPa
is below the allowable stress of 250 MPa.
"""


result = evaluate_answer(
    user_input=user_input,
    reference_answer=reference_answer,
    actual_answer=actual_answer,
)


print(result)