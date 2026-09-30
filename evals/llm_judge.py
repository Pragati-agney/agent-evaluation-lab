import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from app.models import AnswerEvaluation


load_dotenv()

judge_model=ChatOpenAI(
    model="gpt-4.1-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0,
).with_structured_output(AnswerEvaluation)


JUDGE_PROMPT = """
You are evaluating the quality of an AI assistant response.

Evaluate the response using the following rubric.

Correctness:
1 = materially incorrect
3 = partially correct
5 = fully correct

Relevance:
1 = does not answer the user's question
3 = partially relevant
5 = directly answers the question

Groundedness:
1 = contains claims unsupported by the supplied context
3 = partially supported
5 = fully supported by the supplied context

Return integer scores from 1 to 5 and a concise explanation.
"""

def evaluate_answer(
        user_input:str,
        reference_answer: str,
        actual_answer :str,
)->AnswerEvaluation:
    evaluation_input = f"""
    USER INPUT:
    {user_input}

    REFERENCE ANSWER:
    {reference_answer}

    ACTUAL ANSWER:
    {actual_answer}
"""

    return judge_model.invoke(
        [
            ("system", JUDGE_PROMPT),
            ("human", evaluation_input),
        ]
    )
    

