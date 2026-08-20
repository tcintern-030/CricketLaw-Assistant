import os

from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langsmith.schemas import Example, Run

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY not found in .env")

llm = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0,
    api_key=MISTRAL_API_KEY
)

# ---------------------------------------------------------
# Helper: Convert Mistral response to string
# ---------------------------------------------------------

def get_response_text(response):
    """
    Converts Mistral's response content into a normal string.
    """

    content = response.content

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):

        text_parts = []

        for item in content:

            if isinstance(item, str):
                text_parts.append(item)

            elif isinstance(item, dict):

                if "text" in item:
                    text_parts.append(str(item["text"]))

        return "".join(text_parts).strip()

    return str(content).strip()


# ---------------------------------------------------------
# Helper: Extract numerical score
# ---------------------------------------------------------

def get_score(response):
    """
    Extracts a numerical score between 0 and 1.
    """

    try:

        text = get_response_text(response)

        text = text.replace("Score:", "")
        text = text.replace("score:", "")
        text = text.strip()

        score = float(text)

        # Make sure score stays between 0 and 1
        score = max(0.0, min(1.0, score))

        return score

    except (ValueError, TypeError, AttributeError):

        return 0.0


# =========================================================
# CORRECTNESS EVALUATOR
# =========================================================

def correctness_evaluator(run: Run, example: Example):
    """
    Evaluates whether the generated answer is correct
    compared with the expected answer.
    """

    question = example.inputs.get(
        "question",
        ""
    )

    expected_answer = example.outputs.get(
        "expected_answer",
        ""
    )

    generated_answer = run.outputs.get(
        "answer",
        ""
    )

    prompt = f"""
You are an evaluator for a Cricket Laws RAG system.

Evaluate the correctness of the generated answer.

Question:
{question}

Expected Answer:
{expected_answer}

Generated Answer:
{generated_answer}

Compare the generated answer with the expected answer.

Use the following scoring:

1.0 = Completely correct
0.75 = Mostly correct
0.5 = Partially correct
0.25 = Mostly incorrect
0.0 = Completely incorrect

Consider the meaning of the answer, not exact wording.

Return ONLY the numerical score.
"""

    response = llm.invoke(prompt)

    score = get_score(response)

    return {
        "key": "correctness",
        "score": score
    }


# =========================================================
# RELEVANCE EVALUATOR
# =========================================================

def relevance_evaluator(run: Run, example: Example):
    """
    Evaluates whether the generated answer directly
    addresses the user's question.
    """

    question = example.inputs.get(
        "question",
        ""
    )

    generated_answer = run.outputs.get(
        "answer",
        ""
    )

    prompt = f"""
You are an evaluator for a Cricket Laws RAG system.

Evaluate the relevance of the generated answer.

Question:
{question}

Generated Answer:
{generated_answer}

Determine whether the generated answer actually
answers the user's question.

Use the following scoring:

1.0 = Directly and completely relevant
0.75 = Mostly relevant
0.5 = Partially relevant
0.25 = Mostly irrelevant
0.0 = Completely irrelevant

An answer can contain correct cricket information
but still receive a low score if it does not answer
the specific question.

Return ONLY the numerical score.
"""

    response = llm.invoke(prompt)

    score = get_score(response)

    return {
        "key": "relevance",
        "score": score
    }


# =========================================================
# GROUNDEDNESS EVALUATOR
# =========================================================

def groundedness_evaluator(run: Run, example: Example):
    """
    Evaluates whether the generated answer is supported
    by the retrieved context.
    """

    generated_answer = run.outputs.get(
        "answer",
        ""
    )

    retrieved_context = run.outputs.get(
        "retrieved_context",
        ""
    )

    prompt = f"""
You are an evaluator for a Cricket Laws RAG system.

Your job is to determine whether the generated answer
is supported by the retrieved context.

Retrieved Context:
{retrieved_context}

Generated Answer:
{generated_answer}

Check the factual claims made in the generated answer.

Use the following scoring:

1.0 = Fully supported by the retrieved context
0.75 = Mostly supported
0.5 = Partially supported
0.25 = Mostly unsupported
0.0 = Completely unsupported

If the generated answer contains information that
cannot be supported by the retrieved context,
reduce the score.

Return ONLY the numerical score.
"""

    response = llm.invoke(prompt)

    score = get_score(response)

    return {
        "key": "groundedness",
        "score": score
    }