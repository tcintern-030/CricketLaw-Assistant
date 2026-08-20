import os
import json

from dotenv import load_dotenv
from langsmith import Client

from evaluators import (
    correctness_evaluator,
    relevance_evaluator,
    groundedness_evaluator,
)


# =========================================================
# PATHS AND ENVIRONMENT
# =========================================================

# Project root
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# Load .env from project root
load_dotenv(
    os.path.join(
        BASE_DIR,
        ".env"
    )
)


# =========================================================
# LANGSMITH CONFIGURATION
# =========================================================

LANGSMITH_API_KEY = os.getenv(
    "LANGSMITH_API_KEY"
)

if not LANGSMITH_API_KEY:

    raise ValueError(
        "LANGSMITH_API_KEY not found. "
        "Make sure it is present in your .env file."
    )


# Enable LangSmith tracing
os.environ["LANGSMITH_TRACING"] = "true"

os.environ["LANGSMITH_ENDPOINT"] = (
    "https://api.smith.langchain.com"
)


# Create LangSmith client
client = Client(
    api_key=LANGSMITH_API_KEY
)


# =========================================================
# LOAD EVALUATION RESULTS
# =========================================================

RESULTS_FILE = os.path.join(
    BASE_DIR,
    "evaluation",
    "evaluation_results.json"
)


with open(
    RESULTS_FILE,
    "r",
    encoding="utf-8"
) as file:

    results = json.load(file)


# Handle either list or dictionary format
if isinstance(results, dict):

    results = results.get(
        "results",
        []
    )


if not isinstance(results, list):

    raise ValueError(
        "evaluation_results.json must contain "
        "a list of evaluation cases."
    )


print(
    f"Loaded {len(results)} evaluation cases."
)


# =========================================================
# MAP RESULTS BY QUESTION
# =========================================================

results_by_question = {}


for item in results:

    question = item.get(
        "question",
        ""
    ).strip()

    if question:

        results_by_question[question] = item


# =========================================================
# CREATE / LOAD LANGSMITH DATASET
# =========================================================

DATASET_NAME = (
    "Cricket Laws RAG Evaluation"
)


existing_datasets = list(
    client.list_datasets(
        dataset_name=DATASET_NAME
    )
)


if existing_datasets:

    dataset = existing_datasets[0]

    print(
        "\nUsing existing LangSmith dataset:"
    )

    print(dataset.id)


else:

    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description=(
            "15 test cases for evaluating "
            "the Cricket Laws RAG Assistant."
        )
    )

    print(
        "\nCreated new LangSmith dataset:"
    )

    print(dataset.id)


# =========================================================
# CHECK EXISTING EXAMPLES
# =========================================================

existing_examples = list(
    client.list_examples(
        dataset_id=dataset.id
    )
)


existing_questions = set()


for example in existing_examples:

    question = example.inputs.get(
        "question",
        ""
    ).strip()

    if question:

        existing_questions.add(
            question
        )


# =========================================================
# ADD EXAMPLES TO LANGSMITH DATASET
# =========================================================

for item in results:

    question = item.get(
        "question",
        ""
    ).strip()

    expected_answer = item.get(
        "expected_answer",
        ""
    )

    if not question:

        print(
            "Skipping case: question is missing."
        )

        continue


    if question in existing_questions:

        print(
            f"Already exists: {question}"
        )

        continue


    client.create_example(

        inputs={
            "question": question
        },

        outputs={
            "expected_answer": expected_answer
        },

        dataset_id=dataset.id
    )


    print(
        f"Added: {question}"
    )


# =========================================================
# TARGET FUNCTION
# =========================================================

def target_function(inputs):
    """
    Returns the already-generated RAG answer
    and retrieved context.

    The RAG is NOT executed again.
    """

    question = inputs.get(
        "question",
        ""
    ).strip()


    item = results_by_question.get(
        question
    )


    if item is None:

        return {
            "answer": "",
            "retrieved_context": ""
        }


    # ---------------------------------------------
    # Get generated answer
    # ---------------------------------------------

    answer = item.get(
        "generated_answer",
        item.get(
            "answer",
            ""
        )
    )


    # ---------------------------------------------
    # Get retrieved context
    # ---------------------------------------------

    context = item.get(
        "retrieved_context",
        item.get(
            "context",
            item.get(
                "retrieved_documents",
                ""
            )
        )
    )


    # ---------------------------------------------
    # Convert context into readable text
    # ---------------------------------------------

    if isinstance(context, list):

        context_parts = []


        for document in context:

            if isinstance(document, dict):

                content = document.get(
                    "content",
                    ""
                )

                if content:

                    context_parts.append(
                        content
                    )

            else:

                context_parts.append(
                    str(document)
                )


        context = "\n\n".join(
            context_parts
        )


    return {
        "answer": answer,
        "retrieved_context": context
    }


# =========================================================
# RUN LANGSMITH EVALUATION
# =========================================================

print("\n")
print("=" * 60)
print("Starting LangSmith Evaluation")
print("=" * 60)
print()


experiment_results = client.evaluate(

    target_function,

    data=DATASET_NAME,

    evaluators=[
        correctness_evaluator,
        relevance_evaluator,
        groundedness_evaluator,
    ],

    experiment_prefix=(
        "cricket-laws-rag"
    ),

    description=(
        "Evaluation of Cricket Laws RAG "
        "using existing evaluation results."
    )
)


# =========================================================
# FINISHED
# =========================================================

print()
print("=" * 60)
print("Evaluation completed successfully!")
print("=" * 60)
print()

print("Open LangSmith to view:")
print("- Correctness")
print("- Relevance")
print("- Groundedness")
print("- Individual test cases")
print("- Overall evaluation scores")