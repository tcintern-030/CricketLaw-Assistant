import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from RetrivalForLangSmith import run_rag

DATASET_PATH = os.path.join(
    os.path.dirname(__file__),
    "dataset.json"
)


def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def run_evaluation():
    dataset = load_dataset()

    results = []

    print(f"\nRunning evaluation on {len(dataset)} test cases...\n")

    for test_case in dataset:

        question = test_case["question"]

        print("=" * 70)
        print(f"Test Case: {test_case['id']}")
        print(f"Type: {test_case['type']}")
        print(f"Question: {question}")

        try:
            result = run_rag(question)

            answer = result["answer"]
            documents = result["documents"]

            retrieved_context = []

            for document in documents:
                retrieved_context.append({
                    "content": document.page_content,
                    "metadata": document.metadata
                })

            evaluation_result = {
                "id": test_case["id"],
                "type": test_case["type"],
                "question": question,
                "expected_answer": test_case["expected_answer"],
                "key_facts": test_case["key_facts"],
                "generated_answer": answer,
                "retrieved_context": retrieved_context
            }

            results.append(evaluation_result)

            print("\nGenerated Answer:")
            print(answer)

            print(f"\nRetrieved Documents: {len(documents)}")
            print("Status: SUCCESS")

        except Exception as e:

            print(f"\nERROR: {e}")

            evaluation_result = {
                "id": test_case["id"],
                "type": test_case["type"],
                "question": question,
                "expected_answer": test_case["expected_answer"],
                "key_facts": test_case["key_facts"],
                "generated_answer": None,
                "retrieved_context": [],
                "error": str(e)
            }

            results.append(evaluation_result)

    return results


def save_results(results):
    output_path = os.path.join(
        os.path.dirname(__file__),
        "evaluation_results.json"
    )

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False
        )

    print("\n" + "=" * 70)
    print("Evaluation completed.")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    results = run_evaluation()
    save_results(results)