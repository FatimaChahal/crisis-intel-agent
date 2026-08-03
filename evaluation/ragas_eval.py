from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from dotenv import load_dotenv

from evaluation.vector_store import build_vector_store, search_similar
from agents.scout_agent import build_agent

load_dotenv()


def build_eval_dataset() -> Dataset:
    """
    Build a evaluation dataset for the RAG pipeline.

    Returns:
        A HuggingFace Dataset with questions, answers, contexts and ground truths.
    """
    questions = [
        "What actions were taken during the flood in Germany?",
        "How severe was the earthquake in Turkey?",
        "What happened during the wildfire in Greece?",
    ]

    ground_truths = [
        "Emergency evacuation, military deployment, temporary bridges built. Recovery took 6 months.",
        "Magnitude 7.8 earthquake. 50,000 deaths, 100,000 injured. International rescue teams deployed.",
        "3 deaths, 20,000 evacuated. Aerial firefighting and EU civil protection mechanism activated.",
    ]

    collection = build_vector_store()
    agent = build_agent()

    answers = []
    contexts = []

    for question in questions:
        result = agent.invoke(
            {
                "alerte": question,
                "contexte": "",
                "analyse": "",
            }
        )
        answers.append(result["analyse"])
        docs = search_similar(collection, question, n=2)
        contexts.append(docs)

    return Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )


def run_evaluation() -> None:
    """
    Run RAGAS evaluation on the RAG pipeline and print results.
    """
    print("🔄 Building evaluation dataset...")
    dataset = build_eval_dataset()

    print("📊 Running RAGAS evaluation...")
    results = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    print("\n✅ RAGAS Evaluation Results:")
    print(results)


if __name__ == "__main__":
    run_evaluation()
