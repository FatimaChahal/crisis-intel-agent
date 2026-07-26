import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from evaluation.vector_store import build_vector_store, search_similar
from mlflow_tracking.tracker import track_rag_run

load_dotenv()


def score_with_llm(question: str, context: str, answer: str) -> dict:
    """
    Score RAG response using LLM as a judge.

    Args:
        question: The original question/alert.
        context: The retrieved context documents.
        answer: The generated answer.

    Returns:
        Dictionary with faithfulness and relevancy scores.
    """
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    prompt = f"""You are a RAG evaluation expert. Score the following on a scale of 0.0 to 1.0.

QUESTION: {question}

CONTEXT (retrieved documents):
{context}

ANSWER: {answer}

Evaluate:
1. FAITHFULNESS (0.0-1.0): Is the answer based on the context? 1.0 = fully grounded, 0.0 = hallucinated
2. RELEVANCY (0.0-1.0): Does the answer address the question? 1.0 = perfectly relevant, 0.0 = irrelevant

Respond ONLY in this format:
FAITHFULNESS: <score>
RELEVANCY: <score>"""

    response = llm.invoke(prompt).content
    scores = {}
    for line in response.strip().split("\n"):
        if "FAITHFULNESS:" in line:
            scores["faithfulness"] = float(line.split(":")[1].strip())
        elif "RELEVANCY:" in line:
            scores["relevancy"] = float(line.split(":")[1].strip())
    return scores


def run_evaluation() -> None:
    """
    Run RAG evaluation on test questions and track results with MLflow.
    """
    test_cases = [
        {
            "question": "flood in germany severity orange",
            "ground_truth": "Emergency evacuation, military deployment, temporary bridges built."
        },
        {
            "question": "earthquake in turkey severity red",
            "ground_truth": "International rescue teams, tent cities, emergency food and water supply."
        },
        {
            "question": "wildfire in greece severity extreme",
            "ground_truth": "Aerial firefighting, evacuation by sea, EU civil protection mechanism activated."
        },
    ]

    collection = build_vector_store()
    all_faithfulness = []
    all_relevancy = []

    print("\n📊 Running RAG Evaluation...\n")

    for i, tc in enumerate(test_cases):
        print(f"Test {i+1}/{ len(test_cases)}: {tc['question']}")

        start = time.time()
        docs = search_similar(collection, tc["question"], n=2)
        context = "\n\n".join(docs)
        latency = (time.time() - start) * 1000

        from langchain_groq import ChatGroq
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        prompt = f"""You are a crisis analyst. Use the context below to analyze the alert.

CONTEXT: {context}

ALERT: {tc['question']}

Provide 3 bullet points: type, severity, recommended actions."""
        answer = llm.invoke(prompt).content

        scores = score_with_llm(tc["question"], context, answer)
        all_faithfulness.append(scores.get("faithfulness", 0))
        all_relevancy.append(scores.get("relevancy", 0))

        track_rag_run(
            query=tc["question"],
            n_results=2,
            model_name="llama-3.3-70b-versatile",
            latency_ms=latency,
            n_docs_retrieved=len(docs),
            response_length=len(answer),
        )

        print(f"  Faithfulness : {scores.get('faithfulness', 0):.2f}")
        print(f"  Relevancy    : {scores.get('relevancy', 0):.2f}")
        print()

    avg_faith = sum(all_faithfulness) / len(all_faithfulness)
    avg_rel = sum(all_relevancy) / len(all_relevancy)

    print("=" * 40)
    print(f"✅ Avg Faithfulness : {avg_faith:.2f}")
    print(f"✅ Avg Relevancy    : {avg_rel:.2f}")
    print("=" * 40)


if __name__ == "__main__":
    run_evaluation()