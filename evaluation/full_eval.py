import time
import json
import mlflow
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from agents.wildfire_agent_v2 import build_wildfire_agent_v2
from evaluation.vector_store import search_wildfires

load_dotenv()

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("wildfire-agent-evaluation")


# ── TEST DATASET ──────────────────────────────────────
# WILDFIRE_QUESTIONS = [
#     {
#         "question": "What happened during the large wildfire in Greece in 2023?",
#         "expected_topic": "wildfire",
#         "ground_truth": "In 2023, Greece experienced a severe wildfire with 96610 hectares burnt during summer."
#     },
#     {
#         "question": "Which wildfires occurred in Portugal in 2017?",
#         "expected_topic": "wildfire",
#         "ground_truth": "Portugal 2017 had severe wildfires burning over 60000 hectares in Viseu Dão Lafões region."
#     },
#     {
#         "question": "What is the risk of wildfire today in Gironde?",
#         "expected_topic": "wildfire",
#         "ground_truth": "Gironde has high wildfire risk in summer due to heat and drought conditions."
#     },
#     {
#         "question": "Tell me about wildfires in Turkey in 2021",
#         "expected_topic": "wildfire",
#         "ground_truth": "Turkey 2021 had a severe wildfire in Antalya burning 54769 hectares of conifer forest."
#     },
#     {
#         "question": "Quels incendies ont touché la France en été ?",
#         "expected_topic": "wildfire",
#         "ground_truth": "France summer wildfires mainly affected Bouches-du-Rhône with conifer vegetation."
#     },
# ]

# OUT_OF_SCOPE_QUESTIONS = [
#     {"question": "What is the best restaurant in Paris?", "expected_topic": "off_topic"},
#     {"question": "How do I make chocolate cake?", "expected_topic": "off_topic"},
#     {"question": "What is the capital of Germany?", "expected_topic": "off_topic"},
#     {"question": "Tell me about football results", "expected_topic": "off_topic"},
#     {"question": "What is the stock price of Apple?", "expected_topic": "off_topic"},
# ]


def load_test_dataset() -> tuple:
    """
    Load the professional test dataset from JSON file.

    Returns:
        Tuple of (wildfire_questions, off_topic_questions).
    """
    with open("evaluation/test_dataset.json", "r") as f:
        dataset = json.load(f)
    # Use subset to avoid rate limits
    return dataset["wildfire_questions"][:10], dataset["off_topic_questions"][:10]


WILDFIRE_QUESTIONS, OUT_OF_SCOPE_QUESTIONS = load_test_dataset()


def score_with_llm(question: str, context: str, answer: str, ground_truth: str) -> dict:
    """
    Score RAG response using LLM as judge with extended metrics.

    Args:
        question: Original question.
        context: Retrieved context documents.
        answer: Generated answer.
        ground_truth: Expected answer for correctness scoring.

    Returns:
        Dictionary with all RAG evaluation scores.
    """
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    prompt = f"""You are a RAG evaluation expert. Score the following on a scale of 0.0 to 1.0.

QUESTION: {question}

CONTEXT (retrieved documents):
{context}

ANSWER: {answer}

GROUND TRUTH: {ground_truth}

Evaluate:
1. FAITHFULNESS (0.0-1.0): Is the answer based ONLY on the context? 1.0 = fully grounded
2. RELEVANCY (0.0-1.0): Does the answer address the question? 1.0 = perfectly relevant
3. CONTEXT_PRECISION (0.0-1.0): Is the retrieved context relevant to the question? 1.0 = perfectly relevant
4. CORRECTNESS (0.0-1.0): Is the answer factually correct compared to ground truth? 1.0 = perfectly correct

Respond ONLY in this exact format:
FAITHFULNESS: <score>
RELEVANCY: <score>
CONTEXT_PRECISION: <score>
CORRECTNESS: <score>"""

    response = llm.invoke(prompt).content
    scores = {
        "faithfulness": 0.0,
        "relevancy": 0.0,
        "context_precision": 0.0,
        "correctness": 0.0,
    }
    for line in response.strip().split("\n"):
        for key in scores:
            if key.upper() + ":" in line.upper():
                try:
                    scores[key] = float(line.split(":")[1].strip())
                except ValueError:
                    pass
    return scores


def evaluate_guardrails() -> dict:
    """
    Evaluate guardrail precision on wildfire and off-topic questions.

    Returns:
        Dictionary with guardrail metrics.
    """
    print("\n📊 Evaluating Guardrails...")
    agent = build_wildfire_agent_v2()

    correct = 0
    total = len(WILDFIRE_QUESTIONS) + len(OUT_OF_SCOPE_QUESTIONS)

    # Wildfire questions should NOT be blocked
    wildfire_passed = 0
    for tc in WILDFIRE_QUESTIONS:
        result = agent.invoke(
            {
                "question": tc["question"],
                "history": "",
                "language": "English",
                "question_type": "case",
                "is_relevant": False,
                "context": [],
                "metadata": [],
                "analysis": "",
                "answer": "",
                "references": "",
                "weather_data": "",
                "stop_reason": "",
            }
        )
        if result["is_relevant"]:
            wildfire_passed += 1
            correct += 1
        print(
            f"  ✅ [{tc['expected_topic']}] {tc['question'][:50]}... → {'PASS' if result['is_relevant'] else 'FAIL'}"
        )

    # Off-topic questions should be blocked
    offtopic_blocked = 0
    for tc in OUT_OF_SCOPE_QUESTIONS:
        result = agent.invoke(
            {
                "question": tc["question"],
                "history": "",
                "language": "English",
                "question_type": "case",
                "is_relevant": False,
                "context": [],
                "metadata": [],
                "analysis": "",
                "answer": "",
                "references": "",
                "weather_data": "",
                "stop_reason": "",
            }
        )
        if not result["is_relevant"]:
            offtopic_blocked += 1
            correct += 1
        print(
            f"  ❌ [{tc['expected_topic']}] {tc['question'][:50]}... → {'BLOCKED ✅' if not result['is_relevant'] else 'NOT BLOCKED ❌'}"
        )

    precision = correct / total
    recall_wildfire = wildfire_passed / len(WILDFIRE_QUESTIONS)
    recall_offtopic = offtopic_blocked / len(OUT_OF_SCOPE_QUESTIONS)

    print(f"\n  Guardrail Precision     : {precision:.2f}")
    print(f"  Wildfire Recall         : {recall_wildfire:.2f}")
    print(f"  Off-topic Block Rate    : {recall_offtopic:.2f}")

    return {
        "guardrail_precision": precision,
        "wildfire_recall": recall_wildfire,
        "offtopic_block_rate": recall_offtopic,
    }


def evaluate_rag() -> dict:
    """
    Evaluate RAG pipeline with extended professional metrics.

    Returns:
        Dictionary with all RAG evaluation metrics.
    """
    print("\n📊 Evaluating RAG Pipeline...")

    all_faithfulness = []
    all_relevancy = []
    all_context_precision = []
    all_correctness = []
    all_latencies = []

    for tc in WILDFIRE_QUESTIONS:
        start = time.time()
        docs, metas = search_wildfires(tc["question"], n=3)
        latency = (time.time() - start) * 1000
        all_latencies.append(latency)

        if not docs:
            print(f"  ⚠️ No docs found for: {tc['question'][:50]}...")
            continue

        context = "\n\n".join(docs)
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
        answer = llm.invoke(
            f"You are a wildfire expert. Answer based on this context:\n{context}\n\nQuestion: {tc['question']}"
        ).content

        scores = score_with_llm(
            tc["question"], context, answer, tc.get("ground_truth", "")
        )
        all_faithfulness.append(scores["faithfulness"])
        all_relevancy.append(scores["relevancy"])
        all_context_precision.append(scores["context_precision"])
        all_correctness.append(scores["correctness"])

        print(f"  📄 {tc['question'][:50]}...")
        print(
            f"     Faith: {scores['faithfulness']:.2f} | Rel: {scores['relevancy']:.2f} | "
            f"Ctx: {scores['context_precision']:.2f} | Corr: {scores['correctness']:.2f} | "
            f"Lat: {latency:.0f}ms"
        )

    # Compute aggregates
    def avg(lst):
        return sum(lst) / len(lst) if lst else 0

    def p95(lst):
        return sorted(lst)[int(len(lst) * 0.95)] if lst else 0

    return {
        "avg_faithfulness": avg(all_faithfulness),
        "avg_relevancy": avg(all_relevancy),
        "avg_context_precision": avg(all_context_precision),
        "avg_correctness": avg(all_correctness),
        "avg_latency_ms": avg(all_latencies),
        "p95_latency_ms": p95(all_latencies),
    }


def run_full_evaluation() -> None:
    """Run complete evaluation and log results to MLflow."""
    print("🚀 Starting Full Pipeline Evaluation...\n")
    print("=" * 60)

    with mlflow.start_run(run_name="full_pipeline_eval"):

        # Log parameters
        mlflow.log_param("llm_model", "llama-3.3-70b-versatile")
        mlflow.log_param("n_wildfire_tests", len(WILDFIRE_QUESTIONS))
        mlflow.log_param("n_offtopic_tests", len(OUT_OF_SCOPE_QUESTIONS))
        mlflow.log_param("n_docs_retrieved", 3)
        mlflow.log_param("vector_store", "ChromaDB")
        mlflow.log_param("embedding_model", "all-MiniLM-L6-v2")

        # Evaluate guardrails
        guardrail_metrics = evaluate_guardrails()
        for k, v in guardrail_metrics.items():
            mlflow.log_metric(k, v)

        # Evaluate RAG
        rag_metrics = evaluate_rag()
        for k, v in rag_metrics.items():
            mlflow.log_metric(k, v)

        # Summary
        print("\n" + "=" * 60)
        print("✅ EVALUATION SUMMARY")
        print("=" * 60)
        print(
            f"  Guardrail Precision  : {guardrail_metrics['guardrail_precision']:.2f}"
        )
        print(f"  Wildfire Recall      : {guardrail_metrics['wildfire_recall']:.2f}")
        print(
            f"  Off-topic Block Rate : {guardrail_metrics['offtopic_block_rate']:.2f}"
        )
        print(f"  Avg Faithfulness     : {rag_metrics['avg_faithfulness']:.2f}")
        print(f"  Avg Relevancy        : {rag_metrics['avg_relevancy']:.2f}")
        print(f"  Avg Latency (ms)     : {rag_metrics['avg_latency_ms']:.0f}")
        print(f"  Avg Context Precision: {rag_metrics['avg_context_precision']:.2f}")
        print(f"  Avg Correctness      : {rag_metrics['avg_correctness']:.2f}")
        print(f"  Avg Latency (ms)     : {rag_metrics['avg_latency_ms']:.0f}")
        print(f"  P95 Latency (ms)     : {rag_metrics['p95_latency_ms']:.0f}")
        print("=" * 60)
        print("✅ All metrics logged to MLflow")


if __name__ == "__main__":
    run_full_evaluation()
