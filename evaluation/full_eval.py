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
WILDFIRE_QUESTIONS = [
    {
        "question": "What happened during the large wildfire in Greece in 2023?",
        "expected_topic": "wildfire",
        "ground_truth": "In 2023, Greece experienced a severe wildfire with 96610 hectares burnt during summer."
    },
    {
        "question": "Which wildfires occurred in Portugal in 2017?",
        "expected_topic": "wildfire",
        "ground_truth": "Portugal 2017 had severe wildfires burning over 60000 hectares in Viseu Dão Lafões region."
    },
    {
        "question": "What is the risk of wildfire today in Gironde?",
        "expected_topic": "wildfire",
        "ground_truth": "Gironde has high wildfire risk in summer due to heat and drought conditions."
    },
    {
        "question": "Tell me about wildfires in Turkey in 2021",
        "expected_topic": "wildfire",
        "ground_truth": "Turkey 2021 had a severe wildfire in Antalya burning 54769 hectares of conifer forest."
    },
    {
        "question": "Quels incendies ont touché la France en été ?",
        "expected_topic": "wildfire",
        "ground_truth": "France summer wildfires mainly affected Bouches-du-Rhône with conifer vegetation."
    },
]

OUT_OF_SCOPE_QUESTIONS = [
    {"question": "What is the best restaurant in Paris?", "expected_topic": "off_topic"},
    {"question": "How do I make chocolate cake?", "expected_topic": "off_topic"},
    {"question": "What is the capital of Germany?", "expected_topic": "off_topic"},
    {"question": "Tell me about football results", "expected_topic": "off_topic"},
    {"question": "What is the stock price of Apple?", "expected_topic": "off_topic"},
]


def score_with_llm(question: str, context: str, answer: str) -> dict:
    """
    Score RAG response using LLM as judge.

    Args:
        question: Original question.
        context: Retrieved context documents.
        answer: Generated answer.

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
1. FAITHFULNESS (0.0-1.0): Is the answer based on the context? 1.0 = fully grounded
2. RELEVANCY (0.0-1.0): Does the answer address the question? 1.0 = perfectly relevant

Respond ONLY in this format:
FAITHFULNESS: <score>
RELEVANCY: <score>"""

    response = llm.invoke(prompt).content
    scores = {"faithfulness": 0.0, "relevancy": 0.0}
    for line in response.strip().split("\n"):
        if "FAITHFULNESS:" in line:
            try:
                scores["faithfulness"] = float(line.split(":")[1].strip())
            except ValueError:
                pass
        elif "RELEVANCY:" in line:
            try:
                scores["relevancy"] = float(line.split(":")[1].strip())
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
        result = agent.invoke({
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
            "stop_reason": ""
        })
        if result["is_relevant"]:
            wildfire_passed += 1
            correct += 1
        print(f"  ✅ [{tc['expected_topic']}] {tc['question'][:50]}... → {'PASS' if result['is_relevant'] else 'FAIL'}")

    # Off-topic questions should be blocked
    offtopic_blocked = 0
    for tc in OUT_OF_SCOPE_QUESTIONS:
        result = agent.invoke({
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
            "stop_reason": ""
        })
        if not result["is_relevant"]:
            offtopic_blocked += 1
            correct += 1
        print(f"  ❌ [{tc['expected_topic']}] {tc['question'][:50]}... → {'BLOCKED ✅' if not result['is_relevant'] else 'NOT BLOCKED ❌'}")

    precision = correct / total
    recall_wildfire = wildfire_passed / len(WILDFIRE_QUESTIONS)
    recall_offtopic = offtopic_blocked / len(OUT_OF_SCOPE_QUESTIONS)

    print(f"\n  Guardrail Precision     : {precision:.2f}")
    print(f"  Wildfire Recall         : {recall_wildfire:.2f}")
    print(f"  Off-topic Block Rate    : {recall_offtopic:.2f}")

    return {
        "guardrail_precision": precision,
        "wildfire_recall": recall_wildfire,
        "offtopic_block_rate": recall_offtopic
    }


def evaluate_rag() -> dict:
    """
    Evaluate RAG pipeline with faithfulness and relevancy scores.

    Returns:
        Dictionary with RAG evaluation metrics.
    """
    print("\n📊 Evaluating RAG Pipeline...")

    all_faithfulness = []
    all_relevancy = []
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

        # Generate answer
        from langchain_groq import ChatGroq
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        answer = llm.invoke(
            f"You are a wildfire expert. Answer based on this context:\n{context}\n\nQuestion: {tc['question']}"
        ).content

        # Score
        scores = score_with_llm(tc["question"], context, answer)
        all_faithfulness.append(scores["faithfulness"])
        all_relevancy.append(scores["relevancy"])

        print(f"  📄 {tc['question'][:50]}...")
        print(f"     Faithfulness: {scores['faithfulness']:.2f} | Relevancy: {scores['relevancy']:.2f} | Latency: {latency:.0f}ms")

    avg_faith = sum(all_faithfulness) / len(all_faithfulness) if all_faithfulness else 0
    avg_rel = sum(all_relevancy) / len(all_relevancy) if all_relevancy else 0
    avg_latency = sum(all_latencies) / len(all_latencies) if all_latencies else 0

    print(f"\n  Avg Faithfulness   : {avg_faith:.2f}")
    print(f"  Avg Relevancy      : {avg_rel:.2f}")
    print(f"  Avg Latency (ms)   : {avg_latency:.0f}")

    return {
        "avg_faithfulness": avg_faith,
        "avg_relevancy": avg_rel,
        "avg_latency_ms": avg_latency
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
        print(f"  Guardrail Precision  : {guardrail_metrics['guardrail_precision']:.2f}")
        print(f"  Wildfire Recall      : {guardrail_metrics['wildfire_recall']:.2f}")
        print(f"  Off-topic Block Rate : {guardrail_metrics['offtopic_block_rate']:.2f}")
        print(f"  Avg Faithfulness     : {rag_metrics['avg_faithfulness']:.2f}")
        print(f"  Avg Relevancy        : {rag_metrics['avg_relevancy']:.2f}")
        print(f"  Avg Latency (ms)     : {rag_metrics['avg_latency_ms']:.0f}")
        print("=" * 60)
        print("✅ All metrics logged to MLflow")


if __name__ == "__main__":
    run_full_evaluation()