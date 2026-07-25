import mlflow
from dotenv import load_dotenv

load_dotenv()

# Configure MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("crisis-intel-agent")


def track_rag_run(
    query: str,
    n_results: int,
    model_name: str,
    latency_ms: float,
    n_docs_retrieved: int,
    response_length: int,
) -> None:
    """
    Track a RAG pipeline run with MLflow.

    Args:
        query: The input query/alert.
        n_results: Number of documents retrieved.
        model_name: LLM model used.
        latency_ms: Response latency in milliseconds.
        n_docs_retrieved: Number of documents actually retrieved.
        response_length: Length of the generated response.
    """
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("query", query)
        mlflow.log_param("n_results", n_results)
        mlflow.log_param("model_name", model_name)

        # Log metrics
        mlflow.log_metric("latency_ms", latency_ms)
        mlflow.log_metric("n_docs_retrieved", n_docs_retrieved)
        mlflow.log_metric("response_length", response_length)

        print(f"✅ MLflow run tracked successfully")


if __name__ == "__main__":
    # Test tracking
    track_rag_run(
        query="flood in germany severity orange",
        n_results=2,
        model_name="llama-3.3-70b-versatile",
        latency_ms=985.0,
        n_docs_retrieved=2,
        response_length=350,
    )
