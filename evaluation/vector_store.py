import chromadb
from chromadb.utils import embedding_functions

from data.gold.crisis_reports import CRISIS_REPORTS


def build_vector_store() -> chromadb.Collection:
    """
    Build and populate a ChromaDB vector store with crisis reports.

    Returns:
        A ChromaDB collection with embedded crisis documents.
    """
    client = chromadb.Client()

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    collection = client.get_or_create_collection(
        name="crisis_reports",
        embedding_function=embedding_fn,
    )

    collection.add(
        ids=[report["id"] for report in CRISIS_REPORTS],
        documents=[report["content"] for report in CRISIS_REPORTS],
    )

    print(f"✅ Vector store built with {collection.count()} documents")
    return collection


def search_similar(collection: chromadb.Collection, query: str, n: int = 2) -> list:
    """
    Search for similar crisis reports in the vector store.

    Args:
        collection: The ChromaDB collection to search.
        query: The search query (alert text).
        n: Number of results to return.

    Returns:
        List of similar documents.
    """
    results = collection.query(
        query_texts=[query],
        n_results=n,
    )
    return results["documents"][0]


if __name__ == "__main__":
    collection = build_vector_store()
    docs = search_similar(collection, "flood in germany severity orange")
    print("\n📄 Similar documents found:")
    for doc in docs:
        print(f"- {doc[:100]}...")