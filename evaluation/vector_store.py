import chromadb
import pandas as pd
from chromadb.utils import embedding_functions


def build_vector_store_from_gold() -> chromadb.Collection:
    """
    Build ChromaDB vector store from Gold wildfire data.

    Returns:
        ChromaDB collection with all wildfire summaries indexed.
    """
    client = chromadb.PersistentClient(path="data/gold/chroma_db")

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # Delete existing collection if exists
    try:
        client.delete_collection("wildfires")
    except Exception:
        pass

    collection = client.create_collection(
        name="wildfires",
        embedding_function=embedding_fn,
    )

    # Load Gold data
    df = pd.read_csv("data/gold/modis_fires_gold.csv")
    print(f"📦 Indexing {len(df)} wildfire records...")

    # Index in batches of 500
    batch_size = 500
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i + batch_size]
        collection.add(
            ids=[str(idx) for idx in batch.index],
            documents=batch["summary"].tolist(),
            metadatas=[{
                "country_code": str(row["country_code"]),
                "year": int(row["year"]),
                "burnt_area_ha": float(row["burnt_area_ha"]),
                "severity": str(row["severity"]),
                "season": str(row["season"]),
                "risk_score": float(row["risk_score"]),
                "duration_days": int(row["duration_days"]),
                "dominant_vegetation": str(row["dominant_vegetation"]),
            } for _, row in batch.iterrows()]
        )
        print(f"  ✅ Batch {i // batch_size + 1} indexed ({min(i + batch_size, len(df))}/{len(df)})")

    print(f"\n✅ Vector store ready: {collection.count()} documents")
    return collection


def get_vector_store() -> chromadb.Collection:
    """
    Get existing ChromaDB vector store.

    Returns:
        ChromaDB collection with wildfire summaries.
    """
    client = chromadb.PersistentClient(path="data/gold/chroma_db")
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    return client.get_collection(
        name="wildfires",
        embedding_function=embedding_fn
    )


def search_wildfires(query: str, n: int = 3) -> list:
    """
    Search for similar wildfire events.

    Args:
        query: Natural language question from user.
        n: Number of results to return.

    Returns:
        List of similar wildfire summaries.
    """
    collection = get_vector_store()
    results = collection.query(
        query_texts=[query],
        n_results=n
    )
    return results["documents"][0], results["metadatas"][0]


if __name__ == "__main__":
    build_vector_store_from_gold()
    # Test search
    print("\n🔍 Test search: 'large wildfire France summer'")
    docs, metas = search_wildfires("large wildfire France summer")
    for doc, meta in zip(docs, metas):
        print(f"\n{doc}")
        print(f"Metadata: {meta}")