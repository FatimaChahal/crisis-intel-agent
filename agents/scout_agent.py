from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict
from langfuse.langchain import CallbackHandler


from evaluation.vector_store import build_vector_store, search_similar

load_dotenv()

# Build vector store once
collection = build_vector_store()

# Langfuse callback
langfuse_handler = CallbackHandler()


class AgentState(TypedDict):
    """State of the Scout Agent."""

    alerte: str
    contexte: str
    analyse: str


def create_llm() -> ChatGroq:
    """
    Create and return a Groq LLM instance.

    Returns:
        A ChatGroq instance using Llama 3.3 70B.
    """
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
    )


def retrieve_context(state: AgentState) -> AgentState:
    """
    Retrieve similar crisis reports from vector store.

    Args:
        state: Current agent state with the alert.

    Returns:
        Updated state with retrieved context.
    """
    docs = search_similar(collection, state["alerte"])
    contexte = "\n\n".join(docs)
    return {**state, "contexte": contexte}


def analyse_alerte(state: AgentState) -> AgentState:
    """
    Analyse a crisis alert using LLM + retrieved context.

    Args:
        state: Current agent state with alert and context.

    Returns:
        Updated state with the analysis.
    """
    llm = create_llm()
    prompt = f"""
    You are a crisis analyst. Use the context below to analyze the alert.

    CONTEXT (similar past crises):
    {state['contexte']}

    ALERT TO ANALYZE:
    {state['alerte']}

    Provide:
    - Type of crisis
    - Severity level
    - Recommended actions (based on past crises)

    Answer in 3 bullet points.
    """
    response = llm.invoke(prompt, config={"callbacks": [langfuse_handler]})
    return {**state, "analyse": response.content}


def build_agent():
    """
    Build and return the Scout Agent graph with RAG.

    Returns:
        A compiled LangGraph agent.
    """
    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve_context)
    graph.add_node("analyse", analyse_alerte)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "analyse")
    graph.add_edge("analyse", END)
    return graph.compile()


if __name__ == "__main__":
    agent = build_agent()
    result = agent.invoke({
        "alerte": "flood in germany severity orange",
        "contexte": "",
        "analyse": "",
    })
    print("\n🔍 Analysis:")
    print(result["analyse"])
    print("\n✅ Trace sent to Langfuse!")