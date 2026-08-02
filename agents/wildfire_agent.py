import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from evaluation.vector_store import search_wildfires
from mlflow_tracking.tracker import track_rag_run

load_dotenv()

WILDFIRE_KEYWORDS = [
    "fire", "wildfire", "forest", "burn", "burnt", "hectare",
    "smoke", "evacuation", "incendie", "feu", "forêt", "brûlé",
    "flamme", "flame", "blaze", "country", "france", "spain",
    "greece", "turkey", "portugal", "gironde", "severity",
    "summer", "drought", "vegetation", "conifer"
]


class WildfireState(TypedDict):
    """State of the Wildfire Agent."""
    question: str
    history: str
    is_relevant: bool
    context: list
    metadata: list
    answer: str
    stop_reason: str


def create_llm() -> ChatGroq:
    """
    Create and return a Groq LLM instance.

    Returns:
        A ChatGroq instance using Llama 3.3 70B.
    """
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


def guardrail(state: WildfireState) -> WildfireState:
    """
    Check if the question is related to wildfires using LLM — any language.
    Takes conversation history into account.

    Args:
        state: Current agent state with the question.

    Returns:
        Updated state with is_relevant flag.
    """
    llm = create_llm()

    history_text = ""
    if state.get("history"):
        history_text = f"\nConversation history:\n{state['history']}\n"

    check = llm.invoke(
        f"""You are a content filter for a wildfire analysis system.
Determine if the following question is related to wildfires, forest fires,
burnt areas, fire management, or fire crisis events.
The question can be in ANY language.
Consider the conversation history to understand the context.
{history_text}
Current question: {state['question']}

Answer ONLY with YES or NO."""
    ).content.strip().upper()

    is_relevant = "YES" in check
    return {**state, "is_relevant": is_relevant}


def retrieve(state: WildfireState) -> WildfireState:
    """
    Retrieve similar wildfire events from ChromaDB.

    Args:
        state: Current agent state.

    Returns:
        Updated state with retrieved context.
    """
    if not state["is_relevant"]:
        return {**state, "context": [], "metadata": []}

    docs, metas = search_wildfires(state["question"], n=3)
    return {**state, "context": docs, "metadata": metas}


def check_stop(state: WildfireState) -> str:
    """
    Decide whether to answer or stop based on context quality.

    Args:
        state: Current agent state.

    Returns:
        Next node name: 'answer' or 'stop'.
    """
    if not state["is_relevant"]:
        return "stop"
    if not state["context"]:
        return "stop"
    # Stop if best result has very low risk score
    best_score = max(m.get("risk_score", 0) for m in state["metadata"])
    if best_score < 0.01:
        return "stop"
    return "answer"


def answer(state: WildfireState) -> WildfireState:
    """
    Generate a contextualized answer using retrieved wildfire data.

    Args:
        state: Current agent state with context.

    Returns:
        Updated state with generated answer.
    """
    llm = create_llm()
    context_text = "\n\n".join(state["context"])

    prompt = f"""You are a wildfire crisis expert assistant.
Use ONLY the wildfire data below to answer the question.
Do NOT invent information not present in the data.
Answer in the SAME LANGUAGE as the question.

WILDFIRE DATA:
{context_text}

QUESTION: {state['question']}

Provide a structured answer with:
- Similar past wildfire events found
- Key statistics (area, duration, severity)
- Recommended actions based on past events
- Risk level assessment

If the data is insufficient, say so clearly in the same language as the question."""

    response = llm.invoke(prompt).content
    return {**state, "answer": response, "stop_reason": ""}


def stop(state: WildfireState) -> WildfireState:
    """
    Handle out-of-scope or no-result cases.

    Args:
        state: Current agent state.

    Returns:
        Updated state with stop message.
    """
    if not state["is_relevant"]:
        msg = (
            "❌ Je suis spécialisé uniquement dans l'analyse des incendies de forêt. "
            "/ I am specialized exclusively in wildfire analysis. "
            "/ أنا متخصص فقط في تحليل حرائق الغابات.\n\n"
            "Please ask a question related to wildfires, burnt areas, or fire management."
        )
    else:
        msg = ("⚠️ I could not find sufficiently similar wildfire events in my database "
               "for your query. Please try with different keywords (country, year, season...).")

    return {**state, "answer": msg, "stop_reason": "out_of_scope" if not state["is_relevant"] else "no_results"}


def build_wildfire_agent():
    """
    Build and compile the Wildfire LangGraph agent.

    Returns:
        Compiled LangGraph agent.
    """
    graph = StateGraph(WildfireState)

    graph.add_node("guardrail_node", guardrail)
    graph.add_node("retrieve_node", retrieve)
    graph.add_node("generate_node", answer)
    graph.add_node("stop_node", stop)

    graph.add_edge(START, "guardrail_node")
    graph.add_edge("guardrail_node", "retrieve_node")
    graph.add_conditional_edges("retrieve_node", check_stop, {
        "answer": "generate_node",
        "stop": "stop_node"
    })
    graph.add_edge("generate_node", END)
    graph.add_edge("stop_node", END)

    return graph.compile()


def run_agent(question: str, history: list = None) -> str:
    """
    Run the wildfire agent on a question.

    Args:
        question: User question in natural language.
        history: Chat history list.

    Returns:
        Agent answer string.
    """
    history_text = ""
    if history:
        for h in history[-3:]:
            if isinstance(h, dict):
                user_msg = h.get("role") == "user" and h.get("content", "")
                assistant_msg = h.get("role") == "assistant" and h.get("content", "")
                if user_msg:
                    history_text += f"User: {h.get('content', '')}\n"
                if assistant_msg:
                    history_text += f"Assistant: {h.get('content', '')}\n"
            elif isinstance(h, (list, tuple)) and len(h) == 2:
                history_text += f"User: {h[0]}\nAssistant: {h[1]}\n"

    agent = build_wildfire_agent()
    start = time.time()

    result = agent.invoke({
        "question": question,
        "history": history_text,
        "is_relevant": False,
        "context": [],
        "metadata": [],
        "answer": "",
        "stop_reason": ""
    })

    latency = (time.time() - start) * 1000
    track_rag_run(
        query=question,
        n_results=3,
        model_name="llama-3.3-70b-versatile",
        latency_ms=latency,
        n_docs_retrieved=len(result["context"]),
        response_length=len(result["answer"])
    )

    return result["answer"]


if __name__ == "__main__":
    print("Test 1 — Question sur incendies:")
    print(run_agent("What happened during large wildfires in France?"))
    print("\nTest 2 — Question hors sujet:")
    print(run_agent("What is the weather in Paris today?"))