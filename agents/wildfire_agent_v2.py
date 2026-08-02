import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langfuse.callback import CallbackHandler
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from evaluation.vector_store import search_wildfires
from mlflow_tracking.tracker import track_rag_run

load_dotenv()

langfuse_handler = CallbackHandler()


class WildfireState(TypedDict):
    """State shared between all agents."""
    question: str
    history: str
    language: str
    question_type: str
    is_relevant: bool
    context: list
    metadata: list
    analysis: str
    answer: str
    references: str
    stop_reason: str


def create_llm() -> ChatGroq:
    """
    Create and return a Groq LLM instance.

    Returns:
        A ChatGroq instance using Llama 3.3 70B.
    """
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


# ─────────────────────────────────────────
# AGENT 1 — CLASSIFIER
# ─────────────────────────────────────────
def classifier_agent(state: WildfireState) -> WildfireState:
    """
    Classify the question: relevance, language, and type.

    Args:
        state: Current agent state.

    Returns:
        Updated state with language, question_type and is_relevant.
    """
    llm = create_llm()
    history_text = f"\nConversation history:\n{state['history']}" if state.get("history") else ""

    response = llm.invoke(
        f"""You are a classifier for a wildfire analysis system.
Analyze the following question and respond ONLY in this exact format:
RELEVANT: YES or NO
LANGUAGE: (language name in English, e.g. French, English, Arabic, Spanish)
TYPE: stats OR case OR recommendation OR comparison

Definitions:
- stats: questions about numbers, areas, counts
- case: questions about specific events or places
- recommendation: questions about what to do
- comparison: questions comparing countries, years, regions
{history_text}
Question: {state['question']}""",
        config={"callbacks": [langfuse_handler]}
    ).content.strip()

    lines = response.split("\n")
    is_relevant = False
    language = "English"
    question_type = "case"

    for line in lines:
        if "RELEVANT:" in line:
            is_relevant = "YES" in line.upper()
        elif "LANGUAGE:" in line:
            language = line.split(":", 1)[1].strip()
        elif "TYPE:" in line:
            question_type = line.split(":", 1)[1].strip().lower()

    return {
        **state,
        "is_relevant": is_relevant,
        "language": language,
        "question_type": question_type
    }


# ─────────────────────────────────────────
# AGENT 2 — RETRIEVER
# ─────────────────────────────────────────
def retriever_agent(state: WildfireState) -> WildfireState:
    """
    Retrieve similar wildfire events from ChromaDB and validate quality.

    Args:
        state: Current agent state.

    Returns:
        Updated state with context and metadata.
    """
    if not state["is_relevant"]:
        return {**state, "context": [], "metadata": []}

    docs, metas = search_wildfires(state["question"], n=3)

    # Validate quality — check if best score is meaningful
    best_score = max((m.get("risk_score", 0) for m in metas), default=0)
    if best_score < 0.001:
        return {**state, "context": [], "metadata": [], "stop_reason": "no_results"}

    return {**state, "context": docs, "metadata": metas, "stop_reason": ""}


# ─────────────────────────────────────────
# AGENT 3 — ANALYST
# ─────────────────────────────────────────
def analyst_agent(state: WildfireState) -> WildfireState:
    """
    Analyze retrieved wildfire data and compute statistics.

    Args:
        state: Current agent state with retrieved context.

    Returns:
        Updated state with structured analysis.
    """
    if not state["context"]:
        return {**state, "analysis": ""}

    llm = create_llm()
    context_text = "\n\n".join(
        f"[{i+1}] {doc}" for i, doc in enumerate(state["context"])
    )

    analysis = llm.invoke(
        f"""You are a wildfire data analyst.
Analyze the following wildfire records and extract key insights.

WILDFIRE DATA:
{context_text}

Question type: {state['question_type']}

Extract and compute:
1. Key statistics (min, max, average burnt area, duration)
2. Most affected regions/countries
3. Seasonal patterns
4. Dominant vegetation types
5. Risk level summary

Be precise and data-driven. Use numbers from the data.""",
        config={"callbacks": [langfuse_handler]}
    ).content

    return {**state, "analysis": analysis}


# ─────────────────────────────────────────
# AGENT 4 — RESPONDER
# ─────────────────────────────────────────
def responder_agent(state: WildfireState) -> WildfireState:
    """
    Generate final answer with references in user's language.

    Args:
        state: Current agent state with analysis and context.

    Returns:
        Updated state with final answer and formatted references.
    """
    llm = create_llm()

    # Build references section
    refs = []
    for i, meta in enumerate(state["metadata"]):
        ref = (
            f"[{i+1}] 🔥 {meta.get('country_code', 'N/A')} — "
            f"{meta.get('season', '').capitalize()} {meta.get('year', 'N/A')}\n"
            f"    Superficie : {int(meta.get('burnt_area_ha', 0))} ha | "
            f"Durée : {meta.get('duration_days', 'N/A')} jour(s) | "
            f"Sévérité : {meta.get('severity', 'N/A')}\n"
            f"    Score de risque : {meta.get('risk_score', 0):.3f} | "
            f"Végétation : {meta.get('dominant_vegetation', 'N/A')}"
        )
        refs.append(ref)
    references_text = "\n".join(refs)

    response = llm.invoke(
        f"""You are a wildfire crisis expert assistant.
Generate a clear and structured response to the user's question.
Answer in {state['language']}.
Cite sources using [1], [2], [3] when referencing specific events.
Do NOT invent data not present in the analysis.

ANALYSIS:
{state['analysis']}

RAW DATA REFERENCES:
{references_text}

QUESTION: {state['question']}
QUESTION TYPE: {state['question_type']}

Structure your answer with:
- Direct answer to the question
- Key findings with source citations [1], [2], [3]
- Recommended actions (if relevant)
- Risk assessment""",
        config={"callbacks": [langfuse_handler]}
    ).content

    final_answer = f"{response}\n\n📚 **Sources :**\n{references_text}"

    return {**state, "answer": final_answer, "references": references_text}


# ─────────────────────────────────────────
# STOP NODE
# ─────────────────────────────────────────
def stop_node(state: WildfireState) -> WildfireState:
    """
    Handle out-of-scope or no-result cases.

    Args:
        state: Current agent state.

    Returns:
        Updated state with appropriate stop message.
    """
    if not state["is_relevant"]:
        msg = (
            "❌ Je suis spécialisé uniquement dans l'analyse des incendies de forêt.\n"
            "I am specialized exclusively in wildfire analysis.\n"
            "أنا متخصص فقط في تحليل حرائق الغابات.\n\n"
            "Please ask a question related to wildfires, burnt areas, or fire management."
        )
    else:
        msg = (
            "⚠️ Aucun événement suffisamment similaire trouvé dans la base de données.\n"
            "No sufficiently similar wildfire events found in the database.\n"
            "Please try with different keywords (country, year, season, region...)."
        )
    return {**state, "answer": msg}


# ─────────────────────────────────────────
# ROUTING
# ─────────────────────────────────────────
def route_after_classifier(state: WildfireState) -> str:
    """Route after classifier: retrieve or stop."""
    return "retrieve" if state["is_relevant"] else "stop"


def route_after_retriever(state: WildfireState) -> str:
    """Route after retriever: analyze or stop."""
    if not state["context"] or state.get("stop_reason"):
        return "stop"
    return "analyze"


# ─────────────────────────────────────────
# BUILD GRAPH
# ─────────────────────────────────────────
def build_wildfire_agent_v2():
    """
    Build and compile the professional 4-agent Wildfire LangGraph.

    Returns:
        Compiled LangGraph agent with 4 specialized agents.
    """
    graph = StateGraph(WildfireState)

    graph.add_node("classifier", classifier_agent)
    graph.add_node("retriever", retriever_agent)
    graph.add_node("analyst", analyst_agent)
    graph.add_node("responder", responder_agent)
    graph.add_node("stop", stop_node)

    graph.add_edge(START, "classifier")
    graph.add_conditional_edges("classifier", route_after_classifier, {
        "retrieve": "retriever",
        "stop": "stop"
    })
    graph.add_conditional_edges("retriever", route_after_retriever, {
        "analyze": "analyst",
        "stop": "stop"
    })
    graph.add_edge("analyst", "responder")
    graph.add_edge("responder", END)
    graph.add_edge("stop", END)

    return graph.compile()


# ─────────────────────────────────────────
# RUN AGENT
# ─────────────────────────────────────────
def run_agent_v2(question: str, history: list = None) -> str:
    """
    Run the professional wildfire agent v2.

    Args:
        question: User question in any language.
        history: Chat history list.

    Returns:
        Final answer with references.
    """
    history_text = ""
    if history:
        for h in history[-3:]:
            if isinstance(h, dict):
                if h.get("role") == "user":
                    history_text += f"User: {h.get('content', '')}\n"
                elif h.get("role") == "assistant":
                    history_text += f"Assistant: {h.get('content', '')}\n"
            elif isinstance(h, (list, tuple)) and len(h) == 2:
                history_text += f"User: {h[0]}\nAssistant: {h[1]}\n"

    agent = build_wildfire_agent_v2()
    start = time.time()

    result = agent.invoke({
        "question": question,
        "history": history_text,
        "language": "English",
        "question_type": "case",
        "is_relevant": False,
        "context": [],
        "metadata": [],
        "analysis": "",
        "answer": "",
        "references": "",
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
    print("=== Test 1 — Incendies France ===")
    print(run_agent_v2("Quels sont les plus grands incendies en France ?"))
    print("\n=== Test 2 — Hors sujet ===")
    print(run_agent_v2("Quel est le meilleur restaurant à Paris ?"))