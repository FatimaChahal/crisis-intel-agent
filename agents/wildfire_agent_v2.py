import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langfuse.callback import CallbackHandler
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from agents.tools.weather_tool import get_weather_conditions
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
    weather_data: str
    stop_reason: str


def create_llm() -> ChatGroq:
    """Create and return a Groq LLM instance."""
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0)


def classifier_agent(state: WildfireState) -> WildfireState:
    """Classify the question: relevance, language, and type."""
    llm = create_llm()
    history_text = (
        f"\nConversation history:\n{state['history']}" if state.get("history") else ""
    )

    response = llm.invoke(
        f"""You are a classifier for a wildfire analysis system.
Analyze the following question and respond ONLY in this exact format:
RELEVANT: YES or NO
LANGUAGE: (language name in English, e.g. French, English, Arabic, Spanish)
TYPE: stats OR case OR recommendation OR comparison
{history_text}
Question: {state['question']}""",
        config={"callbacks": [langfuse_handler]},
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
        "question_type": question_type,
    }


def weather_node(state: WildfireState) -> WildfireState:
    """Fetch current weather conditions using MCP weather tool."""
    llm = create_llm()
    location = llm.invoke(
        f"""Extract the location (city or country) from this question.
Return ONLY the location name, nothing else.
If no specific location, return 'france'.
Question: {state['question']}"""
    ).content.strip()

    weather = get_weather_conditions.invoke(location)
    return {**state, "weather_data": weather}


def retriever_agent(state: WildfireState) -> WildfireState:
    """Retrieve similar wildfire events from ChromaDB and validate quality."""
    if not state["is_relevant"]:
        return {**state, "context": [], "metadata": []}

    docs, metas = search_wildfires(state["question"], n=3)
    best_score = max((m.get("risk_score", 0) for m in metas), default=0)
    if best_score < 0.001:
        return {**state, "context": [], "metadata": [], "stop_reason": "no_results"}

    return {**state, "context": docs, "metadata": metas, "stop_reason": ""}


def analyst_agent(state: WildfireState) -> WildfireState:
    """Analyze retrieved wildfire data and compute statistics."""
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

Be precise and data-driven.""",
        config={"callbacks": [langfuse_handler]},
    ).content

    return {**state, "analysis": analysis}


def responder_agent(state: WildfireState) -> WildfireState:
    """Generate final answer with references in user's language."""
    llm = create_llm()

    # Build references
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

    # Weather section
    weather_section = ""
    if state.get("weather_data"):
        weather_section = f"\nCURRENT WEATHER CONDITIONS (real-time MCP tool):\n{state['weather_data']}\n"

    response = llm.invoke(
        f"""You are a wildfire crisis expert assistant.
Answer in {state['language']}.
Cite sources using [1], [2], [3] when referencing specific events.
Do NOT invent data not present in the analysis.
{weather_section}
ANALYSIS:
{state['analysis']}

RAW DATA REFERENCES:
{references_text}

QUESTION: {state['question']}
QUESTION TYPE: {state['question_type']}

Structure your answer with:
- Direct answer to the question
- Key findings with source citations [1], [2], [3]
- Current weather conditions and risk assessment (if available)
- Recommended actions
- Risk assessment""",
        config={"callbacks": [langfuse_handler]},
    ).content

    final_answer = f"{response}\n\n📚 **Sources :**\n{references_text}"
    if state.get("weather_data"):
        final_answer += f"\n\n🌤️ **Météo temps réel :**\n{state['weather_data']}"

    return {**state, "answer": final_answer, "references": references_text}


def stop_node(state: WildfireState) -> WildfireState:
    """Handle out-of-scope or no-result cases."""
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
            "No sufficiently similar wildfire events found.\n"
            "Please try with different keywords (country, year, season, region...)."
        )
    return {**state, "answer": msg}


def needs_weather(state: WildfireState) -> str:
    """Decide if weather data is needed."""
    question_lower = state["question"].lower()
    weather_triggers = [
        "today",
        "now",
        "current",
        "risk",
        "danger",
        "aujourd'hui",
        "maintenant",
        "risque",
        "actuellement",
        "conditions",
        "météo",
        "weather",
        "temperature",
    ]
    if any(t in question_lower for t in weather_triggers):
        return "weather"
    return "retrieve"


def route_after_classifier(state: WildfireState) -> str:
    """Route after classifier: weather, retrieve or stop."""
    if not state["is_relevant"]:
        return "stop"
    return needs_weather(state)


def route_after_retriever(state: WildfireState) -> str:
    """Route after retriever: analyze or stop."""
    if not state["context"] or state.get("stop_reason"):
        return "stop"
    return "analyze"


def build_wildfire_agent_v2():
    """Build the professional 4-agent Wildfire LangGraph with MCP weather tool."""
    graph = StateGraph(WildfireState)

    graph.add_node("classifier", classifier_agent)
    graph.add_node("weather", weather_node)
    graph.add_node("retriever", retriever_agent)
    graph.add_node("analyst", analyst_agent)
    graph.add_node("responder", responder_agent)
    graph.add_node("stop", stop_node)

    graph.add_edge(START, "classifier")
    graph.add_conditional_edges(
        "classifier",
        route_after_classifier,
        {"weather": "weather", "retrieve": "retriever", "stop": "stop"},
    )
    graph.add_edge("weather", "retriever")
    graph.add_conditional_edges(
        "retriever", route_after_retriever, {"analyze": "analyst", "stop": "stop"}
    )
    graph.add_edge("analyst", "responder")
    graph.add_edge("responder", END)
    graph.add_edge("stop", END)

    return graph.compile()


def run_agent_v2(question: str, history: list = None) -> str:
    """Run the professional wildfire agent v2."""
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

    result = agent.invoke(
        {
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
            "weather_data": "",
            "stop_reason": "",
        }
    )

    latency = (time.time() - start) * 1000
    track_rag_run(
        query=question,
        n_results=3,
        model_name="llama-3.3-70b-versatile",
        latency_ms=latency,
        n_docs_retrieved=len(result["context"]),
        response_length=len(result["answer"]),
    )

    return result["answer"]


if __name__ == "__main__":
    print("=== Test 1 — Incendies France ===")
    print(run_agent_v2("Quels sont les plus grands incendies en France ?"))
    print("\n=== Test 2 — Hors sujet ===")
    print(run_agent_v2("Quel est le meilleur restaurant à Paris ?"))
    print("\n=== Test 3 — Risque actuel Gironde ===")
    print(run_agent_v2("Y a-t-il un risque d'incendie aujourd'hui en Gironde ?"))
