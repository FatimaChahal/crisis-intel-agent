from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

load_dotenv()


class AgentState(TypedDict):
    """State of the Scout Agent."""

    alerte: str
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


def analyse_alerte(state: AgentState) -> AgentState:
    """
    Analyse a crisis alert using the LLM.

    Args:
        state: The current agent state with the alert.

    Returns:
        Updated state with the analysis.
    """
    llm = create_llm()
    prompt = f"""
    You are a crisis analyst. Analyze this alert and provide:
    - Type of crisis
    - Severity level
    - Recommended action

    Alert: {state['alerte']}

    Answer in 3 bullet points.
    """
    response = llm.invoke(prompt)
    return {"alerte": state["alerte"], "analyse": response.content}


def build_agent():
    """
    Build and return the Scout Agent graph.

    Returns:
        A compiled LangGraph agent.
    """
    graph = StateGraph(AgentState)
    graph.add_node("analyse", analyse_alerte)
    graph.add_edge(START, "analyse")
    graph.add_edge("analyse", END)
    return graph.compile()


if __name__ == "__main__":
    agent = build_agent()
    result = agent.invoke({
        "alerte": "flood in germany, severity orange",
        "analyse": ""
    })
    print(result["analyse"])