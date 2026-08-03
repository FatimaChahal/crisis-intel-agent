import asyncio
import os

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool

from agents.tools.weather_tool import get_weather_conditions
from evaluation.vector_store import search_wildfires

load_dotenv()


def search_wildfire_cases(query: str) -> str:
    """
    Search for similar wildfire events in the database.

    Args:
        query: Natural language query about wildfires.

    Returns:
        String with similar wildfire cases found.
    """
    docs, metas = search_wildfires(query, n=3)
    if not docs:
        return "No similar wildfire cases found in the database."
    result = "Similar wildfire cases found:\n\n"
    for i, (doc, meta) in enumerate(zip(docs, metas)):
        result += f"[{i+1}] {doc}\n"
        result += f"    Risk score: {meta.get('risk_score', 0):.3f} | "
        result += f"Severity: {meta.get('severity', 'N/A')}\n\n"
    return result


def get_current_weather(location: str) -> str:
    """
    Get current weather conditions to assess wildfire risk.

    Args:
        location: City or country name.

    Returns:
        Current weather conditions and wildfire risk assessment.
    """
    return get_weather_conditions.invoke(location)


def build_adk_agent() -> Agent:
    """
    Build a professional ADK wildfire agent with tools.

    Returns:
        Configured ADK Agent instance.
    """
    model = LiteLlm(
        model="groq/llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
    )

    agent = Agent(
        name="wildfire_expert",
        model=model,
        description="Expert wildfire analyst for European forest fires (2016-2026).",
        instruction="""You are a wildfire crisis expert assistant specialized in European forest fires.

Your role:
1. ONLY answer questions related to wildfires, forest fires, or fire management
2. Use the search_wildfire_cases tool to find similar past events
3. Use get_current_weather tool when asked about current risk
4. Always cite your sources [1][2][3] in your answer
5. Answer in the SAME language as the question
6. If the question is NOT about wildfires, refuse politely

For every wildfire question:
- Search for similar past cases
- Provide key statistics (area, duration, severity)
- Give recommended actions based on past events
- Include risk assessment""",
        tools=[
            FunctionTool(search_wildfire_cases),
            FunctionTool(get_current_weather),
        ],
    )
    return agent


async def run_adk_agent(question: str) -> str:
    """
    Run the ADK wildfire agent on a question.

    Args:
        question: User question in any language.

    Returns:
        Agent answer string.
    """
    agent = build_adk_agent()
    session_service = InMemorySessionService()

    runner = Runner(
        agent=agent,
        app_name="crisis_intel_agent",
        session_service=session_service,
    )

    session = await session_service.create_session(
        app_name="crisis_intel_agent",
        user_id="user_1",
    )

    # ✅ Nouveau
    # ✅ Nouveau
    from google.genai.types import Content, Part

    message = Content(role="user", parts=[Part(text=question)])

    answer = ""
    async for event in runner.run_async(
        user_id="user_1",
        session_id=session.id,
        new_message=message,
    ):
        if event.is_final_response() and event.content:
            for part in event.content.parts:
                if part.text:
                    answer += part.text

    return answer if answer else "No response generated."


if __name__ == "__main__":

    async def test():
        """Test the ADK wildfire agent."""
        print("=== Test 1 — Wildfires France ===")
        result = await run_adk_agent("What were the largest wildfires in France?")
        print(result)

        print("\n=== Test 2 — Hors sujet ===")
        result = await run_adk_agent("What is the best restaurant in Paris?")
        print(result)

        print("\n=== Test 3 — Risque Gironde ===")
        result = await run_adk_agent(
            "Y a-t-il un risque d'incendie en Gironde aujourd'hui?"
        )
        print(result)

    asyncio.run(test())
