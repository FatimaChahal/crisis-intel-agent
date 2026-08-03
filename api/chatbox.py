import requests
import gradio as gr

API_URL = "http://localhost:8000"
API_KEY = "crisis-intel-secret-key-2026"


def call_api(question: str, history: list) -> str:
    """
    Call the FastAPI /analyze endpoint.

    Args:
        question: User question in any language.
        history: Chat history list.

    Returns:
        Agent answer string.
    """
    try:
        response = requests.post(
            f"{API_URL}/analyze",
            json={"question": question, "history": history},
            headers={"X-API-Key": API_KEY},
            timeout=60,
        )
        if response.status_code == 200:
            return response.json()["answer"]
        elif response.status_code == 401:
            return "❌ Authentication error — invalid API key."
        else:
            return f"❌ API error {response.status_code}: {response.text}"
    except requests.exceptions.ConnectionError:
        return "❌ Cannot connect to API — make sure FastAPI is running on port 8000."
    except requests.exceptions.Timeout:
        return "⏱️ Request timeout — the agent is taking too long to respond."
    except Exception as e:
        return f"❌ Unexpected error: {str(e)}"


def chat(message: str, history: list) -> str:
    """
    Process user message via FastAPI and return agent response.

    Args:
        message: User input message.
        history: Chat history.

    Returns:
        Agent response string with references.
    """
    if not message.strip():
        return "Please ask a question about wildfires."
    return call_api(message, history)


def launch_chatbox() -> None:
    """Launch the Gradio chatbox interface connected to FastAPI."""
    demo = gr.ChatInterface(
        fn=chat,
        title="🔥 Crisis Intel Agent — Wildfire Assistant",
        description=(
            "Ask me anything about wildfire events in Europe (2016-2026). "
            "I answer in your language with data sources and real-time weather. "
            "Powered by 18,607 real wildfire records from Copernicus EFFIS & MODIS."
        ),
        examples=[
            ["What were the largest wildfires in France?"],
            ["Quels incendies ont touché la Grèce en 2023 ?"],
            ["Y a-t-il un risque d'incendie aujourd'hui en Gironde ?"],
            ["ما هي أكبر حرائق الغابات في أوروبا؟"],
        ],
    )
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    launch_chatbox()
