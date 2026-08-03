# import gradio as gr
# from agents.wildfire_agent import run_agent
import gradio as gr
from agents.wildfire_agent_v2 import run_agent_v2


def chat(message: str, history: list) -> str:
    """
    Process user message and return agent v2 response.

    Args:
        message: User input message.
        history: Chat history.

    Returns:
        Agent response string with references.
    """
    if not message.strip():
        return "Please ask a question about wildfires."
    return run_agent_v2(message, history)


def launch_chatbox() -> None:
    """Launch the Gradio chatbox interface."""
    demo = gr.ChatInterface(
        fn=chat,
        title="🔥 Crisis Intel Agent — Wildfire Assistant",
        description=(
            "Ask me anything about wildfire events in Europe (2016-2026). "
            "I answer in your language with data sources. "
            "Powered by 18,607 real wildfire records from Copernicus EFFIS & MODIS."
        ),
        examples=[
            ["What were the largest wildfires in France?"],
            ["Quels incendies ont touché la Grèce en 2023 ?"],
            ["Which country had the most severe wildfires in 2022?"],
            ["ما هي أكبر حرائق الغابات في أوروبا؟"],
        ],
    )
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    launch_chatbox()
