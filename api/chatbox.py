import gradio as gr
from agents.wildfire_agent import run_agent


def chat(message: str, history: list) -> str:
    """
    Process user message and return agent response.

    Args:
        message: User input message.
        history: Chat history.

    Returns:
        Agent response string.
    """
    if not message.strip():
        return "Please ask a question about wildfires."
    return run_agent(message, history)


def launch_chatbox() -> None:
    """Launch the Gradio chatbox interface."""
    demo = gr.ChatInterface(
        fn=chat,
        title="🔥 Crisis Intel Agent — Wildfire Assistant",
        description=(
            "Ask me anything about wildfire events in Europe (2016-2026). "
            "I can help you find similar past events and recommend actions based on real data."
        ),
        examples=[
            ["What were the largest wildfires in France?"],
            ["Tell me about wildfires in Greece in summer 2023"],
            ["What happened during wildfires in Gironde?"],
            ["Which country had the most severe wildfires in 2022?"],
        ],
    )
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    launch_chatbox()