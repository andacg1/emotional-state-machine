"""

"""
from __future__ import annotations

import os

from langchain_openai import ChatOpenAI

from agent.npc import NPCResponse


def get_llm():
    """Initialize and return the language model."""

    # Use LMStudio to run local LLM model, it provides OpenAI compatible API
    base_url = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
    model_name = os.getenv("LMSTUDIO_MODEL", "gemma-4-e4b")
    api_key = os.getenv("LMSTUDIO_API_KEY", "lm-studio")

    try:
        return ChatOpenAI(
            base_url=base_url,
            api_key=api_key,
            model=model_name,
            temperature=0.7
        )
    except Exception as e:
        print(f"Warning: Could not connect to LMStudio at {base_url}")
        print("Please ensure LMStudio is running and the server is started.")
        print(f"Error: {e}")
        return None


llm = get_llm()
summarization_model = llm.bind(max_tokens=128)
structured_llm = llm.with_structured_output(NPCResponse)
