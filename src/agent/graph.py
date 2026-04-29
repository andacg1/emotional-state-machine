"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, List, Sequence, Annotated

from langchain_core.messages import AnyMessage, SystemMessage
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime
from typing_extensions import TypedDict
from langgraph.graph.message import MessagesState, add_messages

import getpass
import os

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")
os.environ["PYTHONFROZENMODULES"] = "off"

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-5-nano",
    # stream_usage=True,
    # temperature=None,
    # max_tokens=None,
    # timeout=None,
    # reasoning_effort="low",
    # max_retries=2,
    # api_key="...",  # If you prefer to pass api key in directly
    # base_url="...",
    # organization="...",
    # other params...
)

# TODO: set up assistant

class Context(TypedDict):
    """Context parameters for the agent.

    Set these when creating assistants OR when invoking the graph.
    See: https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/
    """
    model_name: str



@dataclass
class State(MessagesState):
    """Input state for the agent.

    Defines the initial structure of incoming data.
    See: https://langchain-ai.github.io/langgraph/concepts/low_level/#state
    """

    emotion: Literal["relaxed", "nervous", "panicking", "angry", "upset", "depressed"]
    milestone: List[Literal["has_seen_evidence", "has_revealed_secret"]]
    topic: str
    summary: str
    # Information revealed by player
    knowledge: List[str] | None

    # Generated content
    draft_response: str | None
    # messages: Sequence[str]
    messages: Annotated[list[AnyMessage], add_messages]
    # messages: Annotated[list[AnyMessage], add_messages]


async def call_model(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Process input and returns output.

    Can use runtime context to alter behavior.
    """
    print(state)
    ai_msg = llm.invoke(state.messages)
    response = ai_msg.text

    return {
        # "response": ai_msg.text,
        "messages": [response]
        # f"Configured with {(runtime.context or {}).get('my_configurable_param')}"
    }

messages = [
    SystemMessage("""
    You are an agent roleplaying as an NPC character in a video game.
    You should:
    - Act as a specific character in a game
        - Have a name
        - Have context on the background of the character
    - Have a behaviour tree (potentially with loops) that dictates the current emotional state of the character 
    - Have some key information that the user is trying to get out of the agent
    - Have "milestones" that they permanently remember
        - Only reveal key information once the agent has reached all the necessary milestones
        - Allow multiple ways of solving a problem by letting any of the necessary milestones be sufficient for revealing key information
    """)
]

# Define the graph
graph = (
    StateGraph(State(messages=messages), context_schema=Context)
    .add_node(call_model)
    .add_edge("__start__", "call_model")
    .compile(name="New Graph")
)
