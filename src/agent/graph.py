"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, List, Sequence, Annotated

from langchain_core.messages import AIMessage, AnyMessage, SystemMessage
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime
from pydantic import BaseModel
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

class NPCResponse(BaseModel):
    message: str
    emotion: Literal["relaxed", "nervous", "panicking", "angry", "upset", "depressed"]
    topic: str
    summary: str
    knowledge: List[str] | None = None


structured_llm = llm.with_structured_output(NPCResponse)


class Context(TypedDict):
    """Context parameters for the agent.

    Set these when creating assistants OR when invoking the graph.
    See: https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/
    """
    model_name: str

class PrivateState(TypedDict):
    secret_knowledge: List[str] | None

class OverallState(MessagesState):
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


SYSTEM_MESSAGES = [
    SystemMessage("""
    You are an agent roleplaying as an NPC character in a video game.
    You should:
    - Act as a specific character in a game
        - Have a name
        - Have context on the background of the character
    - Have a behaviour tree (potentially with loops) that dictates the current emotional state of the character
    - Have some key information that the user is trying to get out of the agent
    - Have "milestones" that you permanently remember
        - NEVER explicitly tell the user any of these milestones 
        - Only reveal key information once you have reached all the necessary milestones
        - Allow multiple ways of solving a problem by letting any of the necessary milestones be sufficient for revealing key information
    """)
]


async def get_user_input(state: OverallState, runtime: Runtime[Context]) -> dict:
    response: NPCResponse = await structured_llm.ainvoke(SYSTEM_MESSAGES + state["messages"])

    return {
        "messages": [AIMessage(content=response.message)],
        "emotion": response.emotion,
        "topic": response.topic,
        "summary": response.summary,
        "knowledge": response.knowledge,
    }

# Define the graph
graph = (
    StateGraph(OverallState, context_schema=Context)
    .add_node(get_user_input)
    .add_edge("__start__", "get_user_input")
    .compile(name="New Graph")
)
