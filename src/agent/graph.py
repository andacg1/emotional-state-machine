"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations

from typing import Literal, List, Annotated

from langchain_core.messages import AIMessage, AnyMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.constants import START
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime
from typing_extensions import TypedDict
from langgraph.graph.message import MessagesState, add_messages
from langmem.short_term import SummarizationNode, RunningSummary
from langgraph.checkpoint.memory import InMemorySaver

import getpass
import os

from agent.llm import structured_llm, summarization_model
from agent.npc import NPCResponse

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")
os.environ["PYTHONFROZENMODULES"] = "off"


# llm = ChatOpenAI(
#     model="gpt-5-nano",
#     # stream_usage=True,
#     # temperature=None,
#     # max_tokens=None,
#     # timeout=None,
#     # reasoning_effort="low",
#     # max_retries=2,
#     # api_key="...",  # If you prefer to pass api key in directly
#     # base_url="...",
#     # organization="...",
#     # other params...
# )


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


def _knowledge_system_message(current_knowledge: List[str] | None) -> SystemMessage:
    existing = "\n".join(f"- {k}" for k in current_knowledge) if current_knowledge else "None yet."
    return SystemMessage(f"""
    ## Structured output rules

    You must populate the `knowledge` field with everything the player has revealed so far.
    The current list is:
    {existing}

    Rules:
    - ALWAYS include every distinct fact from the current list above — never drop an item entirely.
    - You MAY merge or rephrase redundant entries into a single, shorter entry to save space.
    - Add any new facts the player just revealed in this turn.
    - Return `null` only if the player has revealed absolutely nothing across the entire conversation.
    """)

summarization_node = SummarizationNode(
    token_counter=count_tokens_approximately,
    model=summarization_model,
    max_tokens=256,
    max_tokens_before_summary=256,
    max_summary_tokens=128,
)

async def get_user_input(state: OverallState, runtime: Runtime[Context]) -> dict:
    knowledge_msg = _knowledge_system_message(state.get("knowledge"))
    response: NPCResponse = await structured_llm.ainvoke(
        SYSTEM_MESSAGES + [knowledge_msg] + state["messages"]
    )

    return {
        "messages": [AIMessage(content=response.message)],
        "emotion": response.emotion,
        "topic": response.topic,
        "summary": response.summary,
        "knowledge": response.knowledge,
    }

checkpointer = InMemorySaver()
# Define the graph
graph = (
    StateGraph(OverallState, context_schema=Context)
    .add_node(get_user_input)
    .add_node("summarize", summarization_node)
    .add_edge(START, "get_user_input")
    .compile(name="NPC Graph")
)
