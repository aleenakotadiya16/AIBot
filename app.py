
from typing import Annotated
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.graph import START, StateGraph, END
from langgraph.graph import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt

import os
from dotenv import load_dotenv
import streamlit as st

#from langchain_community.llms.ollama import Ollama
from langchain.chat_models import init_chat_model


# Load environment variables
load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
tavily_api_key = os.getenv('TAVILY_API_KEY')


# Initialize LLM
llm = init_chat_model("google_genai:gemini-2.0-flash")

#memory block
memory = MemorySaver()

# Define LangGraph state
class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

# Set up tool and LLM with tool binding
@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]

tool = TavilySearch(api_key=tavily_api_key,max_results=2)
tools = [tool,human_assistance]
llm_with_tools = llm.bind_tools(tools)

# Define chatbot node
def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    assert(len(message.tool_calls) <= 1)
    return {"messages": [message]}

graph_builder.add_node("chatbot", chatbot)

# Add tool node and flow control
tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

# Compile graph
graph = graph_builder.compile(checkpointer=memory)

# Streamlit UI
st.title("LangGraph AI Assistant")

if "state" not in st.session_state:
    st.session_state.state = {"messages": []}

user_input = st.text_input("Ask me anything:")

if user_input:
    # Add user message
    st.session_state.state["messages"].append({"role": "user", "content": user_input})

    # Invoke the LangGraph flow
    st.session_state.state = graph.invoke(st.session_state.state)

    # Get the latest AI message only
    ai_response = st.session_state.state["messages"][-1].content

    st.markdown(f"**Answer:** {ai_response}")

