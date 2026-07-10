from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Literal, Annotated
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import operator
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    temperature=0
)

#State
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    
def chat_node(state: ChatState) -> ChatState:
    #take user query
    messages = state['messages']
    
    #send to llm
    response = llm.invoke(messages)
    
    #store response in state 
    return {'messages': [response]}

checkpointer = MemorySaver()
graph = StateGraph(ChatState)

#Add nodes
graph.add_node("chat_node", chat_node)

#Edges
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

graph.compile(checkpointer=checkpointer)

workflow = graph.compile(checkpointer=checkpointer)

# config = {'configurable': {'thread_id': 'thread-1'}}

# for message_chunk, metadata in workflow.stream(
#     {'messages': [HumanMessage(content='Recipe to make Pasta?')]},
#     config = config,
#     stream_mode="messages"
# ):
#     if message_chunk.content:
#         print(message_chunk.content, end = " ", flush=True)