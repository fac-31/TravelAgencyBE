import requests
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, ToolMessage, AnyMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated
import operator
import os
from dotenv import load_dotenv

@tool
def get_exchange_rate(to_currency: str, from_currency: str = None) -> float:
    """Fetch current exchange rate from local currency to target currency."""
    
    load_dotenv()
    api_key = os.getenv("EXCHANGE_RATE_API_KEY")
    
    if not api_key:
        return "Error: API key not found in environment variables."
    
    url = f"https://v6.exchangerate-api.com/v6/{api_key}/pair/{from_currency}/{to_currency}"
    
    try:
        res = requests.get(url, timeout=5)
        data = res.json()
        
        if data.get("result") == "success":
            return data["conversion_rate"]
        else:
            return f"API error: {data.get('error-type', 'Unknown error')}"
    except Exception as e:
        return f"Error fetching exchange rate: {str(e)}"

model = init_chat_model("anthropic:claude-sonnet-4-5", temperature=0)
model_with_tool = model.bind_tools([get_exchange_rate])

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

def llm_call(state: dict):
    return {
        "messages": [
            model_with_tool.invoke(
                [SystemMessage(content="You are an exchange rate assistant.")] + state["messages"]
            )
        ]
    }

def tool_node(state: dict):
    last = state["messages"][-1]
    results = []
    for tool_call in last.tool_calls:
        tool = get_exchange_rate
        obs = tool.invoke(tool_call["args"])
        results.append(ToolMessage(content=obs, tool_call_id=tool_call["id"]))
    # Append tool results to the existing message history so tool_result
    # blocks properly reference the AI message's tool_use block.
    return {"messages": state["messages"] + results}

def should_continue(state: MessagesState):
    return "tool_node" if state["messages"][-1].tool_calls else END

graph = StateGraph(MessagesState)
graph.add_node("llm_call", llm_call)
graph.add_node("tool_node", tool_node)
graph.add_edge(START, "llm_call")
graph.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
graph.add_edge("tool_node", "llm_call")

exchange_agent = graph.compile()
