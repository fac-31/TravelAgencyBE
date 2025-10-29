from langchain.chat_models import init_chat_model
from langchain.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated
import operator

from .weather_agent import weather_agent
from .exchange_agent import exchange_agent
from .tools.currency_tools import detect_local_currency


# ========== STATE ==========
class RouterState(TypedDict):
    messages: Annotated[list, operator.add]
    route: str
    weather_result: str
    exchange_result: str


# ========== MODEL ==========
router_model = init_chat_model("anthropic:claude-sonnet-4-5", temperature=0)

# ========== LLM DECISION NODE ==========
def decide_route(state: RouterState):
    """LLM decides which specialized agent(s) to use."""
    system_prompt = SystemMessage(
        content=(
            "You are a routing assistant. Given the user's request, decide which specialized agents "
            "should handle it.\n"
            "Available agents:\n"
            "- 'weather': for weather forecasts or destinations.\n"
            "- 'exchange': for currency or travel money queries.\n"
            "You may choose 'weather', 'exchange', or 'both'. "
            "Respond with only one word: weather, exchange, or both."
        )
    )

    decision = router_model.invoke([system_prompt] + state["messages"])
    route = decision.content.strip().lower()

    # Normalize
    if route not in ["weather", "exchange", "both"]:
        route = "weather"  # fallback default

    return {"route": route}


# ========== TOOL INVOCATION NODES ==========
def run_weather(state: RouterState):
    from langchain.messages import HumanMessage

    user_msg = state["messages"][-1].content
    res = weather_agent.invoke({"messages": [HumanMessage(content=user_msg)]})
    content = res["messages"][-1].content
    return {"weather_result": content}


def run_exchange(state: RouterState):
    """Run exchange agent with local currency automatically detected."""
    user_msg = state["messages"][-1].content
    from_currency = detect_local_currency()  # detect local currency automatically
    
    # Pass a system message to inform LLM the from_currency is known
    system_msg = HumanMessage(
        content=(
            f"Provide the exchange rate from the local currency ({from_currency}) "
            f"to the target currency requested by the user. "
            "Do not ask the user for their local currency."
        )
    )
    
    res = exchange_agent.invoke({
        "messages": [system_msg, HumanMessage(content=user_msg)]
    })
    content = res["messages"][-1].content
    return {"exchange_result": content}


# ========== FINAL COMBINER NODE ==========
def combine_results(state: RouterState):
    """Combine results from weather and exchange agents into a single, consistent response."""
    weather = state.get("weather_result", "")
    exchange = state.get("exchange_result", "")
    
    combined_text = "\n\n".join([r for r in [weather, exchange] if r])
    
    if not combined_text:
        combined_text = "No relevant information found."
    
    # Use the router_model to merge into a single voice
    system_prompt = SystemMessage(
        content=(
            "You are a friendly and professional travel assistant. "
            "Combine the following information into a single response in a consistent voice. "
            "Keep it concise, helpful, and natural."
        )
    )
    
    final_response = router_model.invoke([system_prompt, HumanMessage(content=combined_text)])
    
    return {"messages": [AIMessage(content=final_response.content)]}


# ========== GRAPH ==========
router_graph = StateGraph(RouterState)

# nodes
router_graph.add_node("decide_route", decide_route)
router_graph.add_node("run_weather", run_weather)
router_graph.add_node("run_exchange", run_exchange)
router_graph.add_node("combine_results", combine_results)

# edges
router_graph.add_edge(START, "decide_route")

# conditional edges based on route decision
def route_logic(state: RouterState):
    route = state.get("route")
    if route == "weather":
        return "run_weather"
    elif route == "exchange":
        return "run_exchange"
    elif route == "both":
        return "run_weather"  # run weather first, then chain to exchange
    else:
        return "combine_results"

router_graph.add_conditional_edges("decide_route", route_logic,
    ["run_weather", "run_exchange", "combine_results"]
)

# connect weather → exchange (for 'both' route)
router_graph.add_edge("run_weather", "run_exchange")
router_graph.add_edge("run_exchange", "combine_results")
router_graph.add_edge("run_weather", "combine_results")  # in case route='weather' only
router_graph.add_edge("run_exchange", "combine_results")

receptionist = router_graph.compile()
