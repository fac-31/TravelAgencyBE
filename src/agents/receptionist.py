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
    routes: list[str]  # list of chosen routes to run
    results: dict[str, str] # results from each specialized agent


# ========== MODEL ==========
router_model = init_chat_model("anthropic:claude-sonnet-4-5", temperature=0)

# ========== LLM DECISION NODE ==========
def decide_route(state: RouterState):
    """LLM decides which specialized agent(s) to use."""

    names = []
    texts = []
    for name, info in nodes.items():
        names.append(f"'{name}'")
        texts.append(f"- '{name}': {info['prompt']}")

    system_prompt = SystemMessage(
        content=(
            "You are a routing assistant. Given the user's request, decide which specialized agents "
            "should handle it.\n"
            "Available agents:\n"
            "\n".join(texts) + "\n"
            "You may choose " + ", ".join(names) + "."
            "You can choose multiple of it, respond with only those options seperated with a ','."
        )
    )

    decision = router_model.invoke([system_prompt] + state["messages"])
    routes = decision.content.strip().lower().split(",")

    if len(routes) == 0:
        routes = ["weather"]  # fallback default

    return {"routes": routes}


# ========== TOOL INVOCATION NODES ==========
def run_weather(state: RouterState):
    from langchain.messages import HumanMessage

    user_msg = state["messages"][-1].content
    res = weather_agent.invoke({"messages": [HumanMessage(content=user_msg)]})
    content = res["messages"][-1].content

    results = state.get("results", {})
    results["weather"] = content
    return {"results": results}


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
    
    results = state.get("results", {})
    results["exchange"] = content
    return {"results": results}


# ========== FINAL COMBINER NODE ==========
def combine_results(state: RouterState):
    """Combine results from weather and exchange agents into a single, consistent response."""
    results = state.get("results", {})
    texts = []
    for key, text in results.items():
        texts.append(text)

    if len(texts) == 0:
        combined_text = "No relevant information found."
    else:
        combined_text = "\n\n".join(texts)
    
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


# node mapping
#- callback: langraph node function to invoke
#- prompt: description to send to AI prompt for routing decisions
nodes = {
    "weather": {
        "callback": run_weather,
        "prompt": "for weather forecasts or destinations",
    },
    "exchange": {
        "callback": run_exchange,
        "prompt": "for currency or travel money queries",
    },
}


# ========== GRAPH ==========
router_graph = StateGraph(RouterState)

# nodes
router_graph.add_node("decide_route", decide_route)
router_graph.add_node("combine_results", combine_results)

for name, info in nodes.items():
    router_graph.add_node(name, info["callback"])

# edges
router_graph.add_edge(START, "decide_route")

# conditional edges based on route decision
def route_logic(state: RouterState):
    routes = state.get("routes")
    
    # Go through each routes that has not been run yet
    for route in routes:
        if route not in state.get("results", {}):
            return route

    # After all routes have been run, go to combine_results
    return "combine_results"

router_graph.add_conditional_edges("decide_route", route_logic)

for name in nodes.keys():
    router_graph.add_conditional_edges(name, route_logic)

receptionist = router_graph.compile()
