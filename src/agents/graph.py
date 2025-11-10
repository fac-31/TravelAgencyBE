"""
Main LangGraph router for the Travel Agency Backend.
Orchestrates specialized agents (weather, exchange, form) based on user requests.
"""

from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated
import operator

from .weather_agent import weather_agent
from .exchange_agent import exchange_agent
from .form_agent import form_agent


# ========== STATE ==========
class RouterState(TypedDict):
    """State for the main router graph"""

    messages: Annotated[list, operator.add]
    routes: list[str]  # list of chosen routes to run
    results: dict[str, str]  # results from each specialized agent


# ========== MODEL ==========
router_model = init_chat_model("anthropic:claude-sonnet-4-5", temperature=0)


# ========== AGENT DESCRIPTIONS ==========
AGENTS = {
    "weather": "for weather forecasts or destinations",
    "exchange": "for currency or travel money queries",
    "form": "for creating and collecting travel booking information",
}


# ========== ROUTER NODE ==========
def decide_route(state: RouterState):
    """LLM decides which specialized agent(s) to use."""
    agent_list = "\n".join(f"- '{name}': {desc}" for name, desc in AGENTS.items())

    system_prompt = SystemMessage(
        content=(
            "You are a routing assistant. Given the user's request, decide which specialized agents "
            "should handle it.\n"
            "Available agents:\n"
            f"{agent_list}\n"
            f"Respond with only the agent names separated by commas, or pick one."
        )
    )

    decision = router_model.invoke([system_prompt] + state["messages"])
    routes = [r.strip() for r in decision.content.strip().lower().split(",")]

    if not routes or routes == [""]:
        routes = ["weather"]  # fallback default

    return {"routes": routes}


# ========== RESULT COMBINER NODE ==========
def combine_results(state: RouterState):
    """Combine results from all agents into a single response."""
    results = state.get("results", {})
    texts = list(results.values())

    if not texts:
        combined_text = "No relevant information found."
    else:
        combined_text = "\n\n".join(texts)

    system_prompt = SystemMessage(
        content=(
            "You are a friendly and professional travel assistant. "
            "Combine the following information into a single response in a consistent voice. "
            "Keep it concise, helpful, and natural."
        )
    )

    final_response = router_model.invoke([system_prompt, HumanMessage(content=combined_text)])

    return {"messages": [AIMessage(content=final_response.content)]}


# ========== AGENT RUNNER NODES ==========
def run_weather(state: RouterState):
    """Call the weather agent function."""
    user_msg = state["messages"][-1].content
    result = weather_agent(user_msg)

    results = state.get("results", {})
    results["weather"] = result
    return {"results": results}


def run_exchange(state: RouterState):
    """Call the exchange agent function."""
    user_msg = state["messages"][-1].content
    result = exchange_agent(user_msg)

    results = state.get("results", {})
    results["exchange"] = result
    return {"results": results}


def run_form(state: RouterState):
    """Call the form agent function."""
    user_msg = state["messages"][-1].content
    result = form_agent(user_msg)

    results = state.get("results", {})
    results["form"] = result.get("response", "")
    return {"results": results}


# ========== GRAPH CONSTRUCTION ==========
def _build_router_graph():
    """Build the main router orchestration graph."""
    graph = StateGraph(RouterState)

    # Add router nodes
    graph.add_node("decide_route", decide_route)
    graph.add_node("combine_results", combine_results)

    # Add agent nodes
    graph.add_node("weather", run_weather)
    graph.add_node("exchange", run_exchange)
    graph.add_node("form", run_form)

    # Graph edges
    graph.add_edge(START, "decide_route")
    graph.add_edge("combine_results", END)

    # Routing logic
    def route_logic(state: RouterState):
        """Route to next agent or to combine_results."""
        routes = state.get("routes", [])
        results = state.get("results", {})

        # Find first agent that hasn't been run yet
        for route in routes:
            if route not in results:
                return route

        # All agents done, combine results
        return "combine_results"

    graph.add_conditional_edges("decide_route", route_logic)

    # From each agent, check if more agents need to run
    for agent_name in ["weather", "exchange", "form"]:
        graph.add_conditional_edges(agent_name, route_logic)

    return graph.compile()


# ========== COMPILED GRAPH ==========
travel_agent = _build_router_graph()
