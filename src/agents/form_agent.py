"""
Form agent that conducts a natural conversation to fill out a travel booking form.
Similar to the weather agent, this agent is a node in the LangGraph tree.
"""

import json
from pathlib import Path
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage, AnyMessage
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated
import operator
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

# Load the form schema
form_schema_path = Path(__file__).parent.parent.parent / "form.json"
with open(form_schema_path, "r") as f:
    FORM_SCHEMA = json.load(f)

class FormState(TypedDict):
    """State for the form filling agent"""
    messages: Annotated[list[AnyMessage], operator.add]
    form_data: dict  # Current form data being filled
    completed_fields: list[str]  # Fields that have been filled


model = init_chat_model("anthropic:claude-sonnet-4-5", temperature=0.7)

FORM_SYSTEM_PROMPT = """You are a friendly travel advisor having a casual conversation.

Your hidden goal is to naturally learn about:
1. Their budget for the trip (how much they want to spend)
2. Type of holiday (adventure, beach, cultural, relaxation, etc.)
3. Travel group (solo, couple, family, friends)
4. Trip dates (when they want to travel - start and end dates)
5. Destination preferences (where they'd like to go)

IMPORTANT: You are NOT filling out a form. You're having a natural, friendly chat. Ask casual questions like you're talking to a friend. When they mention something relevant to any of these topics, just acknowledge it naturally and move on to the next topic.

Be conversational, warm, and genuinely interested. Ask one topic at a time. Don't be rigid or formal.
The user may give vague answers like "around 2k or 3k" and you should interpret and decide on a reasonable value.

Current collected info will be provided. Ask about missing topics naturally."""

def llm_call(state: FormState) -> dict:
    """LLM node that responds naturally or thanks user when done"""
    form_data = state.get("form_data", {})
    completed = state.get("completed_fields", [])
    messages = state.get("messages", [])

    # Check if form is complete - if so, thank the user
    if is_form_complete(state):
        thank_you_prompt = f"""The user has provided all the information needed for their trip:
- Budget: ${form_data.get('budget')}
- Type: {form_data.get('typeOfHoliday')}
- Traveling: {form_data.get('travelGroup')}
- Dates: {form_data.get('availability', {}).get('startDate')} to {form_data.get('availability', {}).get('endDate')}
- Destinations: {', '.join(form_data.get('destinationPreferences', []))}

Write a warm, brief thank you message. Let them know you have everything you need and will help them find the perfect trip. Keep it to 2-3 sentences."""

        response = model.invoke([SystemMessage(content=thank_you_prompt)])
        return {"messages": [response]}

    # Build context about what we've collected so far
    collected_info = ""
    if completed:
        collected_info = f"\n\nSo far I know: {', '.join(completed)}"

    # If no messages yet, add an initial greeting
    if not messages:
        messages = [HumanMessage(content="Hi, I want to book a trip")]

    response = model.invoke(
        [
            SystemMessage(content=FORM_SYSTEM_PROMPT + collected_info),
            *messages
        ]
    )

    return {
        "messages": [response]
    }


def extract_form_data(state: FormState) -> dict:
    """Extract form data from the last user message"""
    current_form = state.get("form_data", {})
    messages = state.get("messages", [])

    if not messages or len(messages) < 2:
        return {"form_data": current_form, "completed_fields": state.get("completed_fields", [])}

    # Get only the last user message
    last_user_message = None
    for msg in reversed(messages):
        if hasattr(msg, 'type') and msg.type == 'human':
            last_user_message = msg.content
            break
        elif isinstance(msg, HumanMessage):
            last_user_message = msg.content
            break

    if not last_user_message:
        return {"form_data": current_form, "completed_fields": state.get("completed_fields", [])}

    # Use LLM to extract information - be very explicit about format
    extraction_prompt = f"""Extract travel information from this user message.

User said: "{last_user_message}"

Already have: {json.dumps(current_form, indent=2)}

Rules:
- budget: extract as NUMBER only (e.g., 2500 not "2500" or "2.5k")
- typeOfHoliday: beach, adventure, cultural, relaxation, etc
- travelGroup: solo, couple, family, friends, group
- availability: only if they mention specific dates, as {{"startDate": "YYYY-MM-DD", "endDate": "YYYY-MM-DD"}}
- destinationPreferences: list of place names

Return ONLY valid JSON. If no info found, return {{}}.

Examples:
{{"budget": 2500}}
{{"typeOfHoliday": "beach", "travelGroup": "family"}}
{{"destinationPreferences": ["Bali", "Thailand"]}}

JSON only:"""

    # Need at least a user message along with the system message
    response = model.invoke([
        SystemMessage(content=extraction_prompt),
        HumanMessage(content="Extract the information.")
    ])
    response_text = response.content if isinstance(response.content, str) else str(response.content)

    # Parse JSON from response
    try:
        import re
        # Find JSON object
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            extracted = json.loads(json_str)

            # Merge extracted data
            for key, value in extracted.items():
                if key == "availability" and isinstance(value, dict):
                    if "availability" not in current_form:
                        current_form["availability"] = {}
                    current_form["availability"].update(value)
                elif key == "destinationPreferences" and isinstance(value, list):
                    current_form["destinationPreferences"] = value
                elif value:  # Only set non-empty values
                    current_form[key] = value
    except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
        pass  # Could not parse JSON

    # Determine completed fields
    completed_fields = []
    if current_form.get("budget"):
        completed_fields.append("budget")
    if current_form.get("typeOfHoliday"):
        completed_fields.append("type of holiday")
    if current_form.get("travelGroup"):
        completed_fields.append("travel group")
    if current_form.get("availability", {}).get("startDate"):
        completed_fields.append("start date")
    if current_form.get("availability", {}).get("endDate"):
        completed_fields.append("end date")
    if current_form.get("destinationPreferences"):
        completed_fields.append("destinations")

    return {"form_data": current_form, "completed_fields": completed_fields}


def is_form_complete(state: FormState) -> bool:
    """Check if all required form fields are complete"""
    form_data = state.get("form_data", {})

    # Check required fields (all must be non-empty strings or valid data)
    required_fields = {
        "budget": lambda x: bool(x),
        "typeOfHoliday": lambda x: bool(x),
        "travelGroup": lambda x: bool(x),
    }

    for field, validator in required_fields.items():
        if not validator(form_data.get(field)):
            return False

    # Check nested fields
    availability = form_data.get("availability", {})
    if not (availability.get("startDate") and availability.get("endDate")):
        return False

    # Check destination preferences (at least one destination)
    destinations = form_data.get("destinationPreferences", [])
    if not destinations or (isinstance(destinations, list) and len(destinations) == 0):
        return False

    return True


def should_continue(state: FormState) -> str:
    """Determine if we should continue collecting data or end"""
    if is_form_complete(state):
        return END
    return "extract_and_ask"


# Build the internal graph (for testing/standalone use)
def build_form_graph():
    """Build the internal form-filling graph

    Flow per user interaction:
    1. extract_and_ask: Extract info from latest user message and check if complete
    2. If complete: END
    3. If not complete: llm_call to ask next question
    """
    graph = StateGraph(FormState)
    graph.add_node("extract_and_ask", extract_form_data)
    graph.add_node("llm_call", llm_call)

    graph.add_edge(START, "extract_and_ask")
    graph.add_conditional_edges("extract_and_ask", should_continue, {"extract_and_ask": "llm_call", END: END})
    graph.add_edge("llm_call", END)

    return graph.compile()


# Compile for standalone use
form_agent = build_form_graph()


# ============================================================================
# EXPORT: Node function for use in parent LangGraph
# ============================================================================
# When used as a node in a parent graph, call this single function:
# It runs the internal form-filling loop until the form is complete,
# then returns the completed form_data

async def form_agent_node(state: dict) -> dict:
    """
    Standalone node function for use in parent LangGraph.

    Input state should have:
    - messages: conversation history (optional)
    - form_data: current form data (optional)
    - completed_fields: list of completed fields (optional)

    Returns:
    - form_data: the completed form with all required fields
    - completed_fields: list of fields that were filled
    - messages: full conversation history
    """
    # Initialize state if needed
    initial_state = {
        "messages": state.get("messages", []),
        "form_data": state.get("form_data", {}),
        "completed_fields": state.get("completed_fields", [])
    }

    # Run the internal form agent graph until completion
    result = form_agent.invoke(initial_state)

    # Return the form data for the next node
    return {
        "form_data": result.get("form_data", {}),
        "completed_fields": result.get("completed_fields", []),
        "messages": result.get("messages", [])
    }
