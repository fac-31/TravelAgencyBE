"""
Form agent - collects travel booking information through natural conversation
"""

import json
import re
from pathlib import Path
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

# Load the form schema
form_schema_path = Path(__file__).parent.parent.parent / "form.json"
with open(form_schema_path, "r") as f:
    FORM_SCHEMA = json.load(f)

model = init_chat_model("anthropic:claude-sonnet-4-5", temperature=0.7)

def build_system_prompt():
    """Build system prompt dynamically from form schema."""
    fields = list(FORM_SCHEMA.keys())

    field_descriptions = {
        "budget": "their budget for the trip (how much they want to spend)",
        "typeOfHoliday": "type of holiday (adventure, beach, cultural, relaxation, etc.)",
        "travelGroup": "travel group (solo, couple, family, friends)",
        "availability": "trip dates (when they want to travel - start and end dates)",
        "destinationPreferences": "destination preferences (where they'd like to go)"
    }

    field_list = "\n".join(
        f"{i+1}. {field_descriptions.get(field, field)}"
        for i, field in enumerate(fields)
    )

    return f"""You are a friendly travel advisor having a casual conversation.

Your hidden goal is to naturally learn about:
{field_list}

IMPORTANT: You are NOT filling out a form. You're having a natural, friendly chat. Ask casual questions like you're talking to a friend. When they mention something relevant to any of these topics, just acknowledge it naturally and move on to the next topic.

Be conversational, warm, and genuinely interested. Ask one topic at a time. Don't be rigid or formal.
The user may give vague answers and you should interpret and decide on a reasonable value.

Current collected info will be provided. Ask about missing topics naturally."""

FORM_SYSTEM_PROMPT = build_system_prompt()


def _build_extraction_rules() -> str:
    """Build extraction rules dynamically from form schema."""
    field_descriptions = {
        "budget": "extract as NUMBER only (e.g., 2500 not \"2500\" or \"2.5k\")",
        "typeOfHoliday": "beach, adventure, cultural, relaxation, etc",
        "travelGroup": "solo, couple, family, friends, group",
        "availability": "only if they mention specific dates, as {\"startDate\": \"YYYY-MM-DD\", \"endDate\": \"YYYY-MM-DD\"}",
        "destinationPreferences": "list of place names",
    }

    rules = []
    for field in FORM_SCHEMA.keys():
        description = field_descriptions.get(field, "")
        if description:
            rules.append(f"- {field}: {description}")

    return "\n".join(rules)


def extract_form_data(user_message: str, current_form: dict) -> dict:
    """Extract form data from user message using LLM."""
    extraction_rules = _build_extraction_rules()

    extraction_prompt = f"""Extract travel information from this user message.

User said: "{user_message}"

Already have: {json.dumps(current_form, indent=2)}

Rules:
{extraction_rules}

Return ONLY valid JSON. If no info found, return {{}}.

Examples:
{{"budget": 2500}}
{{"typeOfHoliday": "beach", "travelGroup": "family"}}
{{"destinationPreferences": ["Bali", "Thailand"]}}

JSON only:"""

    response = model.invoke([
        SystemMessage(content=extraction_prompt),
        HumanMessage(content="Extract the information.")
    ])
    response_text = response.content if isinstance(response.content, str) else str(response.content)

    # Parse JSON from response
    try:
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

    return current_form


def _format_field_name(field: str) -> str:
    """Convert camelCase field name to readable format: 'typeOfHoliday' -> 'type of holiday'."""
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', field)
    return re.sub('([a-z0-9])([A-Z])', r'\1 \2', s1).lower()


def get_completed_fields(form_data: dict) -> list[str]:
    """Determine which form fields have been completed based on schema structure."""
    completed_fields = []

    # Check flat fields from schema (budget, typeOfHoliday, travelGroup, destinationPreferences)
    for field in FORM_SCHEMA.keys():
        if field == "availability":
            # availability is nested, handle separately
            continue

        value = form_data.get(field)
        if value:  # Check if field has data
            completed_fields.append(_format_field_name(field))

    # Check nested availability fields
    availability = form_data.get("availability", {})
    if isinstance(availability, dict):
        if availability.get("startDate"):
            completed_fields.append("start date")
        if availability.get("endDate"):
            completed_fields.append("end date")

    return completed_fields


def is_form_complete(form_data: dict) -> bool:
    """Check if all required form fields are complete based on schema structure."""
    # Check flat required fields (all non-nested fields are required: budget, typeOfHoliday, travelGroup)
    for field in FORM_SCHEMA.keys():
        if field == "availability" or field == "destinationPreferences":
            # Handle these specially
            continue

        value = form_data.get(field)
        if not value:  # Field must have a truthy value
            return False

    # Check availability (must have both startDate and endDate)
    availability = form_data.get("availability", {})
    if not isinstance(availability, dict):
        return False
    if not (availability.get("startDate") and availability.get("endDate")):
        return False

    # Check destinationPreferences (must be non-empty list)
    destinations = form_data.get("destinationPreferences", [])
    if not destinations or (isinstance(destinations, list) and len(destinations) == 0):
        return False

    return True


def form_agent(user_message: str, form_data: dict = None, messages: list = None) -> dict:
    """
    Process user message for travel booking form.

    Args:
        user_message: The user's message
        form_data: Current form data being filled (default: empty dict)
        messages: Conversation history (default: empty list)

    Returns:
        Dict with keys: 'response', 'form_data', 'completed_fields'
    """
    if form_data is None:
        form_data = {}
    if messages is None:
        messages = []

    # Extract any form data from the user message
    form_data = extract_form_data(user_message, form_data)
    completed_fields = get_completed_fields(form_data)

    # Check if form is complete
    if is_form_complete(form_data):
        thank_you_prompt = f"""The user has provided all the information needed for their trip:
- Budget: ${form_data.get('budget')}
- Type: {form_data.get('typeOfHoliday')}
- Traveling: {form_data.get('travelGroup')}
- Dates: {form_data.get('availability', {}).get('startDate')} to {form_data.get('availability', {}).get('endDate')}
- Destinations: {', '.join(form_data.get('destinationPreferences', []))}

Write a warm, brief thank you message. Let them know you have everything you need and will help them find the perfect trip. Keep it to 2-3 sentences."""

        response = model.invoke([
            SystemMessage(content=thank_you_prompt),
            HumanMessage(content="Generate the thank you message.")
        ])
        agent_response = response.content
    else:
        # Build context about what we've collected
        collected_info = ""
        if completed_fields:
            collected_info = f"\n\nSo far I know: {', '.join(completed_fields)}"

        # Generate next question
        system_prompt = FORM_SYSTEM_PROMPT + collected_info
        response = model.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ])
        agent_response = response.content

    return {
        "response": agent_response,
        "form_data": form_data,
        "completed_fields": completed_fields
    }
