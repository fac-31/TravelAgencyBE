#!/usr/bin/env python3
"""Interactive CLI test for the form agent"""

from src.agents.form_agent import form_agent
from langchain.messages import HumanMessage

def main():
    print("Travel Booking Form Agent")
    print("-" * 50)
    print("(Type 'exit' or 'quit' to stop)\n")

    state = {
        "messages": [],
        "form_data": {},
        "completed_fields": []
    }

    # Start the conversation
    print("Starting conversation...\n")
    result = form_agent.invoke(state, {"recursion_limit": 100})
    state["messages"] = result["messages"]
    state["form_data"] = result["form_data"]
    state["completed_fields"] = result["completed_fields"]

    last_message = result["messages"][-1]
    print(f"Agent: {last_message.content}\n")

    while True:
        # Get user input
        user_input = input("You: ").strip()
        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit"]:
            break

        # Add user message to state
        state["messages"].append(HumanMessage(content=user_input))

        # Run agent
        result = form_agent.invoke(state, {"recursion_limit": 100})

        # Update state for next iteration
        state["messages"] = result["messages"]
        state["form_data"] = result["form_data"]
        state["completed_fields"] = result["completed_fields"]

        # Print agent response
        last_message = result["messages"][-1]
        print(f"\nAgent: {last_message.content}")

        # Check if form is complete
        if result["completed_fields"]:
            print(f"Filled fields: {', '.join(result['completed_fields'])}")

        # If all required fields are complete
        if len(result["completed_fields"]) == 6:  # All 6 required fields
            print("\nForm Complete!")
            print("\nFinal form data:")
            import json
            print(json.dumps(result["form_data"], indent=2))
            break

        print()


if __name__ == "__main__":
    main()
