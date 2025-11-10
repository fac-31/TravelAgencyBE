#!/usr/bin/env python3
"""Interactive CLI test for the form agent"""

from src.agents.form_agent import form_agent

def main():
    print("Travel Booking Form Agent")
    print("-" * 50)
    print("(Type 'exit' or 'quit' to stop)\n")

    form_data = {}
    completed_fields = []

    # Start the conversation with initial greeting
    print("Starting conversation...\n")
    result = form_agent("Hi, I want to book a trip", form_data=form_data)
    form_data = result["form_data"]
    completed_fields = result["completed_fields"]

    print(f"Agent: {result['response']}\n")

    while True:
        # Get user input
        user_input = input("You: ").strip()
        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit"]:
            break

        # Run agent with user input
        result = form_agent(user_input, form_data=form_data)

        # Update state for next iteration
        form_data = result["form_data"]
        completed_fields = result["completed_fields"]

        # Print agent response
        print(f"\nAgent: {result['response']}")

        # Check if form is complete
        if completed_fields:
            print(f"Filled fields: {', '.join(completed_fields)}")

        # If all required fields are complete
        if len(completed_fields) == 6:  # All 6 required fields
            print("\nForm Complete!")
            print("\nFinal form data:")
            import json
            print(json.dumps(form_data, indent=2))
            break

        print()


if __name__ == "__main__":
    main()
