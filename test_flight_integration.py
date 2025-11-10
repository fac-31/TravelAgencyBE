#!/usr/bin/env python3
"""Test flight agent integration with the main graph"""

from src.agents.flight_agent import flight_agent
from src.agents.graph import AGENTS, travel_agent
from langchain.messages import HumanMessage

print("=" * 60)
print("FLIGHT AGENT INTEGRATION TEST")
print("=" * 60)

# Test 1: Flight agent available in AGENTS dict
print("\n1. Checking flight agent is registered in AGENTS...")
if "flight" in AGENTS:
    print(f"   ✅ Flight agent found: '{AGENTS['flight']}'")
else:
    print(f"   ❌ Flight agent not found in AGENTS")
    print(f"   Available agents: {list(AGENTS.keys())}")

# Test 2: Flight agent function works
print("\n2. Testing flight_agent function...")
try:
    result = flight_agent("Find flights from NYC to LAX on 2025-12-15")
    if isinstance(result, str):
        print(f"   ✅ flight_agent returned string response")
        print(f"   Response preview: {result[:100]}...")
    else:
        print(f"   ❌ Unexpected return type: {type(result)}")
except Exception as e:
    print(f"   ❌ Error calling flight_agent: {e}")

# Test 3: Graph routing decides on flight agent
print("\n3. Testing graph routing for flight request...")
try:
    test_request = {
        "client": {"host": "127.0.0.1"}
    }
    result = travel_agent.invoke({
        "messages": [HumanMessage(content="I need to find flights from NYC to LA")],
        "request": test_request,
    })

    if "messages" in result and len(result["messages"]) > 0:
        print(f"   ✅ Graph invoked successfully")
        response_content = result["messages"][-1].content
        if isinstance(response_content, str):
            print(f"   Response preview: {response_content[:100]}...")
        else:
            print(f"   ❌ Unexpected message content type: {type(response_content)}")
    else:
        print(f"   ❌ Unexpected result structure: {result.keys()}")
except Exception as e:
    print(f"   ❌ Error invoking graph: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("FLIGHT AGENT INTEGRATION TEST COMPLETE")
print("=" * 60)
