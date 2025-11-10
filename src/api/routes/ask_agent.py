from fastapi import APIRouter, Request
from pydantic import BaseModel
from langchain.messages import HumanMessage
from src.agents.graph import travel_agent

router = APIRouter()


class Query(BaseModel):
    input: str


@router.post("/ask")
def ask_agent(query: Query, request: Request):
    try:
        messages = [HumanMessage(content=query.input)]
        print({"input": query.input})

        # Prepare request context for agents that may need geolocation
        request_context = {
            "client": {"host": request.client.host if request.client else "127.0.0.1"}
        }

        # Invoke graph with messages and request context
        result = travel_agent.invoke({
            "messages": messages,
            "request": request_context,
        })
        final_response = result["messages"][-1].content
        print({"final-response": final_response})
        return {"response": final_response}
    except Exception as e:
        print({"error": str(e)})
        return {"error": str(e)}
