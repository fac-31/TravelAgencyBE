from fastapi import APIRouter
from pydantic import BaseModel
from langchain.messages import HumanMessage
from src.agents.receptionist import receptionist

router = APIRouter()

class Query(BaseModel):
    input: str

@router.post("/ask")
def ask_agent(query: Query):
    try:
        messages = [HumanMessage(content=query.input)]
        print({"input": query.input})
        result = receptionist.invoke({"messages": messages})
        final_response = result["messages"][-1].content
        print({"final-response": final_response})
        return {"response": final_response}
    except Exception as e:
        print({"error": str(e)})
        return {"error": str(e)}
