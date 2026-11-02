from fastapi import APIRouter, Request
from app.schemas import AgentRequest, ChatResponse
from app.agent.loop import agent_loop

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("", response_model=ChatResponse)
async def chat(req: AgentRequest, request: Request):
    reply, steps = await agent_loop(
        client=request.app.state.llm_client,
        model_name=request.app.state.model_name,
        messages=[m.model_dump() for m in req.messages],
    )
    return ChatResponse(reply=reply, steps=steps)