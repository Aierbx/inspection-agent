from fastapi import APIRouter, Request
from app.schemas import AgentRequest, ChatResponse
from app.agent.loop import agent_loop
from app.llm import chat_stream
from starlette.responses import StreamingResponse

import json

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("", response_model=ChatResponse)
async def chat(req: AgentRequest, request: Request):
    reply, steps = await agent_loop(
        client=request.app.state.llm_client,
        model_name=request.app.state.model_name,
        messages=[m.model_dump() for m in req.messages],
    )
    return ChatResponse(reply=reply, steps=steps)

@router.post("/stream")
async def chat_sse(req: AgentRequest, request: Request):
    async def generate():
        try:
            async for chunk in chat_stream(
                        client=request.app.state.llm_client,
                        model_name=request.app.state.model_name,
                        messages = [m.model_dump() for m in req.messages]
                    ):
                yield f"data: {json.dumps({'delta': chunk}, ensure_ascii=False)}\n\n"

            yield f"data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
    )