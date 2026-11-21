from typing import Literal
from contextlib import aclosing

from fastapi import APIRouter, HTTPException, Request
from app.schemas import AgentRequest, ChatResponse
from app.agent.loop import agent_loop, agent_loop_stream
from app.llm import chat_stream
from starlette.responses import StreamingResponse
from app.session_store import create_session, session_exists, load_history, save_messages
from app.chat_context import build_chat_context
from app.chat_tracing import trace_chat_request
from app.prompts import SYSTEM_PROMPT
from app.agent.tools import get_tool_schema
from weakref import WeakValueDictionary

import asyncio
import json
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])
session_locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()


def get_session_lock(session_id: str) -> asyncio.Lock:
    lock = session_locks.get(session_id)
    if lock is None:
        lock = asyncio.Lock()
        session_locks[session_id] = lock
    return lock


async def prepare_session(session_id: str | None) -> str:
    if session_id is None:
        return await asyncio.to_thread(create_session)
    if not await asyncio.to_thread(session_exists, session_id):
        raise HTTPException(status_code=404, detail="会话不存在")
    return session_id


def prepare_context(history: list[dict], new_messages: list[dict], request: Request):
    settings = request.app.state.settings
    try:
        return build_chat_context(
            history,
            new_messages,
            SYSTEM_PROMPT,
            get_tool_schema(),
            settings.CHAT_HISTORY_MAX_TURNS,
            settings.CHAT_CONTEXT_TOKEN_BUDGET,
        )
    except ValueError as error:
        raise HTTPException(status_code=413, detail=str(error)) from error


@router.post("", response_model=ChatResponse)
async def chat(req: AgentRequest, request: Request):
    session_id = await prepare_session(req.session_id)
    new_messages = [m.model_dump() for m in req.messages]
    metadata = {"model": request.app.state.model_name}
    with trace_chat_request(
        request, "chat", session_id, new_messages, metadata
    ) as span:
        async with get_session_lock(session_id):
            history = await asyncio.to_thread(load_history, session_id)
            messages = prepare_context(history, new_messages, request)
            generated_messages = []
            reply, steps, _ = await agent_loop(
                client=request.app.state.llm_client,
                model_name=request.app.state.model_name,
                messages=messages,
                max_steps=request.app.state.max_steps,
                mode="chat",
                output_messages=generated_messages,
                langfuse_client=getattr(request.app.state, "langfuse", None),
            )
            if generated_messages:
                await asyncio.to_thread(
                    save_messages, session_id, new_messages + generated_messages
                )
            if span is not None:
                span.update(output=reply, metadata={**metadata, "steps": steps})
    return ChatResponse(reply=reply, steps=steps, session_id=session_id)

# @router.post("/stream")
# async def chat_sse(req: AgentRequest, request: Request):
#     async def generate():
#         try:
#             async for chunk in chat_stream(
#                         client=request.app.state.llm_client,
#                         model_name=request.app.state.model_name,
#                         messages = [m.model_dump() for m in req.messages]
#                     ):
#                 yield f"data: {json.dumps({'delta': chunk}, ensure_ascii=False)}\n\n"
#
#             yield f"data: [DONE]\n\n"
#         except Exception as e:
#             yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
#
#     return StreamingResponse(
#         generate(),
#         media_type="text/event-stream",
#     )

@router.post("/stream")
async def chat_sse(
    req: AgentRequest, request: Request, strategy: Literal["A", "B"] = "B"
):
    session_id = await prepare_session(req.session_id)
    lock = get_session_lock(session_id)
    new_messages = [m.model_dump() for m in req.messages]

    async def generate_locked():
        try:
            metadata = {"model": request.app.state.model_name, "strategy": strategy}
            with trace_chat_request(
                request, "chat-stream", session_id, new_messages, metadata
            ) as span:
                history = await asyncio.to_thread(load_history, session_id)
                messages = prepare_context(history, new_messages, request)
                session_event = {"type": "session", "session_id": session_id}
                yield f"data: {json.dumps(session_event, ensure_ascii=False)}\n\n"
                generated_messages = []
                steps = 0
                async with aclosing(agent_loop_stream(
                        client=request.app.state.llm_client,
                        model_name=request.app.state.model_name,
                        messages=messages,
                        max_steps=request.app.state.max_steps,
                        mode="final_stream" if strategy == "A" else "all_stream",
                        output_messages=generated_messages,
                        langfuse_client=getattr(request.app.state, "langfuse", None),
                )) as agent_events:
                    async for event in agent_events:
                        if event["type"] == "step":
                            steps = event["step"]
                        if event["type"] == "error" and span is not None:
                            span.update(
                                level="ERROR",
                                status_message=event.get("message", "Agent execution failed"),
                                metadata={**metadata, "steps": steps, "error_type": "AgentError"},
                            )
                        if event["type"] == "done":
                            await asyncio.to_thread(
                                save_messages, session_id, new_messages + generated_messages
                            )
                            if span is not None:
                                span.update(
                                    output=generated_messages[-1]["content"],
                                    metadata={**metadata, "steps": event["steps"]},
                                )
                        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except HTTPException as error:
            event = {"type": "error", "message": error.detail}
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception:
            logger.exception("流式聊天执行失败，session_id=%s", session_id)
            error = {"type": "error", "message": "生成回答时错误"}
            yield f"data: {json.dumps(error, ensure_ascii=False)}\n\n"

    async def generate():
        async with lock:
            async with aclosing(generate_locked()) as events:
                async for chunk in events:
                    yield chunk

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
    )
