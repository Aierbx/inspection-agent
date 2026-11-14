import json
import asyncio
from time import perf_counter

from app.prompts import SYSTEM_PROMPT, REPORT_SYSTEM_PROMPT
from app.agent.tools import get_tool_schema, dispatch_tool, TOOL_REGISTRY
from app.config import Settings
from app.llm import chat_stream

settings = Settings()

async def agent_loop(
    client,
    model_name: str,
    messages: list[dict],
    max_steps: int,
    mode: str
) -> tuple[str | None, int, list[dict]]:
    steps = 1
    retrieved_sources = []

    if mode == "chat":
        conversation = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *messages,
        ]
    else:
        conversation = [
            {"role": "system", "content": REPORT_SYSTEM_PROMPT},
            *messages,
        ]

    while True:
        tools = get_tool_schema()

        response = await client.chat.completions.create(
            model=model_name,
            messages=conversation,
            tools=tools,
            tool_choice="auto",
        )

        assistant_message = response.choices[0].message

        conversation.append(assistant_message.model_dump(exclude_none=True))

        tool_calls = assistant_message.tool_calls

        if not tool_calls:
            reply = assistant_message.content
            break

        if steps >= max_steps:
            reply = "已达到最大工具调用轮数，任务未能全部完成。"
            break

        for tool_call in tool_calls:
            try:
                arguments = json.loads(tool_call.function.arguments)

                tool_res = await asyncio.wait_for(
                    dispatch_tool(tool_call.function.name, arguments),
                    timeout=settings.TOOL_TIMEOUT_SECONDS,  # config 里新增，默认 30
                )
                if tool_call.function.name == "search_knowledge_base_tool" and tool_res.get("success"):
                    for chunk in tool_res["top_k_chunks"]:
                        retrieved_sources.append({
                            "chunk_id": chunk["chunk_id"],
                            "text": chunk["text"],
                            "document_id": chunk["metadata"]["document_id"],
                            "chapter_id": chunk["chapter_id"],
                            "section_id": chunk["section_id"],
                            "citation_eligible": chunk["citation_eligible"],
                        }
                        )

            except json.JSONDecodeError:
                tool_res = {
                    "success": False,
                    "error": "invalid_tool_arguments_json",
                }

            except asyncio.TimeoutError:
                tool_res = {
                    "success": False,
                    "error": "tool_timeout"
                }

            conversation.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(tool_res, ensure_ascii=False)
                }
            )


        steps += 1

    return reply, steps, retrieved_sources

async def  _parse_stream_response(response):

    text_parts = []
    tool_calls_by_index = {}

    async for chunk in response:
        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta
        if delta.content:
            text_parts.append(delta.content)

        if delta.tool_calls:
            for item in delta.tool_calls or []:
                call = tool_calls_by_index.setdefault(
                        item.index,
                        {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        },
                    )

                # 向tool_call_dict中填工具调用数据
                if item.id:
                    call["id"] = item.id

                if item.function and item.function.name:
                    call["function"]["name"] = item.function.name

                if item.function and item.function.arguments:
                    call["function"]["arguments"] += item.function.arguments

    if tool_calls_by_index:
        yield {
            "type": "tool_calls",
            "message": {
                "role": "assistant",
                "content": "".join(text_parts) or None,
                "tool_calls": [
                    tool_calls_by_index[index]
                    for index in sorted(tool_calls_by_index)
                ],
            },
        }
        return

    for text in text_parts:
        yield {"type": "delta", "text": text}

async def agent_loop_stream(
    client,
    model_name: str,
    messages: list[dict],
    max_steps: int,
    mode: str
):
    started_at = perf_counter()
    first_delta_ms = None
    call_tokens: list[int | None] = []
    steps = 1
    retrieved_sources = []

    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *messages,
    ]

    async def track_stream(response):
        total_tokens = None
        async for chunk in response:
            if chunk.usage is not None:
                total_tokens = chunk.usage.total_tokens
            yield chunk
        call_tokens.append(total_tokens)

    def done_event():
        return {
            "type": "done",
            "steps": steps,
            "citations": retrieved_sources,
            "metrics": {
                "server_ttft_ms": first_delta_ms,
                "llm_calls": len(call_tokens),
                "total_tokens": (
                    None if None in call_tokens
                    else sum(tokens for tokens in call_tokens if tokens is not None)
                ),
            },
        }

    while True:
        yield {"type": "step", "step": steps}

        if mode == "all_stream":
            response = await client.chat.completions.create(
                model=model_name,
                messages=conversation,
                tools=get_tool_schema(),
                tool_choice="auto",
                stream=True,
                stream_options={"include_usage": True},
            )

            assistant_message = None

            async for event in _parse_stream_response(track_stream(response)):
                if event["type"] == "tool_calls":
                    assistant_message = event["message"]
                else:
                    if first_delta_ms is None:
                        first_delta_ms = round((perf_counter() - started_at) * 1000, 2)
                    yield event

            if assistant_message is None:
                yield done_event()
                return

        else:
            response = await client.chat.completions.create(
                model=model_name,
                messages=conversation,
                tools=get_tool_schema(),
                tool_choice="auto",
                stream=False
            )

            call_tokens.append(response.usage.total_tokens if response.usage else None)
            message = response.choices[0].message

            if not message.tool_calls:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=conversation,
                    stream=True,
                    stream_options={"include_usage": True},
                )
                async for chunk in track_stream(response):
                    if chunk.choices and (text := chunk.choices[0].delta.content):
                        if first_delta_ms is None:
                            first_delta_ms = round((perf_counter() - started_at) * 1000, 2)
                        yield {"type": "delta", "text": text}

                yield done_event()
                return

            assistant_message = message.model_dump(exclude_none=True)

        tool_calls: list[dict] = assistant_message["tool_calls"]
        if steps >= max_steps:
            yield {"type": "error", "message": "已达到最大工具调用轮数。"}
            return

        conversation.append(assistant_message)

        for tool_call in tool_calls:
            try:
                arguments = json.loads(tool_call["function"]["arguments"])

                tool_res = await asyncio.wait_for(
                    dispatch_tool(tool_call["function"]["name"], arguments),
                    timeout=settings.TOOL_TIMEOUT_SECONDS,  # config 里新增，默认 30
                )
                if tool_call["function"]["name"] == "search_knowledge_base_tool" and tool_res.get("success"):
                    for chunk in tool_res["top_k_chunks"]:
                        retrieved_sources.append({
                            "chunk_id": chunk["chunk_id"],
                            "text": chunk["text"],
                            "document_id": chunk["metadata"]["document_id"],
                            "chapter_id": chunk["chapter_id"],
                            "section_id": chunk["section_id"],
                            "citation_eligible": chunk["citation_eligible"],
                        }
                        )

            except json.JSONDecodeError:
                tool_res = {
                    "success": False,
                    "error": "invalid_tool_arguments_json",
                }

            except asyncio.TimeoutError:
                tool_res = {
                    "success": False,
                    "error": "tool_timeout"
                }

            conversation.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(tool_res, ensure_ascii=False)
                }
            )

        steps += 1
