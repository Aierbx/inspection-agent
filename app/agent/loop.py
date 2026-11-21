import json
import asyncio
from time import perf_counter
from datetime import datetime, timezone
from contextlib import aclosing

from app.prompts import SYSTEM_PROMPT, REPORT_SYSTEM_PROMPT
from app.agent.tools import get_tool_schema
from app.config import Settings
from app.llm import chat_stream
from app.chat_tracing import trace_operation
from app.agent.tracing import execute_tool, record_generation

settings = Settings()

async def agent_loop(
    client,
    model_name: str,
    messages: list[dict],
    max_steps: int,
    mode: str,
    output_messages: list[dict] | None = None,
    langfuse_client=None,
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

        with trace_operation(
            langfuse_client, "llm-call", "generation",
            list(conversation), {"step": steps, "stream": False}, model=model_name,
        ) as generation:
            response = await client.chat.completions.create(
                model=model_name,
                messages=conversation,
                tools=tools,
                tool_choice="auto",
            )
            record_generation(
                generation,
                response.choices[0].message.model_dump(exclude_none=True),
                response.usage,
            )

        assistant_message = response.choices[0].message

        conversation.append(assistant_message.model_dump(exclude_none=True))

        tool_calls = assistant_message.tool_calls

        if not tool_calls:
            reply = assistant_message.content
            if output_messages is not None:
                output_messages.extend(conversation[1 + len(messages):])
            break

        if steps >= max_steps:
            reply = "已达到最大工具调用轮数，任务未能全部完成。"
            break

        for tool_call in tool_calls:
            tool_res = await execute_tool(
                tool_call.function.name, tool_call.function.arguments,
                settings.TOOL_TIMEOUT_SECONDS, langfuse_client, steps,
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
                    })

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
            yield {"type": "delta", "text": delta.content}

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

async def agent_loop_stream(
    client,
    model_name: str,
    messages: list[dict],
    max_steps: int,
    mode: str,
    output_messages: list[dict] | None = None,
    langfuse_client=None,
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

    async def track_stream(response, generation):
        total_tokens = None
        first_output = True
        try:
            async for chunk in response:
                if chunk.usage is not None:
                    total_tokens = chunk.usage.total_tokens
                    record_generation(generation, usage=chunk.usage)
                if first_output and chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta.content or delta.tool_calls:
                        first_output = False
                        if generation is not None:
                            generation.update(completion_start_time=datetime.now(timezone.utc))
                yield chunk
            call_tokens.append(total_tokens)
        finally:
            close = getattr(response, "close", None)
            if close is not None:
                await close()

    def done_event():
        if output_messages is not None:
            output_messages.extend(conversation[1 + len(messages):])
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
            assistant_message = None
            answer_parts = []
            with trace_operation(
                langfuse_client, "llm-call", "generation",
                list(conversation), {"step": steps, "stream": True}, model=model_name,
            ) as generation:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=conversation,
                    tools=get_tool_schema(),
                    tool_choice="auto",
                    stream=True,
                    stream_options={"include_usage": True},
                )
                async with aclosing(track_stream(response, generation)) as chunks:
                    async with aclosing(_parse_stream_response(chunks)) as events:
                        async for event in events:
                            if event["type"] == "tool_calls":
                                assistant_message = event["message"]
                            else:
                                answer_parts.append(event["text"])
                                if first_delta_ms is None:
                                    first_delta_ms = round((perf_counter() - started_at) * 1000, 2)
                                yield event
                record_generation(
                    generation,
                    assistant_message or {"role": "assistant", "content": "".join(answer_parts)},
                )

            if assistant_message is None:
                conversation.append({"role": "assistant", "content": "".join(answer_parts)})
                yield done_event()
                return

        else:
            with trace_operation(
                langfuse_client, "llm-call", "generation",
                list(conversation), {"step": steps, "stream": False}, model=model_name,
            ) as generation:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=conversation,
                    tools=get_tool_schema(),
                    tool_choice="auto",
                    stream=False,
                )
                call_tokens.append(response.usage.total_tokens if response.usage else None)
                message = response.choices[0].message
                record_generation(
                    generation, message.model_dump(exclude_none=True), response.usage
                )

            if not message.tool_calls:
                answer_parts = []
                with trace_operation(
                    langfuse_client, "llm-call", "generation",
                    list(conversation), {"step": steps, "stream": True}, model=model_name,
                ) as generation:
                    response = await client.chat.completions.create(
                        model=model_name,
                        messages=conversation,
                        stream=True,
                        stream_options={"include_usage": True},
                    )
                    async with aclosing(track_stream(response, generation)) as chunks:
                        async for chunk in chunks:
                            if chunk.choices and (text := chunk.choices[0].delta.content):
                                answer_parts.append(text)
                                if first_delta_ms is None:
                                    first_delta_ms = round((perf_counter() - started_at) * 1000, 2)
                                yield {"type": "delta", "text": text}
                    record_generation(
                        generation, {"role": "assistant", "content": "".join(answer_parts)}
                    )

                conversation.append({"role": "assistant", "content": "".join(answer_parts)})
                yield done_event()
                return

            assistant_message = message.model_dump(exclude_none=True)

        tool_calls: list[dict] = assistant_message["tool_calls"]
        if steps >= max_steps:
            yield {"type": "error", "message": "已达到最大工具调用轮数。"}
            return

        conversation.append(assistant_message)

        for tool_call in tool_calls:
            tool_res = await execute_tool(
                tool_call["function"]["name"], tool_call["function"]["arguments"],
                settings.TOOL_TIMEOUT_SECONDS, langfuse_client, steps,
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
                    })

            conversation.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(tool_res, ensure_ascii=False)
                }
            )

        steps += 1
