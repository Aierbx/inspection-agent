import json

from app.prompts import SYSTEM_PROMPT
from app.agent.tools import get_tool_schema, dispatch_tool, TOOL_REGISTRY

tools = get_tool_schema()

async def agent_loop(
    client,
    model_name: str,
    messages: list[dict],
    max_steps: int,
) -> tuple[str | None, int]:
    steps = 1

    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *messages,
    ]

    while True:
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
                tool_res = await dispatch_tool(tool_call.function.name, arguments)

            except json.JSONDecodeError:
                tool_res = {
                    "success": False,
                    "error": "invalid_tool_arguments_json",
                }

            conversation.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(tool_res, ensure_ascii=False)
                }
            )


        steps += 1

    return reply, steps

