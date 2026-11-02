from app.prompts import SYSTEM_PROMPT

async def agent_loop(
    client,
    model_name: str,
    messages: list[dict],
) -> tuple[str | None, int]:
    max_step = 5
    steps = 1

    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *messages,
    ]

    while True:
        response = await client.chat.completions.create(
            model=model_name,
            messages=conversation,
        )

        assistant_message = response.choices[0].message

        conversation.append(assistant_message.model_dump(exclude_none=True))

        tool_calls = assistant_message.tool_calls

        if not tool_calls:
            reply = assistant_message.content
            break

        if steps >= max_step:
            reply = "已达到最大工具调用轮数，任务未能全部完成。"
            break

        steps += 1

    return reply, steps

