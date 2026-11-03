from openai import AsyncOpenAI
from app.config import Settings

async def chat_stream(client, model_name, messages: list[dict]):
    stream = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        stream=True,
    )

    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta