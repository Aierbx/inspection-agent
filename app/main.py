from fastapi import FastAPI
from openai import AsyncOpenAI

from app.config import Settings
from app.routers.chat import router as chat_router

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()

    # 服务启动时初始化
    client = AsyncOpenAI(
        api_key=settings.API_KEY,
        base_url=settings.BASE_URL,
    )

    app.state.llm_client = client
    app.state.max_steps = settings.MAX_STEPS
    app.state.model_name = settings.model_name

    try:
        yield
    finally:
        await client.close()

app = FastAPI(lifespan=lifespan, debug=True,)

app.include_router(chat_router)