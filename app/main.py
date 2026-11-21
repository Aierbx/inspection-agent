import asyncio
import logging

from fastapi import FastAPI
from openai import AsyncOpenAI
from langfuse import Langfuse

from app.config import Settings
from app.session_store import init_db
from app.routers.chat import router as chat_router
from app.routers.report import router as report_router

from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    await asyncio.to_thread(init_db)

    # 服务启动时初始化
    client = AsyncOpenAI(
        api_key=settings.API_KEY,
        base_url=settings.BASE_URL,
    )

    app.state.llm_client = client
    app.state.max_steps = settings.MAX_STEPS
    app.state.model_name = settings.model_name
    app.state.settings = settings
    app.state.langfuse = None
    langfuse_client = None

    try:
        if settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY:
            langfuse_client = Langfuse(
                public_key=settings.LANGFUSE_PUBLIC_KEY,
                secret_key=settings.LANGFUSE_SECRET_KEY,
                base_url=settings.LANGFUSE_BASE_URL,
            )
            app.state.langfuse = langfuse_client
            try:
                authenticated = await asyncio.to_thread(langfuse_client.auth_check)
                if authenticated:
                    logger.info("Langfuse 连接验证通过")
                else:
                    logger.warning("Langfuse 连接验证失败，请检查项目密钥和服务地址")
            except Exception:
                logger.warning("Langfuse 连接验证异常，请检查本地观测服务")
        yield
    finally:
        try:
            if langfuse_client is not None:
                await asyncio.to_thread(langfuse_client.shutdown)
        finally:
            await client.close()

app = FastAPI(lifespan=lifespan, debug=True,)

app.include_router(chat_router)
app.include_router(report_router)
