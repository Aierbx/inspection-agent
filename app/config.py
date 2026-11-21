from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# 获取项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    # 大模型配置与agent_loop循环上限
    API_KEY: str
    BASE_URL: str
    model_name: str = "deepseek-chat"
    MAX_STEPS: int = 10
    TOOL_TIMEOUT_SECONDS: float = 30.0

    # Langfuse 本地观测服务
    LANGFUSE_BASE_URL: str = "http://localhost:3000"
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = Field(default="", repr=False)

    # 历史会话存储路径
    SESSION_DB_PATH: Path = BASE_DIR / "data" / "sessions.db"
    CHAT_HISTORY_MAX_TURNS: int = Field(default=10, ge=0)
    CHAT_CONTEXT_TOKEN_BUDGET: int = Field(default=12000, gt=0)

    # 知识库文件路径与文档切块大小限制
    DATA_DIR: Path = BASE_DIR / "data"
    STANDARDS_DIR: Path = DATA_DIR / "standards"
    REPORTS_DIR: Path = DATA_DIR / "reports"
    RAG_CHUNK_MAX_CHARS: int = 1000
    RAG_CHUNK_OVERLAP_CHARS: int = 150

    # 向量库文件路径
    CHROMA_DIR: Path = DATA_DIR / "chroma"
    INDEX_MANIFEST_DIR: Path = DATA_DIR / "index_manifest"
    CATALOG_PATH: Path = DATA_DIR / "catalog.json"
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-zh-v1.5"
    EMBEDDING_MODEL_DIR: Path = DATA_DIR / "models"
    REINSPECTION_CONFIDENCE_THRESHOLD: float = 0.8

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

if __name__ == "__main__":
    import os

    print("配置文件路径:", BASE_DIR / ".env")
    print("文件是否存在:", os.path.exists(BASE_DIR / ".env"))

    settings = Settings()
