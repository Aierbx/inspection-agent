from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# 获取项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    API_KEY: str
    BASE_URL: str
    model_name: str = "deepseek-chat"

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