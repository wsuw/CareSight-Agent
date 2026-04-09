import os
from dotenv import load_dotenv

# 加载 .env 环境变量
load_dotenv()


class Settings:
    """系统全局配置类"""

    # LLM Settings
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "").strip('"').strip("'")
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))

    # LangSmith Settings (Optional)
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
    LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"

    # RAG Settings
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "embedding-gemma")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Tools Settings
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")

    # Persistence Settings
    POSTGRES_URL = os.getenv(
        "POSTGRES_URI",
        "",
    )


# 单例配置对象
settings = Settings()
