import os
from dotenv import load_dotenv
from llama_index.core import Settings

load_dotenv()


class SettingsConfig:
    CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./storage")

    # OpenAI (optional, only used if PROVIDER=openai and key is set)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

    # Ollama (default)
    OLLAMA_LLM = os.getenv("OLLAMA_LLM", "llama3.2:3b")
    OLLAMA_EMBED = os.getenv("OLLAMA_EMBED", "nomic-embed-text")

    # Force provider via .env (default = ollama)
    PROVIDER = os.getenv("PROVIDER", "ollama").lower()

    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))


settings = SettingsConfig()


def configure_provider(use_llm: bool = True):
    """
    Configure embeddings (always) + optional LLM.
    """
    # Embeddings
    if settings.PROVIDER == "openai" and settings.OPENAI_API_KEY:
        from llama_index.embeddings.openai import OpenAIEmbedding
        Settings.embed_model = OpenAIEmbedding(
            model=settings.OPENAI_EMBED_MODEL,
            api_key=settings.OPENAI_API_KEY,
        )
    else:
        from llama_index.embeddings.ollama import OllamaEmbedding
        Settings.embed_model = OllamaEmbedding(model_name=settings.OLLAMA_EMBED)

    # LLM
    if not use_llm:
        Settings.llm = None
        return

    if settings.PROVIDER == "openai" and settings.OPENAI_API_KEY:
        from llama_index.llms.openai import OpenAI
        Settings.llm = OpenAI(
            model=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0.2,
            timeout=60.0,
        )
    else:
        from llama_index.llms.ollama import Ollama
        Settings.llm = Ollama(
            model=settings.OLLAMA_LLM,
            temperature=0.2,
            request_timeout=600.0,  # longer to handle cold starts
            keep_alive="10m",
        )

print(f"🔧 Provider set to: {settings.PROVIDER}")
print(f"🔧 LLM: {settings.OLLAMA_LLM}, Embed: {settings.OLLAMA_EMBED}")
