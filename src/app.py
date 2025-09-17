# src/app.py
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.core.base.response.schema import Response

from src.settings import settings, configure_provider

# --- Configure embeddings at startup so /health never touches OpenAI by accident ---
configure_provider(use_llm=False)

# ---------- Load index once ----------
INDEX = None

def get_index():
    global INDEX
    if INDEX is not None:
        return INDEX
    persist_dir = Path(settings.CHROMA_DB_DIR)
    if not persist_dir.exists():
        raise RuntimeError(
            f"Persist dir not found at {persist_dir}. "
            "Run ingestion first:  python -m src.ingest"
        )
    storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
    INDEX = load_index_from_storage(storage_context)
    return INDEX


# ---------- FastAPI app ----------
app = FastAPI(title="Cafe RAG Bot", version="0.1.0")

# CORS (dev-friendly)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str
    k: int = 3
    use_llm: bool = True  # False = retrieval-only


class SourceNode(BaseModel):
    score: Optional[float] = None
    metadata: Dict[str, Any] = {}
    preview: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceNode]


@app.get("/health")
def health():
    try:
        _ = get_index()
        return {
            "status": "ok",
            "provider": settings.PROVIDER,
            "llm": settings.OLLAMA_LLM if settings.PROVIDER == "ollama" else settings.OPENAI_MODEL,
            "embed": settings.OLLAMA_EMBED if settings.PROVIDER == "ollama" else settings.OPENAI_EMBED_MODEL,
        }
    except Exception as e:
        return {
            "status": "error",
            "detail": str(e),
            "provider": settings.PROVIDER,
        }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # Configure embeddings + LLM for this request
    configure_provider(use_llm=req.use_llm)

    index = get_index()

    if Settings.llm is None:
        # Retrieval-only mode
        retriever = index.as_retriever(similarity_top_k=req.k)
        nodes = retriever.retrieve(req.question)
        sources = [
            SourceNode(
                score=getattr(n, "score", None),
                metadata=n.metadata,
                preview=n.get_content()[:160].replace("\n", " "),
            )
            for n in nodes
        ]
        return ChatResponse(
            answer="(LLM disabled — showing top results)",
            sources=sources,
        )

    # Full RAG
    system_prefix = (
        "You are a helpful cafe menu assistant. "
        "Always answer in natural sentences, not just numbers. "
        "List item names with their prices clearly. "
        "If nothing matches, say it's not available."
    )

    query_engine = index.as_query_engine(similarity_top_k=req.k, response_mode="compact")
    resp: Response = query_engine.query(f"{system_prefix}\n\nUser: {req.question}")

    srcs = []
    for sn in getattr(resp, "source_nodes", []) or []:
        srcs.append(SourceNode(
            score=getattr(sn, "score", None),
            metadata=sn.node.metadata or {},
            preview=sn.node.get_text()[:160].replace("\n", " "),
        ))

    return ChatResponse(answer=resp.response.strip(), sources=srcs)


# ---------- Static UI (served at "/") ----------
PUBLIC_DIR = Path(__file__).parent.parent / "public"
PUBLIC_DIR.mkdir(exist_ok=True)  # ensure folder exists
app.mount("/", StaticFiles(directory=str(PUBLIC_DIR), html=True), name="ui")


# Optional: run via `python -m src.app` (no need if using `uvicorn src.app:app`)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app:app", host=settings.HOST, port=settings.PORT, reload=True)
