# src/query_test.py
import argparse
from pathlib import Path

from llama_index.core import StorageContext, Settings, load_index_from_storage
from src.settings import settings


def configure_provider(use_llm: bool):
    """Configure embeddings + (optional) LLM."""
    from llama_index.embeddings.ollama import OllamaEmbedding
    Settings.embed_model = OllamaEmbedding(model_name=settings.OLLAMA_EMBED)

    if not use_llm:
        Settings.llm = None
        return

    # LLM setup (Ollama by default)
    from llama_index.llms.ollama import Ollama
    Settings.llm = Ollama(
        model=settings.OLLAMA_LLM,      # e.g., "llama3.2:3b"
        temperature=0.2,
        request_timeout=600.0,          # allow up to 10 minutes for cold start
        keep_alive="10m",
    )


def load_index():
    persist_dir = Path(settings.CHROMA_DB_DIR)
    if not persist_dir.exists():
        raise FileNotFoundError(f"Persist dir not found at {persist_dir} — run ingestion first.")
    storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
    return load_index_from_storage(storage_context)


def main():
    parser = argparse.ArgumentParser(description="Query the cafe RAG index.")
    parser.add_argument("question", type=str, help="Your question, e.g., 'What are vegan options under $6?'")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM; print retrieved context only.")
    parser.add_argument("--k", type=int, default=3, help="similarity_top_k (default: 3)")
    args = parser.parse_args()

    configure_provider(use_llm=not args.no_llm)
    index = load_index()

    if Settings.llm is None:
        # Retrieval-only mode
        retriever = index.as_retriever(similarity_top_k=args.k)
        nodes = retriever.retrieve(args.question)
        print("\n🔎 Top retrieved chunks:")
        for i, node in enumerate(nodes, 1):
            print(f"\n[{i}] -----------------------------")
            print(node.get_content())
        print("\n(LLM is disabled; run without --no-llm once you have a local Ollama LLM running.)")
        return

    # Full RAG with system prompt
    system_prefix = (
        "You are a helpful cafe menu assistant. "
        "Always answer in natural sentences, not just numbers. "
        "List item names with their prices clearly. "
        "If nothing matches, say it's not available."
    )

    query_engine = index.as_query_engine(
        similarity_top_k=args.k,
        response_mode="compact",  # concise but full sentences
    )
    try:
        answer = query_engine.query(f"{system_prefix}\n\nUser: {args.question}")
        print("\n💬 Answer:\n")
        print(str(answer))
        if getattr(answer, "source_nodes", None):
            print("\n📚 Sources:")
            for sn in answer.source_nodes:
                print(f"- score={sn.score:.3f} | {sn.node.get_text()[:120].replace('\n',' ')}")
    except Exception as e:
        print("\n⚠️ LLM call failed. Details:\n", repr(e))
        print("Falling back to retrieval-only...\n")
        retriever = index.as_retriever(similarity_top_k=args.k)
        nodes = retriever.retrieve(args.question)
        for i, node in enumerate(nodes, 1):
            print(f"\n[{i}] -----------------------------")
            print(node.get_content())


if __name__ == "__main__":
    main()
