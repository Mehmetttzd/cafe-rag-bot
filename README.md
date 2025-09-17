# Café RAG Bot

A tiny Retrieval-Augmented Generation chatbot that answers café menu questions from a CSV.  
Runs fully **local** using **Ollama**, **LlamaIndex**, and **FastAPI**. Answers are grounded in the menu and include source snippets for transparency.

---

## 👀 Preview

> Put a screenshot or GIF at `public/screenshot.png` so GitHub shows it here.

![Café RAG Bot – UI](public/screenshot.png)

---

## 🎯 Highlights (for recruiters)

- **Real RAG on tabular data** (CSV → vector index) with a small, readable codebase.
- **Local-first, privacy-friendly**: Ollama (`llama3.2:3b`, `nomic-embed-text`) for LLM + embeddings.
- **Production shape**: ingestion pipeline, API, minimal web UI, health check.

---

## ⚡ Quickstart

```bash
# 1) Pull local models (first time)
ollama pull llama3.2:3b
ollama pull nomic-embed-text

# 2) Install Python deps
pip install -r requirements.txt

# 3) Ingest CSV → build vector index under ./storage
python -m src.ingest

# 4) Run API + UI (Windows-friendly)
python -m uvicorn src.app:app --host 0.0.0.0 --port 5000 --reload
```

Endpoints

UI: http://localhost:5000/

Swagger: http://localhost:5000/docs

Health: http://localhost:5000/health

🧩 Architecture
flowchart LR
  A[Web UI] -->|JSON /chat| B(FastAPI)
  B -->|Query| C[LlamaIndex Query Engine]
  C -->|Top-K Retrieval| D[(Vector Store\nstorage/)]
  D -->|Embeddings| E[Ollama Embedding\nnomic-embed-text]
  C -->|Synthesis| F[Ollama LLM\nllama3.2:3b]
  F -->|Answer + Sources| B -->|JSON| A

  subgraph Ingestion (offline)
    G[data/menu.csv] --> H[Row → Structured Document]
    H --> E
    H --> D
  end
  🛠️ Tech

LLM / Embeddings: Ollama (llama3.2:3b, nomic-embed-text)

RAG Framework: LlamaIndex (indexing, retrieval, synthesis)

API: FastAPI + Uvicorn

Data & Storage: CSV → persisted local vector store (./storage)

UI: Static HTML/JS served from /public (right-side professional menu)

📁 Project Structure

cafe-rag-bot/
  data/
    menu.csv                 # Source dataset
  public/                    # Minimal browser UI
    index.html               # Chat UI
    menu.html                # Right-side professional sidebar
  src/
    __init__.py
    settings.py              # Env + provider config (Ollama/OpenAI switchable)
    ingest.py                # CSV → documents → embeddings → persisted index
    query_test.py            # CLI tester (with/without LLM)
    app.py                   # FastAPI (/health, /chat) + static UI hosting
  storage/                   # Vector index (created by ingest)
  .env                       # Local config (see below)
  requirements.txt

  🔍 How it works (short)

Ingestion (src/ingest.py)
Parses menu.csv, converts each row into a self-contained document (item, category, ingredients, price), embeds, and persists a vector index to ./storage.

Querying (src/app.py)
/chat retrieves top-K chunks and uses the LLM to synthesize a concise, grounded answer.
Returns: answer + sources[] (score, metadata preview).

Provider config (src/settings.py)
configure_provider() wires embeddings + LLM. Defaults to Ollama; switch to OpenAI by setting PROVIDER=openai + key.

UI (public/)
Clean chat UI with a right-side professional menu (filters, quick prompts, collapsible).

🧰 Troubleshooting

OpenAI error in /health → ensure .env has PROVIDER=ollama and no OPENAI_*; fully restart terminal & server.

“Persist dir not found” → run ingestion: python -m src.ingest.

First LLM call slow → model cold start; try once more (timeouts increased).

Windows → use: python -m uvicorn src.app:app --host 0.0.0.0 --port 5000.

🗺️ Roadmap

 Metadata-aware retrieval (e.g., filter by vegan=yes)

 Streaming responses

 React/Vite front-end

 Docker compose (Ollama + API + UI)