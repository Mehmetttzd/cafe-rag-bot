# Café RAG Bot

A tiny Retrieval-Augmented Generation chatbot that answers menu questions from a CSV.  
Runs fully **local** using **Ollama**, **FastAPI**, and **LlamaIndex**. Answers are grounded in your menu and return source snippets for transparency.

## 🎯 Why this project

- **Real-world RAG** on a tabular dataset (CSV → vector index).
- **Local inference**: no external APIs required.
- **Production shape**: ingestion pipeline, API, and minimal web UI.

---

## 🧩 Architecture

```mermaid
flowchart LR
  A[User / Web UI] -->|JSON /chat| B(FastAPI)
  B -->|Query| C[LlamaIndex Query Engine]
  C -->|Top-K Retrieval| D[(Vector Store\n(storage/))]
  D -->|Embeddings| E[Ollama Embedding\nnomic-embed-text]
  C -->|Synthesis| F[Ollama LLM\nllama3.2:3b]
  F -->|Answer + Sources| B -->|JSON| A

  subgraph Ingestion (offline)
    G[menu.csv] --> H[Row → Document (structured)]
    H --> E
    H --> D
  end
