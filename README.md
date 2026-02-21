# 🏠 Local NotebookLM — Offline RAG-Powered PDF Chat

A fully local, offline, free alternative to Google's NotebookLM. Chat with your PDF documents using a local LLM running on Apple Silicon via Ollama — no API keys, no cloud, no cost.

**Built for:** Mac Mini M4 (16GB RAM) running macOS Tahoe 26.2

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Streamlit   │────▶│  LangChain   │────▶│   Ollama    │
│  Chat UI     │     │  RAG Chain   │     │  (Llama 3.1)│
└─────────────┘     └──────┬───────┘     └─────────────┘
                           │
                    ┌──────▼───────┐
                    │   ChromaDB   │
                    │ Vector Store │
                    └──────────────┘
                           ▲
                    ┌──────┴───────┐
                    │  PDF Loader  │
                    │  + Chunker   │
                    └──────────────┘
```

## Tech Stack

| Component        | Tool                        | Why                                      |
|------------------|-----------------------------|------------------------------------------|
| LLM Runtime      | Ollama                      | Native Apple Silicon support, simple API  |
| LLM Model        | Llama 3.1 8B (Q4_K_M)      | Best quality/speed at 16GB RAM            |
| Embeddings       | nomic-embed-text            | Local, fast, high quality                 |
| Vector Store     | ChromaDB                    | Lightweight, no server, persistent        |
| Orchestration    | LangChain                   | Mature RAG primitives                     |
| PDF Parsing      | PyMuPDF                     | Fast, reliable extraction                 |
| Frontend         | Streamlit                   | Rapid prototyping, clean UI               |

## Quick Start

### 1. Install Ollama

```bash
# Download from https://ollama.ai or:
brew install ollama
```

### 2. Pull Models

```bash
# Start Ollama (runs as a background service)
ollama serve &

# Pull the LLM (4.7GB download)
ollama pull llama3.1:8b

# Pull the embedding model (274MB download)
ollama pull nomic-embed-text
```

### 3. Install Python Dependencies

```bash
cd local-notebooklm
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Run the App

```bash
streamlit run app.py
```

Open http://localhost:8501 and start uploading PDFs.

## Project Structure

```
local-notebooklm/
├── app.py                 # Streamlit chat UI
├── rag_engine.py          # RAG pipeline (ingest, embed, retrieve, generate)
├── config.py              # Model and chunking configuration
├── requirements.txt       # Python dependencies
├── setup.sh               # One-line setup script
├── chroma_db/             # Persistent vector store (auto-created)
└── README.md
```

## Performance on Mac Mini M4 (16GB)

- **Llama 3.1 8B:** ~35 tokens/sec generation
- **PDF ingestion:** ~50 pages/sec
- **Query latency:** 2-4 seconds end-to-end (retrieval + generation)
- **Embedding throughput:** ~100 chunks/sec with nomic-embed-text

## Configuration

Edit `config.py` to tune:
- Chunk size and overlap
- Number of retrieved chunks (top-k)
- Model selection
- Temperature and other generation params

## License

MIT
