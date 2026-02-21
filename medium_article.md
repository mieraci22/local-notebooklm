# I Built a Free, Offline Alternative to NotebookLM — Here's How

## How I set up a local LLM on Apple Silicon and built a RAG-powered PDF chat app that runs entirely on my Mac Mini

---

*No API keys. No cloud. No subscription. Just your machine, your documents, and a local LLM.*

Google's NotebookLM is impressive — you upload documents and have an AI conversation with them. But it requires internet access, sends your data to Google's servers, and will inevitably hit usage limits or get paywalled.

I wanted the same thing, but fully local. So I built it.

This article walks through the entire process: setting up a local LLM on Apple Silicon, building a Retrieval-Augmented Generation (RAG) pipeline, and wrapping it in a clean chat interface. Everything runs on a Mac Mini M4 with 16GB of RAM.

---

## Why Go Local?

Before we dive in, here's why a local setup makes sense:

**Privacy.** Your documents never leave your machine. If you're working with sensitive data — medical records, legal documents, financial reports — this matters.

**Cost.** After the initial setup (free), every query is free. No API credits, no token limits, no monthly fees.

**Availability.** Works on a plane, in a coffee shop with bad WiFi, during an API outage. Your tools should work when you need them.

**Speed.** No network latency. On the M4, responses start generating in under 2 seconds.

---

## The Architecture

Here's what we're building:

```
User uploads PDF
    → PyMuPDF extracts text
    → LangChain splits into chunks
    → Ollama generates embeddings (nomic-embed-text)
    → ChromaDB stores vectors locally

User asks a question
    → ChromaDB retrieves relevant chunks
    → Chunks injected into prompt as context
    → Ollama generates answer (Llama 3.1 8B)
    → Streamlit displays response with source citations
```

The entire stack is open-source and runs locally:

| Component | Tool | Purpose |
|-----------|------|---------|
| LLM Runtime | Ollama | Runs models on Apple Silicon via Metal |
| Language Model | Llama 3.1 8B | Text generation (~35 tokens/sec on M4) |
| Embeddings | nomic-embed-text | Converts text to vectors for search |
| Vector Store | ChromaDB | Stores and queries embeddings locally |
| Orchestration | LangChain | Wires the RAG pipeline together |
| PDF Parser | PyMuPDF | Fast, reliable text extraction |
| Frontend | Streamlit | Chat UI with file upload |

---

## Step 1: Install Ollama and Pull Models

Ollama is the key piece — it lets you run open-source LLMs locally with a single command. On macOS, it leverages Metal for GPU acceleration on Apple Silicon, which means the M4's unified memory architecture gets fully utilized.

```bash
# Install Ollama
brew install ollama

# Start the service
ollama serve &

# Pull our LLM (4.7GB download, quantized to 4-bit)
ollama pull llama3.1:8b

# Pull our embedding model (274MB)
ollama pull nomic-embed-text
```

**Why Llama 3.1 8B?** With 16GB of RAM, this is the sweet spot. The model loads entirely into unified memory, leaving enough headroom for the embedding model, ChromaDB, and the OS. Larger models (13B+) would work but with slower generation and less room for document context.

**Why nomic-embed-text?** It runs locally through Ollama (no separate embedding server), produces high-quality 768-dimensional vectors, and is fast enough to embed hundreds of chunks per second.

Let's verify everything works:

```bash
# Quick test — you should get a response in ~1 second
ollama run llama3.1:8b "What is retrieval-augmented generation? Answer in 2 sentences."
```

If you see a coherent response, your local LLM is running.

---

## Step 2: The RAG Pipeline

RAG (Retrieval-Augmented Generation) is the technique that makes this work. Instead of hoping the LLM memorized your documents during training (it didn't), we:

1. **Parse** the PDF into text
2. **Chunk** the text into overlapping segments
3. **Embed** each chunk into a vector representation
4. **Store** vectors in a local database
5. **Retrieve** the most relevant chunks for each query
6. **Generate** an answer using the LLM with retrieved context

Here's the core engine (`rag_engine.py`):

```python
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma

class RAGEngine:
    def __init__(self):
        # Local embeddings via Ollama
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")

        # Local LLM via Ollama
        self.llm = ChatOllama(model="llama3.1:8b", temperature=0.1)

        # Text splitter with overlap for context continuity
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

        # Persistent local vector store
        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory="./chroma_db",
        )
```

### PDF Parsing

PyMuPDF handles the PDF extraction. It's significantly faster than alternatives like pdfplumber and handles most PDF formats reliably:

```python
def _extract_pdf_text(self, pdf_path: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        if text.strip():
            pages.append({
                "text": text,
                "page": page_num,
                "source": os.path.basename(pdf_path),
            })
    return pages
```

### Chunking Strategy

We use recursive character splitting with 1000-character chunks and 200-character overlap. The overlap is critical — without it, information that spans chunk boundaries gets lost. The recursive splitter tries to split on paragraph breaks first, then sentences, then words, preserving semantic coherence.

### Retrieval and Generation

When a user asks a question, we embed the query, find the top-5 most similar chunks via cosine similarity, and inject them into the prompt:

```python
def query_stream(self, question: str):
    # Retrieve relevant chunks
    context_docs = self.retrieve(question)
    context = self._format_context(context_docs)

    # Generate with context
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),  # Includes {context} placeholder
        ("human", "{question}"),
    ])

    chain = prompt | self.llm | StrOutputParser()
    for chunk in chain.stream({"context": context, "question": question}):
        yield chunk
```

The system prompt explicitly instructs the model to only answer from the provided context and to cite sources — this prevents hallucination and keeps responses grounded in your documents.

---

## Step 3: The Chat Interface

Streamlit gives us a production-quality chat UI with minimal code. The key features:

- **PDF upload** in the sidebar with automatic ingestion
- **Streaming responses** for real-time token generation
- **Source citations** in expandable sections below each response
- **Collection management** — see what's loaded, clear the database

```python
# Streaming response display
with st.chat_message("assistant"):
    response_placeholder = st.empty()
    full_response = ""

    for chunk in engine.query_stream(prompt):
        full_response += chunk
        response_placeholder.markdown(full_response + "▌")

    response_placeholder.markdown(full_response)
```

---

## Step 4: Run It

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/local-notebooklm.git
cd local-notebooklm

# One-line setup
chmod +x setup.sh && ./setup.sh

# Or manually:
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501`, upload a PDF, and start chatting.

---

## Performance on Mac Mini M4

Here's what I measured on my setup (M4, 16GB RAM, macOS Tahoe):

| Metric | Result |
|--------|--------|
| LLM generation speed | ~35 tokens/sec |
| PDF ingestion | ~50 pages/sec |
| End-to-end query latency | 2–4 seconds |
| Embedding throughput | ~100 chunks/sec |
| Memory usage (idle) | ~5.5 GB |
| Memory usage (generating) | ~7.2 GB |

The M4's unified memory architecture is the hero here — the model weights sit in the same memory pool as the CPU and GPU, so there's no transfer overhead.

---

## Tuning Tips

A few things I learned through experimentation:

**Chunk size matters.** 1000 characters with 200 overlap worked best for my documents (academic papers, technical docs). For short-form content like emails or meeting notes, try 500/100. For books, try 1500/300.

**Temperature at 0.1** keeps responses factual and grounded. Bump to 0.3–0.5 if you want more creative synthesis across documents.

**Top-K retrieval at 5** balances context coverage with prompt length. Going above 8 starts to dilute relevance and slow generation (more context = more tokens to process).

**Quantization tradeoffs.** Llama 3.1 8B at Q4_K_M (Ollama's default) is the right call for 16GB. Q5 gives marginally better quality but uses ~1GB more RAM and slows generation by ~15%.

---

## What's Next

This is a v1. Some directions I'm exploring:

- **Multi-modal support** — adding image/table extraction from PDFs using `unstructured`
- **Conversation memory** — maintaining context across chat turns with a sliding window
- **Multiple collections** — separate vector stores per project
- **Audio summaries** — local TTS to generate podcast-style summaries (the NotebookLM killer feature)
- **Hybrid search** — combining vector similarity with BM25 keyword search for better retrieval

---

## The Bigger Picture

The gap between cloud AI and local AI is closing fast. A year ago, running a capable LLM locally required expensive hardware and deep technical knowledge. Today, it's `brew install ollama && ollama pull llama3.1`.

For anyone working with sensitive documents — researchers, lawyers, healthcare professionals, financial analysts — the ability to chat with your documents without sending them to a third party is not a nice-to-have. It's a requirement.

The full source code is available on [GitHub](https://github.com/YOUR_USERNAME/local-notebooklm). Star it, fork it, make it better.

---

*If you found this useful, follow me on Medium for more practical AI engineering content. I write about building real tools with local models, RAG systems, and applied ML.*

*Currently pursuing an MS in Data Science at UVA while working as a Data Engineer. Previously 6 years in private equity analytics.*
