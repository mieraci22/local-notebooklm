"""
Configuration for Local NotebookLM
Tuned for Mac Mini M4 with 16GB RAM
"""

# ─── Model Configuration ───────────────────────────────────────────
LLM_MODEL = "llama3.1:8b"
EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL = "http://localhost:11434"

# ─── Chunking Configuration ────────────────────────────────────────
CHUNK_SIZE = 1000          # Characters per chunk
CHUNK_OVERLAP = 200        # Overlap between chunks for context continuity
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# ─── Retrieval Configuration ───────────────────────────────────────
TOP_K = 5                  # Number of chunks to retrieve per query
SCORE_THRESHOLD = 0.3      # Minimum relevance score (0-1, lower = more permissive)

# ─── Generation Configuration ──────────────────────────────────────
TEMPERATURE = 0.1          # Low temp for factual Q&A
MAX_TOKENS = 2048          # Max response length
CONTEXT_WINDOW = 8192      # Llama 3.1 context window

# ─── Vector Store ──────────────────────────────────────────────────
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "local_notebooklm"

# ─── System Prompt ─────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a helpful research assistant that answers questions 
based on the provided document context. 

Rules:
1. ONLY answer based on the provided context. If the context doesn't contain 
   enough information, say so clearly.
2. Cite specific sections or page numbers when possible.
3. Be concise but thorough.
4. If asked about something outside the documents, politely redirect to 
   document-related questions.
5. Use direct quotes from the source material when relevant, wrapped in 
   quotation marks.

Context from uploaded documents:
{context}
"""

# ─── UI Configuration ─────────────────────────────────────────────
APP_TITLE = "📚 Local NotebookLM"
APP_SUBTITLE = "Chat with your PDFs — 100% offline, 100% free"
MAX_FILE_SIZE_MB = 200
ALLOWED_EXTENSIONS = ["pdf"]
