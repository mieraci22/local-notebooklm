"""
Local NotebookLM — Streamlit Chat Interface
Upload PDFs and chat with them using a fully local LLM.
"""

import os
import tempfile

import streamlit as st

from rag_engine import RAGEngine
import config


# ─── Page Config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Local NotebookLM",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Clean up the default Streamlit look */
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stChatMessage {
        padding: 1rem 1.5rem;
        border-radius: 12px;
    }
    .sidebar-stats {
        background: #1a1a2e;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .sidebar-stats p {
        margin: 0.25rem 0;
        color: #e0e0e0;
    }
    .file-badge {
        display: inline-block;
        background: #16213e;
        color: #64ffda;
        padding: 0.25rem 0.75rem;
        border-radius: 16px;
        margin: 0.25rem;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ─── Initialize Engine (cached) ───────────────────────────────────
@st.cache_resource
def get_engine():
    """Initialize RAG engine once and cache it."""
    return RAGEngine()


def check_ollama_running() -> bool:
    """Verify Ollama is running and models are available."""
    try:
        import requests
        resp = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            # Check for required models (handle tag variations)
            has_llm = any(config.LLM_MODEL.split(":")[0] in m for m in models)
            has_embed = any(config.EMBEDDING_MODEL in m for m in models)
            return has_llm and has_embed
    except Exception:
        pass
    return False


# ─── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.title("📚 Local NotebookLM")
    st.caption(config.APP_SUBTITLE)
    st.divider()

    # Ollama status check
    if not check_ollama_running():
        st.error("⚠️ Ollama not detected or models missing!")
        st.code(
            "# Run these commands:\n"
            "ollama serve &\n"
            f"ollama pull {config.LLM_MODEL}\n"
            f"ollama pull {config.EMBEDDING_MODEL}",
            language="bash",
        )
        st.stop()
    else:
        st.success("✅ Ollama connected")

    st.divider()

    # File Upload
    st.subheader("📄 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Drop your PDFs here",
        type=config.ALLOWED_EXTENSIONS,
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        engine = get_engine()
        for uploaded_file in uploaded_files:
            file_bytes = uploaded_file.read()
            uploaded_file.seek(0)  # Reset for potential re-read

            # Save to temp file for PyMuPDF
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".pdf", prefix=uploaded_file.name.replace(".pdf", "_")
            ) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

            try:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    result = engine.ingest_pdf(tmp_path, file_bytes=file_bytes)

                if result["status"] == "success":
                    st.success(
                        f"✅ **{result['filename']}**\n\n"
                        f"{result['pages']} pages → {result['chunks']} chunks"
                    )
                elif result["status"] == "skipped":
                    st.info(f"ℹ️ **{result['filename']}** already loaded")
                else:
                    st.error(f"❌ {result['message']}")
            finally:
                os.unlink(tmp_path)

    st.divider()

    # Collection Stats
    st.subheader("📊 Knowledge Base")
    engine = get_engine()
    stats = engine.get_collection_stats()

    if stats["total_files"] > 0:
        st.metric("Documents", stats["total_files"])
        st.metric("Chunks indexed", stats["total_chunks"])

        with st.expander("Loaded files"):
            for f in stats["files"]:
                st.markdown(f"- `{f}`")

        if st.button("🗑️ Clear all documents", type="secondary", use_container_width=True):
            engine.clear_collection()
            st.cache_resource.clear()
            st.rerun()
    else:
        st.info("No documents loaded yet. Upload PDFs to get started.")

    st.divider()

    # Model info
    with st.expander("⚙️ Model Info"):
        st.markdown(f"""
        - **LLM:** `{config.LLM_MODEL}`
        - **Embeddings:** `{config.EMBEDDING_MODEL}`
        - **Chunk size:** {config.CHUNK_SIZE} chars
        - **Top-K retrieval:** {config.TOP_K}
        - **Temperature:** {config.TEMPERATURE}
        """)


# ─── Main Chat Area ───────────────────────────────────────────────
st.header(config.APP_TITLE)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    engine = get_engine()
    stats = engine.get_collection_stats()

    # Check if documents are loaded
    if stats["total_files"] == 0:
        st.warning("⬅️ Upload PDF documents first using the sidebar.")
    else:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and stream response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            try:
                for chunk in engine.query_stream(prompt):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")

                response_placeholder.markdown(full_response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                response_placeholder.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )

        # Show sources in expander
        with st.expander("📖 View sources"):
            source_docs = engine.retrieve(prompt)
            for i, doc in enumerate(source_docs, 1):
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "?")
                st.markdown(f"**Source {i}** — `{source}`, Page {page}")
                st.markdown(f"> {doc.page_content[:300]}...")
                st.divider()
