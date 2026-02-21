"""
RAG Engine for Local NotebookLM
Handles: PDF parsing → Chunking → Embedding → Storage → Retrieval → Generation
"""

import os
import hashlib
from typing import Generator

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import config


class RAGEngine:
    """End-to-end RAG pipeline using local models via Ollama."""

    def __init__(self):
        self.embeddings = OllamaEmbeddings(
            model=config.EMBEDDING_MODEL,
            base_url=config.OLLAMA_BASE_URL,
        )
        self.llm = ChatOllama(
            model=config.LLM_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=config.TEMPERATURE,
            num_predict=config.MAX_TOKENS,
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=config.SEPARATORS,
        )
        self.vectorstore = Chroma(
            collection_name=config.COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=config.CHROMA_PERSIST_DIR,
        )
        self.chain = self._build_chain()

    # ─── PDF Ingestion ─────────────────────────────────────────────

    def _extract_pdf_text(self, pdf_path: str) -> list[dict]:
        """Extract text from PDF with page-level metadata."""
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
        doc.close()
        return pages

    def _file_hash(self, file_bytes: bytes) -> str:
        """Generate hash to avoid re-ingesting duplicate files."""
        return hashlib.md5(file_bytes).hexdigest()

    def ingest_pdf(self, pdf_path: str, file_bytes: bytes | None = None) -> dict:
        """
        Parse, chunk, embed, and store a PDF.
        Returns stats about the ingestion.
        """
        # Check for duplicates via hash
        if file_bytes:
            file_hash = self._file_hash(file_bytes)
        else:
            with open(pdf_path, "rb") as f:
                file_hash = self._file_hash(f.read())

        # Check if already ingested
        existing = self.vectorstore.get(where={"file_hash": file_hash})
        if existing and existing["ids"]:
            return {
                "status": "skipped",
                "message": "File already ingested",
                "filename": os.path.basename(pdf_path),
                "chunks": len(existing["ids"]),
            }

        # Extract text
        pages = self._extract_pdf_text(pdf_path)
        if not pages:
            return {
                "status": "error",
                "message": "No text found in PDF (may be scanned/image-based)",
                "filename": os.path.basename(pdf_path),
            }

        # Create documents with metadata
        documents = []
        for page_data in pages:
            docs = self.text_splitter.create_documents(
                texts=[page_data["text"]],
                metadatas=[{
                    "source": page_data["source"],
                    "page": page_data["page"],
                    "file_hash": file_hash,
                }],
            )
            documents.extend(docs)

        # Embed and store
        self.vectorstore.add_documents(documents)

        return {
            "status": "success",
            "filename": os.path.basename(pdf_path),
            "pages": len(pages),
            "chunks": len(documents),
        }

    # ─── Retrieval ─────────────────────────────────────────────────

    def _format_context(self, docs: list[Document]) -> str:
        """Format retrieved documents into a context string with citations."""
        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "?")
            context_parts.append(
                f"[Source {i}: {source}, Page {page}]\n{doc.page_content}"
            )
        return "\n\n---\n\n".join(context_parts)

    def retrieve(self, query: str) -> list[Document]:
        """Retrieve relevant chunks for a query."""
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": config.TOP_K},
        )
        return retriever.invoke(query)

    # ─── Generation Chain ──────────────────────────────────────────

    def _build_chain(self):
        """Build the RAG chain: retrieve → format → generate."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", config.SYSTEM_PROMPT),
            ("human", "{question}"),
        ])

        chain = (
            {
                "context": lambda x: self._format_context(self.retrieve(x["question"])),
                "question": lambda x: x["question"],
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain

    def query(self, question: str) -> str:
        """Run a RAG query and return the response."""
        return self.chain.invoke({"question": question})

    def query_stream(self, question: str) -> Generator[str, None, None]:
        """Stream a RAG query response token by token."""
        context_docs = self.retrieve(question)
        context = self._format_context(context_docs)

        prompt = ChatPromptTemplate.from_messages([
            ("system", config.SYSTEM_PROMPT),
            ("human", "{question}"),
        ])

        chain = prompt | self.llm | StrOutputParser()

        for chunk in chain.stream({"context": context, "question": question}):
            yield chunk

    # ─── Collection Management ─────────────────────────────────────

    def get_ingested_files(self) -> list[str]:
        """Return list of unique filenames in the vector store."""
        results = self.vectorstore.get()
        if not results or not results["metadatas"]:
            return []
        sources = set()
        for meta in results["metadatas"]:
            if meta and "source" in meta:
                sources.add(meta["source"])
        return sorted(sources)

    def get_collection_stats(self) -> dict:
        """Return stats about the current collection."""
        results = self.vectorstore.get()
        total_chunks = len(results["ids"]) if results["ids"] else 0
        files = self.get_ingested_files()
        return {
            "total_chunks": total_chunks,
            "total_files": len(files),
            "files": files,
        }

    def clear_collection(self):
        """Delete all documents from the vector store."""
        self.vectorstore.delete_collection()
        # Recreate empty collection
        self.vectorstore = Chroma(
            collection_name=config.COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=config.CHROMA_PERSIST_DIR,
        )
        self.chain = self._build_chain()
