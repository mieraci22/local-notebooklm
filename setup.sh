#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# Local NotebookLM — Setup Script
# For Mac Mini M4 (16GB) running macOS Tahoe
# ─────────────────────────────────────────────────────────────────

set -e

echo "📚 Local NotebookLM Setup"
echo "========================="
echo ""

# ─── Check for Homebrew ────────────────────────────────────────
if ! command -v brew &> /dev/null; then
    echo "📦 Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# ─── Install Ollama ────────────────────────────────────────────
if ! command -v ollama &> /dev/null; then
    echo "🤖 Installing Ollama..."
    brew install ollama
else
    echo "✅ Ollama already installed"
fi

# ─── Start Ollama ──────────────────────────────────────────────
echo "🚀 Starting Ollama service..."
ollama serve &> /dev/null &
sleep 3

# ─── Pull Models ──────────────────────────────────────────────
echo ""
echo "📥 Pulling Llama 3.1 8B (~4.7GB)..."
ollama pull llama3.1:8b

echo ""
echo "📥 Pulling nomic-embed-text (~274MB)..."
ollama pull nomic-embed-text

# ─── Verify Models ────────────────────────────────────────────
echo ""
echo "🔍 Verifying models..."
ollama list

# ─── Python Environment ───────────────────────────────────────
echo ""
echo "🐍 Setting up Python environment..."

if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Install via: brew install python3"
    exit 1
fi

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "📦 Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

# ─── Quick Test ────────────────────────────────────────────────
echo ""
echo "🧪 Running quick test..."
python3 -c "
from langchain_ollama import ChatOllama, OllamaEmbeddings
import chromadb

# Test LLM
llm = ChatOllama(model='llama3.1:8b')
response = llm.invoke('Say hello in exactly 5 words.')
print(f'  LLM test: {response.content}')

# Test embeddings
embed = OllamaEmbeddings(model='nomic-embed-text')
vec = embed.embed_query('test')
print(f'  Embedding test: vector dim = {len(vec)}')

# Test ChromaDB
client = chromadb.Client()
print(f'  ChromaDB test: OK')

print('')
print('✅ All systems operational!')
"

# ─── Done ──────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════"
echo "  🎉 Setup complete!"
echo ""
echo "  To start the app:"
echo "    source venv/bin/activate"
echo "    streamlit run app.py"
echo ""
echo "  Then open: http://localhost:8501"
echo "════════════════════════════════════════════"
