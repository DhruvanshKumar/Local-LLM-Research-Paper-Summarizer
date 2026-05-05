# Local Research Paper Summarizer
A fully local, zero-cost RAG (Retrieval-Augmented Generation) chatbot that lets you upload research papers and ask questions about them in natural language.
##  Tech Stack
| Component | Technology |
|---|---|
| LLM | LLaMA 3 via Ollama |
| Orchestration | LangChain |
| Vector Database | FAISS |
| Embeddings | Sentence Transformers |
| PDF Loading | PyPDFLoader |
| UI | Streamlit |
| Python | 3.12 |

---

## Architecture

```
PDF files
    ↓
PyPDFLoader → pages (Document objects)
    ↓
RecursiveCharacterTextSplitter → chunks (1000 chars, 200 overlap)
    ↓
Sentence Transformers → embeddings (vectors)
    ↓
FAISS vectorstore → stored on disk
    ↓
User asks question
    ↓
Question embedded → FAISS similarity search → top 5 chunks retrieved
    ↓
chunks + chat history + question → LLaMA 3 (via Ollama)
    ↓
Answer + Sources displayed in chat UI
```

---

##  Project Structure

```
llm/
├── app.py                  # Streamlit UI — file upload, chat interface
├── ingestion.py            # PDF loading and chunking
├── vectorstore.py          # FAISS vectorstore creation and persistence
├── rag_pipeline.py         # RAG chain and question answering
└── requirements.txt        # Python dependencies
```

---

##  Prerequisites

- [Ollama](https://ollama.com) installed on your machine
- Python 3.12
- pyenv (recommended for managing Python versions)

---

## 🚀Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/DhruvanshKumar/Local-LLM-Research-Paper-Summarizer.git
cd llm
```

### 2. Install Python 3.12 via pyenv

```bash
pyenv install 3.12
pyenv local 3.12
```

### 3. Create and activate a virtual environment

```bash
python3.12 -m venv venv
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install requirements.txt
```

## Running the App

You need **three terminals** running simultaneously:

**Terminal 1 — Start Ollama:**
```bash
ollama serve
```
**Terminal 2 - Pull Ollama:**
```bash
ollama pull
```
**Terminal 3 — Start the app:**
```bash
cd llm
source venv/bin/activate
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

---

##  How to Use

1. Upload one or more PDF files using the **sidebar**
2. Click **"Ingest PDFs"** and wait for processing
3. Once ingested, type your question in the chat box
4. The app will retrieve relevant sections and generate an answer
5. Expand **"Sources"** below each answer to see which PDFs were referenced
6. Use **"Clear chat history"** to start a fresh conversation

---

## 🔧 Configuration

You can tweak these settings in the source files:

| Setting | File | Default | Description |
|---|---|---|---|
| `OLLAMA_MODEL` | `rag_pipeline.py` | `"llama3"` | Which Ollama model to use |
| `temperature` | `rag_pipeline.py` | `0` | LLM creativity (0 = factual) |
| `chunk_size` | `ingestion.py` | `1000` | Characters per chunk |
| `chunk_overlap` | `ingestion.py` | `200` | Overlap between chunks |
| `k` | `rag_pipeline.py` | `5` | Number of chunks retrieved per query |

---

## 📦 Requirements

```
langchain==0.3.25
langchain-community==0.3.23
langchain-core==0.3.58
langchain-ollama==0.3.2
langchain-text-splitters
streamlit
faiss-cpu
sentence-transformers
pypdf
```

---

##  Common Issues

**`Connection refused` error**
> Ollama isn't running. Start it with `ollama serve` in a separate terminal.

**`ModuleNotFoundError`**
> Your venv isn't activated. Run `source venv/bin/activate` first.

**Squiggles in VS Code**
> Select the correct interpreter: `Cmd+Shift+P` → "Python: Select Interpreter" → choose `./venv/bin/python`

**MRO / Python version errors**
> This project requires Python 3.12. Do not use Python 3.14+.

---
