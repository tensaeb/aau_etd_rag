# AAU ETD Advanced RAG Pipeline 🚀

A highly modular, production-ready Retrieval-Augmented Generation (RAG) system designed for extracting knowledge from academic documents (AAU ETD) with high precision and reliability.

This system moves beyond "naive RAG" by implementing a layered architecture with advanced retrieval strategies, semantic reranking, and a robust component-based design.

---

## 🏛️ Advanced Architecture

The system follows a **SOLID-compliant, modular architecture** that decouples the retrieval logic from specific model implementations.

### 1. Hybrid Retrieval Engine (FAISS + BM25)
Retrieval is the most critical part of any RAG system. We use a **two-stream hybrid approach**:
*   **Semantic Search (FAISS)**: Uses `intfloat/e5-base-v2` embeddings to find conceptually related chunks. It understands synonyms and intent.
*   **Keyword Search (BM25)**: Uses the Okapi BM25 algorithm to find exact matches for technical terms, IDs, or specific Ethiopian academic jargon.
*   **Reciprocal Rank Fusion (RRF)**: Merges results from both streams into a single ranked list, ensuring we don't miss anything that only one method would find.

### 2. Cross-Encoder Reranking
Initial retrieval might return 20-50 candidates. We then pass these through a **Cross-Encoder model** (`cross-encoder/ms-marco-MiniLM-L-6-v2`).
*   Unlike bi-encoders, the cross-encoder processes the query and the document chunk *simultaneously*, allowing it to understand deep contextual relevance.
*   It re-scores the candidates, and only the **top-K** most relevant chunks are sent to the LLM.

### 3. Context Builder & Semantic Truncation
*   **Layout-Aware Extraction**: Uses `unstructured` to parse PDFs, handling multi-column text and tables correctly.
*   **Recursive Chunking**: Splits text into 1,500-character chunks with overlap to maintain context.
*   **Smart Truncation**: When building the prompt, we use **NLTK sentence tokenization** to truncate text at sentence boundaries, preventing the LLM from seeing cut-off words that might lead to hallucinations.

### 4. Modular Model Layer
Everything is swappable. By implementing abstract interfaces, you can switch from LM Studio to OpenAI or Anthropic by changing just one file, without touching the core RAG logic.

---

## 🛠️ Getting Started

### 1. Installation
We recommend using the included virtual environment setup:

```powershell
# Create environment
python -m venv venv
# Activate (Windows)
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration
All parameters (model names, chunk sizes, API ports, file paths) are centralized in `config.yaml`.
**Crucial**: If your LLM is running on a different port or machine, update the `lm_studio` section.

### 3. The Data Pipeline
Place your PDFs in `data/raw_pdfs/` and run the end-to-end pipeline:

```powershell
python -m src.run_pipeline
```
This script handles:
1.  `extract_text`: Layout-aware PDF parsing.
2.  `chunk_text`: Semantic splitting.
3.  `build_bm25`: Keyword index generation.
4.  `embed_chunks`: Vector generation.
5.  `build_faiss`: Vector index construction.

### 4. Running the API
Start the high-performance FastAPI server:

```powershell
python -m src.api
```

---

## 📂 Project Structure

```text
.
├── config.yaml               # ⚙️ Centralized system configuration
├── logs/                     # 📝 Structured execution logs
├── data/
│   ├── raw_pdfs/             # 📂 Source PDF storage
│   ├── extracted_text/       # 📄 Clean text files
│   ├── chunks/               # 🧩 Processed text chunks
│   └── embeddings/           # 🔢 Vector & metadata storage
├── indexes/                  # 🔍 FAISS & BM25 index files
└── src/
    ├── core/                 # 🛠️ Configuration & Logger utilities
    ├── models/               # 🧠 Abstract interfaces (Embedder, LLM, Reranker)
    ├── retrieval/            # 🔍 Vector Store & Hybrid Search logic
    ├── rag/                  # 🤖 Orchestration & Context Building
    ├── pipeline/             # ⚙️ Individual ETL scripts
    └── api.py                # 🌐 FastAPI implementation
```

---

## ⚠️ Important Notes for LM Studio
If you experience "Channel Errors" or crashes:
1.  **Switch to CPU only**: Set "GPU Offload" to 0 in LM Studio.
2.  **Smaller Models**: Use `Phi-3-mini` or `Qwen2-1.5B` for better stability on integrated graphics.
3.  **Context Size**: We have pre-set the `max_context_chars` to `2000` to prevent crashes on limited hardware.
