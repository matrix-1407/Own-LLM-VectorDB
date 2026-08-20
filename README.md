# VectorDB — Vector Database & Advanced RAG Engine Built from Scratch in Python

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-Local_AI-black?style=for-the-badge)
![Phase](https://img.shields.io/badge/Phase-4_Complete-6c63ff?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A complete, educational, and production-grade vector database & multi-stage RAG engine built from scratch in Python with a live web UI, 3 search algorithms, disk persistence, SQ8 quantization, BM25 hybrid search, semantic chunking, cross-encoder re-ranking, sentence-level grounding, and HyDE — 100% local, zero cloud dependencies.**

[Features](#features) • [How Global RAG & VectorDBs Work](#how-global-rag--vectordbs-work) • [Phase 4 Deep Dive](#phase-4-deep-dive--advanced-rag) • [Setup](#setup) • [API Reference](#api-reference) • [Architecture](#architecture)

</div>

---

## What Is This?

VectorDB is an **educational and production-grade vector database and multi-stage RAG pipeline** built from the ground up — no FAISS, no Chroma, no LangChain, no Pinecone. Every algorithm is implemented in clean, readable Python so you can see exactly how modern AI search and retrieval-augmented generation systems work on a global scale.

It combines:
- **3 core search algorithms** running side-by-side with live speed comparison (HNSW, KD-Tree, Brute Force)
- **Local AI embeddings & generation** using Ollama (`nomic-embed-text` and `llama3.2`)
- **2D PCA scatter visualizer** showing real-time clustering in semantic space
- **💾 Disk Persistence** with atomic JSON snapshots
- **⚡ SQ8 Scalar Quantization** for 4× memory compression (>98% recall)
- **🔍 Hybrid Search** (BM25 keyword + HNSW vector via Reciprocal Rank Fusion)
- **🧠 Semantic Chunking** using embedding distance curves for topic boundary detection
- **🎯 Two-Stage Re-ranking** (Cross-Attention Scorer & Pointwise LLM Evaluator)
- **🛡️ Sentence-Level Grounding & Interactive Citations** (hallucination detection and source passage mapping)
- **🔮 HyDE (Hypothetical Document Embeddings)** for query expansion

---

## Features

| Feature | Details |
|---|---|
| **3 Search Algorithms** | HNSW (production-grade graph), KD-Tree (spatial partitioning), Brute Force (exact scan) |
| **3 Distance Metrics** | Cosine similarity, Euclidean distance, Manhattan distance |
| **16D Demo Vectors** | 20 pre-loaded semantic vectors across 4 categories (CS, Math, Food, Sports) |
| **2D PCA Scatter Plot** | Live visualization of semantic space — watch clusters form |
| **Real Document Embedding** | Paste any text → Ollama embeds it with `nomic-embed-text` (768D) |
| **💾 Disk Persistence** | Auto-save/load on shutdown/startup — documents survive server restarts |
| **⚡ SQ8 Compression** | Float32 → int8 quantization for 4× memory reduction with >98% recall |
| **🔍 Hybrid Search** | BM25 keyword + HNSW vector merged via Reciprocal Rank Fusion (RRF) |
| **🏷️ Metadata Filtering** | Category-filtered vector search (`?category=cs`) |
| **🧠 Semantic Chunking** | Dynamic topic boundary detection using sentence embedding distance spikes |
| **🎯 2-Stage Re-ranking** | Fast candidate retrieval + Cross-Attention / LLM pointwise precision scoring |
| **🛡️ Sentence Grounding** | Sentence-by-sentence verification against context + interactive citation links `[#1]` |
| **🔮 HyDE Query Expansion** | LLM generates hypothetical passage → embeds it to bridge query-document asymmetry |
| **Zero Cloud Cost** | 100% local. No API keys. No monthly bills. |

---

## How Global RAG & VectorDBs Work

```
User Query
    │
    ├─── [Optional] HyDE (Hypothetical Document Embedding)
    │    └── LLM writes hypothetical answer -> embed hypothetical text
    │
    ▼
Stage 1: Fast Candidate Retrieval (Bi-Encoder / Hybrid)
    ├── HNSW Vector Search (Semantic) ───────┐
    └── BM25 Inverted Index (Lexical) ───────┴──> Reciprocal Rank Fusion (Top 10-15)
                                                        │
Stage 2: Precision Re-Ranking (Cross-Encoder / LLM)     ▼
    ├── Cross-Attention Scorer (Token Interaction & Alignment)
    └── LLM Pointwise Evaluator (Score 0-10 + Reasoning) ───> Top K (e.g. Top 3)
                                                                    │
Stage 3: Grounded Context Assembly & LLM Generation                 ▼
    └── Llama 3.2 generates response with strict factual grounding
                                                                    │
Stage 4: Post-Generation Grounding & Citation Extraction            ▼
    ├── Sentence-by-sentence semantic verification against retrieved chunks
    ├── Factuality confidence meter (0-100%) & Hallucination detection
    └── Interactive citation badges [#1], [#2] linked to exact source passages
```

---

## Phase 4 Deep Dive — Advanced RAG

### 1. 🧠 Semantic Chunking vs. Fixed Word Chunking
- **Traditional Fixed Chunking** cuts text strictly at 250 words. This frequently splits sentences or combines unrelated topics (e.g. a Python paragraph followed by a Pizza recipe).
- **Semantic Chunking** (`vectordb/semantic_chunker.py`):
  1. Splits document into natural sentences.
  2. Embeds each sentence with `nomic-embed-text`.
  3. Computes the cosine distance curve between consecutive sentences ($d_i = 1 - \text{sim}(s_i, s_{i+1})$).
  4. Calculates a dynamic threshold (e.g. 75th percentile of distances).
  5. Splits only when a distance spike exceeds the threshold, preserving conceptual unity in every chunk.

### 2. 🎯 Two-Stage Retrieval & Re-ranking
- **The Problem**: Bi-encoders (HNSW) encode queries and documents separately to enable $O(\log N)$ search, but they lose token-level cross interactions.
- **The Solution** (`vectordb/reranker.py`):
  - **Stage 1**: Fast candidate retrieval retrieves the top 10–15 candidate chunks.
  - **Stage 2**: `CrossScoreReranker` evaluates query-document token alignment, exact n-gram matching, and term density, or `LLMReranker` uses `llama3.2` to grade relevance from 0 to 10 with written justification.

### 3. 🛡️ Sentence-Level Grounding & Citations
- **The Problem**: LLMs can hallucinate or blend outside training data into technical answers.
- **The Solution** (`vectordb/grounding.py`):
  - Decomposes the generated response into individual sentences.
  - Measures cosine similarity between each generated sentence and all sentences in the retrieved context.
  - Classifies each claim as **Grounded** ($\ge 0.68$), **Partially Grounded** ($0.50 - 0.67$), or **Ungrounded** ($< 0.50$).
  - Calculates an overall **Factuality Confidence Meter** and attaches interactive citation badges (`[#1]`, `[#2]`).
  - Clicking any sentence or citation badge in the Web UI highlights the exact source document and sentence.

### 4. 🔮 HyDE (Hypothetical Document Embeddings)
- **The Problem**: A user asks a short question ("Why is KD-tree slow in 100D?"), but the database stores long technical paragraphs. The question vector and document vector reside in different semantic spaces.
- **The Solution** (`vectordb/hyde.py`):
  - Prompts `llama3.2` to generate a hypothetical technical passage answering the question.
  - Embeds the hypothetical passage with `nomic-embed-text`.
  - Searches HNSW using the hypothetical passage vector, bridging the query-document asymmetry.

---

## Setup

### Prerequisites
1. **Python 3.10+**
2. **[Ollama](https://ollama.com)** (free, local LLM runner)

### Step 1 — Clone the Repository
```bash
git clone https://github.com/matrix-1407/VectorDB.git
cd VectorDB
```

### Step 2 — Create Virtual Environment & Install Dependencies
```bash
python -m venv venv

# Windows
.\venv\Scripts\Activate.ps1

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### Step 3 — Pull Ollama Models
```bash
ollama pull nomic-embed-text   # ~274 MB — embedding model
ollama pull llama3.2           # ~2 GB  — language model
```

### Step 4 — Run the Server
```bash
python app.py
```

Open your browser at **[http://localhost:8080](http://localhost:8080)**

---

## Using the Web UI

### Tab 1: Search (Demo 16D Vectors)
- Search concepts: `binary tree`, `sushi`, `basketball`, `calculus`
- Select algorithm: **HNSW**, **KD-Tree**, or **Brute Force**
- Select distance metric: **Cosine**, **Euclidean**, or **Manhattan**
- Filter by category with the **Category Filter** dropdown
- Benchmark all 3 algorithms head-to-head with **▶ COMPARE ALL ALGOS**
- Persist indexes with **💾 SAVE TO DISK**

### Tab 2: Documents (Semantic Chunking & Embedding)
1. Select chunking strategy: **📏 FIXED (250w)** or **🧠 SEMANTIC (Embed)**
2. Click **👁️ PREVIEW SPLITS** to visualize the sentence distance curve and topic split boundaries
3. Enter title, paste text, and click **⚡ EMBED & INSERT**

### Tab 3: Ask AI (Advanced RAG Studio)
1. Choose retrieval pipeline:
   - `🔬 VECTOR` (Pure HNSW)
   - `⚡ HYBRID` (BM25 + HNSW via RRF)
   - `🎯 RERANK` (2-Stage Candidate Oversampling + Cross-Attention Re-scoring)
   - `🔮 HYDE` (Hypothetical Document Embeddings)
2. Type any question about your documents and click **🤖 ASK AI (RAG)**
3. View the **Factuality & Grounding Meter** (e.g. `95% Grounded`)
4. Hover or click any underlined sentence or citation `[#1]` to highlight the exact source chunk and sentence below

### Tab 4: System
- View real-time index metrics (vector count, doc chunks, BM25 token count, SQ8 compression ratio)
- Inspect active Phase 4 architecture engines
- Check disk snapshot status and trigger manual persistence

---

## Architecture

```
VectorDB/
├── app.py                      # FastAPI server — REST endpoints & lifespan
├── index.html                  # Single-Page Web UI — PCA scatter, Grounding inspector, RAG studio
├── requirements.txt
├── data/                       # On-disk JSON snapshots (auto-created)
│   ├── vectordb_index.json     # Demo vector snapshot
│   └── document_index.json     # Document chunk snapshot
└── vectordb/                   # Core vector database & RAG engine
    ├── metrics.py              # Euclidean, Cosine, Manhattan distance metrics
    ├── chunker.py              # Fixed-size word chunker (250w / 30 overlap)
    ├── semantic_chunker.py     # Phase 4: Embedding-driven topic boundary chunker
    ├── reranker.py             # Phase 4: CrossScoreReranker & LLMReranker
    ├── grounding.py            # Phase 4: Sentence-level grounding & citation engine
    ├── hyde.py                 # Phase 4: Hypothetical Document Embeddings
    ├── ollama_client.py        # Local Ollama HTTP wrapper
    ├── vector_db.py            # VectorDB manager — 16D demo index & filtering
    ├── document_db.py          # DocumentDB manager — 768D multi-pipeline RAG index
    ├── demo_data.py            # 20 pre-loaded categorical demo vectors
    ├── persistence.py          # Atomic JSON snapshot save/load
    ├── quantization.py         # SQ8 int8 compression & flat index
    ├── bm25.py                 # BM25 inverted index & keyword scoring
    └── algorithms/
        ├── brute_force.py      # Exact scan  O(N·d)
        ├── kd_tree.py          # KD-Tree     O(log N) — low dimensions
        └── hnsw.py             # HNSW graph  O(log N) — high dimensions ✓
```

---

## API Reference

### Demo Vector Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/search?v=f1,f2,...&k=5&metric=cosine&algo=hnsw` | K-NN search |
| `GET` | `/search?v=...&category=cs` | Filtered K-NN search |
| `POST` | `/insert` | Insert a demo vector |
| `DELETE` | `/delete/{id}` | Delete by ID |
| `GET` | `/items` | List all demo vectors |
| `GET` | `/categories` | List unique category labels |
| `GET` | `/benchmark?v=...&k=5&metric=cosine` | Compare all 3 algorithms |
| `GET` | `/hnsw-info` | HNSW graph structure and layer stats |
| `GET` | `/stats` | Database statistics (incl. SQ8, BM25, Phase 4 info) |

### Advanced RAG & Document Endpoints (Phase 4)

| Method | Endpoint | Body | Description |
|---|---|---|---|
| `POST` | `/doc/insert` | `{"title":"...","text":"...","chunking_strategy":"semantic"}` | Insert with Fixed or Semantic chunking |
| `POST` | `/doc/semantic-chunk-preview` | `{"text":"...","threshold_percentile":75.0}` | Preview distance curves & topic split boundaries |
| `POST` | `/doc/advanced-ask` | `{"question":"...","k":3,"pipeline":"rerank","grounding":true}` | Unified multi-pipeline RAG with grounding |
| `POST` | `/doc/rerank` | `{"query":"...","k":5,"strategy":"cross"}` | Test 2-stage candidate re-ranking |
| `POST` | `/doc/hybrid-search` | `{"question":"...","k":3,"rrf_k":60}` | BM25 + HNSW hybrid search |
| `GET` | `/doc/list` | — | List all stored document chunks |
| `DELETE` | `/doc/delete/{id}` | — | Delete document chunk |
| `GET` | `/status` | — | Ollama status and model info |

### Persistence Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/persist/save` | Manually save both indexes to disk |
| `GET` | `/persist/status` | View snapshot file paths and sizes |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Ollama: OFFLINE` | Run `ollama serve` in a separate terminal |
| Embedding takes forever | Ollama is downloading the model on first use — wait ~2 min |
| Port 8080 in use | Kill process on 8080 or press `Ctrl + C` in previous terminal |
| LLM answer is slow | Normal — llama3.2 takes 5–20s on CPU. Use `llama3.2:1b` for faster responses |
| Semantic chunk preview fails | Ensure Ollama is running (`ollama serve`) with `nomic-embed-text` pulled |

---

## Roadmap

- [x] **Phase 1**: Architecture design & technical specifications
- [x] **Phase 2**: Core engine (HNSW, KD-Tree, Brute Force, Ollama RAG)
- [x] **Phase 3**: Engine upgrades (Disk persistence, SQ8 quantization, BM25 hybrid search, metadata filtering)
- [x] **Phase 4**: Advanced RAG (Semantic chunking, Two-stage re-ranking, Sentence grounding & citations, HyDE)
- [ ] **Phase 5**: Visualization & Introspection (3D vector visualizer, interactive HNSW graph inspector)
- [ ] **Phase 6**: Multi-Tenant & Production Deployment (Multi-collection, client SDK, Docker)

---

## License

MIT — use this however you want.

---

<div align="center">

Built with ❤️ to learn how vector databases and RAG architectures actually work.

</div>
