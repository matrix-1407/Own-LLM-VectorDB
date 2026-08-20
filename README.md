# VectorDB — Complete Vector Database & Advanced RAG Studio Built from Scratch

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-Local_AI-black?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-6c63ff?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A fully working, educational, and production-grade vector database engine & multi-stage RAG studio implemented from scratch in Python with an interactive 3D visualizer, HNSW graph inspector, search trajectory tracer, disk persistence, SQ8 quantization, BM25 hybrid search, semantic chunking, cross-encoder re-ranking, and sentence grounding — 100% local, zero cloud dependencies.**

[Features](#features) • [How Global VectorDBs & RAG Work](#how-global-vectordbs--rag-work) • [3D & Graph Introspection](#3d--graph-introspection-phase-5) • [Advanced RAG Deep Dive](#advanced-rag-deep-dive-phase-4) • [Setup](#setup) • [API Reference](#api-reference)

</div>

---

## What Is This?

VectorDB is a complete **educational and production-grade vector database & RAG studio** built from the ground up — no FAISS, no Chroma, no LangChain, no Pinecone. Every algorithm is implemented in pure, readable Python so you can see exactly how modern AI search and retrieval-augmented generation systems work on a global scale.

It combines:
- **3 core search algorithms** running side-by-side with live speed comparison (HNSW, KD-Tree, Brute Force)
- **Local AI embeddings & generation** using Ollama (`nomic-embed-text` and `llama3.2`)
- **🌐 Interactive 3D Vector Space** with orbital camera controls (drag/zoom/pan), cluster centroids, and category glow
- **🪜 Stacked HNSW Multi-Layer Graph Inspector** visualizing hierarchical highway layers ($L_0, L_1, L_2$)
- **🎬 Search Trajectory Path Tracer** animating greedy graph hops across layers step-by-step
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
| **🌐 Interactive 3D Visualizer** | Orbital 3D point cloud with perspective projection, cluster centroids, and category glow |
| **🪜 Stacked HNSW Inspector** | Hierarchical multi-layer graph view with intra-layer and inter-layer connection guides |
| **🎬 Search Trajectory Tracer** | Step-by-step recording and animated playback of greedy graph traversal |
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

## How Global VectorDBs & RAG Work

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

## 3D & Graph Introspection (Phase 5)

### 1. 🌐 Interactive 3D Vector Space
- High-dimensional embeddings are reduced to 3D via **3D Principal Component Analysis (PCA)** (`vectordb/manifold.py`).
- **Orbital Camera**: Left-click and drag to rotate, mouse wheel to zoom, with depth-sorted billboarding.
- **Cluster Centroids & Radii**: Calculates 3D centers of mass and cluster variances for all semantic categories.

### 2. 🪜 Stacked HNSW Multi-Layer Graph Inspector
- Renders the hierarchical layers of the HNSW index as isometric stacked planes ($L_0, L_1, L_2$).
- Upper layers act as expressways; lower layers contain fine-grained nearest neighbors.
- Displays inter-layer vertical guides and intra-layer graph edges.

### 3. 🎬 Search Trajectory Path Tracer & Player
- Traces every greedy hop taken by HNSW during a search:
  1. Entry point at top layer $L_{\text{top}}$.
  2. Greedy hops across neighbors to minimize query distance.
  3. Layer-drop transitions down to ground layer $L_0$.
  4. Beam search candidate explorations.
- Interactive playback controls: **Play**, **Pause**, **Step Forward/Backward**, and **Seek Slider**.

### 4. 📐 Metric Space & Distance Iso-Contour Visualizer
- Compares neighbor rankings across **Cosine Similarity**, **Euclidean Distance ($L_2$)**, and **Manhattan Distance ($L_1$)** for the same query point.

---

## Advanced RAG Deep Dive (Phase 4)

### 1. 🧠 Semantic Chunking (`vectordb/semantic_chunker.py`)
- Slices documents at true topic transitions instead of arbitrary word counts.
- Computes sentence embedding distance curves ($d_i = 1 - \text{sim}(s_i, s_{i+1})$) and dynamic split thresholds.

### 2. 🎯 Two-Stage Retrieval & Re-ranking (`vectordb/reranker.py`)
- **Stage 1**: Fast candidate oversampling (top 10–15).
- **Stage 2**: `CrossScoreReranker` (token alignment & phrasing) and `LLMReranker` (pointwise LLM evaluation).

### 3. 🛡️ Sentence-Level Grounding & Citations (`vectordb/grounding.py`)
- Verifies every sentence of the LLM answer against retrieved chunks.
- Computes **Factuality Confidence Score** and attaches clickable citation badges (`[#1]`, `[#2]`).

### 4. 🔮 HyDE: Hypothetical Document Embeddings (`vectordb/hyde.py`)
- Generates a hypothetical answer via `llama3.2` and embeds it to bridge query-document format asymmetry.

---

## Setup

### Prerequisites
1. **Python 3.10+**
2. **[Ollama](https://ollama.com)** (free, local LLM runner)

### Step 1 — Clone the Repository
```bash
git clone https://github.com/matrix-1407/Own-LLM-VectorDB.git
cd Own-LLM-VectorDB
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

## Architecture

```
VectorDB/
├── app.py                      # FastAPI server — REST endpoints & lifespan
├── index.html                  # Single-Page Web UI — 3D Canvas, HNSW inspector, RAG studio
├── requirements.txt
├── data/                       # On-disk JSON snapshots (auto-created)
│   ├── vectordb_index.json     # Demo vector snapshot
│   └── document_index.json     # Document chunk snapshot
└── vectordb/                   # Core vector database & RAG engine
    ├── metrics.py              # Euclidean, Cosine, Manhattan distance metrics
    ├── chunker.py              # Fixed-size word chunker (250w / 30 overlap)
    ├── semantic_chunker.py     # Embedding-driven topic boundary chunker
    ├── reranker.py             # CrossScoreReranker & LLMReranker
    ├── grounding.py            # Sentence-level grounding & citation engine
    ├── hyde.py                 # Hypothetical Document Embeddings
    ├── manifold.py             # 3D PCA, cluster centroids & metric comparison
    ├── ollama_client.py        # Local Ollama HTTP wrapper
    ├── vector_db.py            # VectorDB manager — 16D demo index & 3D methods
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

### Demo Vector & Introspection Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/search?v=f1,f2,...&k=5&metric=cosine&algo=hnsw` | K-NN search |
| `GET` | `/items/3d` | Get all items with 3D PCA coordinates |
| `GET` | `/hnsw/trace?v=...&k=5&metric=cosine` | Step-by-step greedy search trajectory |
| `GET` | `/hnsw/topology` | Full multi-layer HNSW graph topology |
| `GET` | `/analytics/clusters` | 3D cluster centroids, radii, and variances |
| `GET` | `/analytics/metric-compare?v=...&k=5` | Cosine vs Euclidean vs Manhattan comparison |
| `GET` | `/benchmark?v=...&k=5&metric=cosine` | Compare all 3 algorithms |
| `GET` | `/stats` | Database statistics (SQ8, BM25, Phase 4/5 info) |

### Advanced RAG & Document Endpoints

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

## Roadmap

- [x] **Phase 1**: Architecture design & technical specifications
- [x] **Phase 2**: Core engine (HNSW, KD-Tree, Brute Force, Ollama RAG)
- [x] **Phase 3**: Engine upgrades (Disk persistence, SQ8 quantization, BM25 hybrid search, metadata filtering)
- [x] **Phase 4**: Advanced RAG (Semantic chunking, Two-stage re-ranking, Sentence grounding & citations, HyDE)
- [x] **Phase 5**: Visualization & Introspection (3D vector visualizer, stacked HNSW graph inspector, search trajectory tracer, metric space geometry)
- [ ] **Future Scope**: Multi-tenant collections & Docker deployment (to be explored in future milestones)

---

## License

MIT — use this however you want.

---

<div align="center">

Built with ❤️ to learn how vector databases and RAG architectures actually work.

</div>
