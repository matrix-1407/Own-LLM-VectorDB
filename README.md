# VectorDB — Vector Database Built from Scratch in Python

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-Local_AI-black?style=for-the-badge)
![Phase](https://img.shields.io/badge/Phase-3_Complete-6c63ff?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A fully working vector database engine implemented from scratch in Python with a live web UI, RAG pipeline, three search algorithms, disk persistence, SQ8 compression, BM25 hybrid search, and metadata filtering — no cloud dependencies, runs 100% locally.**

[Features](#features) • [How It Works](#how-it-works) • [Setup](#setup) • [API Reference](#api-reference) • [Architecture](#architecture)

</div>

---

## What Is This?

VectorDB is an **educational and production-grade vector database** built from the ground up — no FAISS, no Chroma, no Pinecone. Every algorithm is implemented in pure Python so you can see exactly how modern AI-powered search actually works under the hood.

It combines:
- **3 search algorithms** running side-by-side with live speed comparison
- **Real AI embeddings** from a locally running Ollama model
- **A RAG pipeline** that lets you upload documents and ask questions about them
- **A live 2D scatter plot** showing your vectors clustering by semantic meaning
- **Phase 3**: Disk persistence, SQ8 compression, BM25 hybrid search, and metadata filtering

---

## Features

| Feature | Details |
|---|---|
| **3 Search Algorithms** | HNSW (production-grade), KD-Tree, Brute Force — run all three and compare speed |
| **3 Distance Metrics** | Cosine similarity, Euclidean distance, Manhattan distance |
| **16D Demo Vectors** | 20 pre-loaded semantic vectors across 4 categories (CS, Math, Food, Sports) |
| **2D PCA Scatter Plot** | Live visualization of semantic space — watch clusters form |
| **Real Document Embedding** | Paste any text → Ollama embeds it with `nomic-embed-text` (768D) |
| **RAG Pipeline** | Upload documents → HNSW retrieves context → local LLM answers your questions |
| **Full REST API** | CRUD endpoints: insert, delete, search, benchmark, hnsw-info |
| **💾 Disk Persistence** | Auto-save/load on shutdown/startup — documents survive server restarts |
| **⚡ SQ8 Compression** | Float32 → int8 quantization for 4× memory reduction with >98% recall |
| **🔍 Hybrid Search** | BM25 keyword + HNSW vector merged via Reciprocal Rank Fusion (RRF) |
| **🏷️ Metadata Filtering** | Category-filtered vector search (`?category=cs`) |
| **Zero Cloud Cost** | 100% local. No API keys. No monthly bills. |

---

## How It Works

```
Your Text
    │
    ▼
Ollama (nomic-embed-text)      ← converts text to a 768-dimensional vector
    │
    ▼
HNSW Index (Python)            ← indexes the vector in a multilayer graph
    │
    ├─── Vector Search (cosine / euclidean / manhattan)
    │
    ├─── Hybrid Search (BM25 + HNSW → Reciprocal Rank Fusion)
    │
    ▼
Ollama (llama3.2)              ← reads retrieved chunks, generates an answer
    │
    ▼
Answer
```

**HNSW (Hierarchical Navigable Small World)** is the same algorithm powering Pinecone, Weaviate, Chroma, and Milvus. It builds a multi-layer graph where upper layers act as a highway to quickly reach the right neighborhood — achieving O(log N) search instead of O(N).

**BM25 (Best Match 25)** is the classic keyword retrieval algorithm used by Elasticsearch and Solr. Combined with HNSW via **Reciprocal Rank Fusion**, it handles exact keyword matches that pure vector search might miss.

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

### Tab 1: Search (Demo Vectors)
- Type any concept: `binary tree`, `sushi`, `basketball`, `calculus`
- Choose algorithm: **HNSW**, **KD-Tree**, or **Brute Force**
- Choose distance metric: **Cosine**, **Euclidean**, or **Manhattan**
- **Phase 3**: Filter by category using the **Category Filter** dropdown
- Click **⚡ SEARCH** — results appear, matching point glows on scatter plot
- Click **▶ COMPARE ALL ALGOS** to benchmark all 3 algorithms head-to-head
- Click **💾 SAVE TO DISK** to persist your indexes

### Tab 2: Documents (Real Embeddings)
1. Enter a title and paste any text (lecture notes, articles, docs)
2. Click **⚡ EMBED & INSERT**
3. Long documents auto-split into 250-word overlapping chunks

### Tab 3: Ask AI (RAG Pipeline)
1. Insert documents first via Tab 2
2. Choose mode: **🔬 VECTOR ONLY** (pure HNSW) or **⚡ HYBRID BM25+HNSW** (Phase 3)
3. Type any question about your documents
4. Click **🤖 ASK AI** — watch the typewriter answer stream in

### Tab 4: System (Phase 3)
- View real-time index statistics (vector count, doc chunks, BM25 token count)
- Monitor **SQ8 compression** ratios and memory savings
- View **disk persistence** snapshot status (file path, size)
- Manually save indexes to disk

---

## Architecture

```
VectorDB/
├── app.py                      # FastAPI server — all REST endpoints
├── index.html                  # Web UI — PCA scatter, benchmark, RAG chat
├── requirements.txt
├── data/                       # Phase 3: auto-created on first save
│   ├── vectordb_index.json     # Demo vector snapshot
│   └── document_index.json     # Document chunk snapshot
└── vectordb/                   # Core vector database engine
    ├── metrics.py              # Euclidean, Cosine, Manhattan distance functions
    ├── chunker.py              # Text word chunker (250 words / 30 overlap)
    ├── ollama_client.py        # Local Ollama HTTP wrapper
    ├── vector_db.py            # VectorDB manager — 16D demo index
    ├── document_db.py          # DocumentDB manager — 768D RAG index + hybrid search
    ├── demo_data.py            # 20 pre-loaded categorical vectors
    ├── persistence.py          # Phase 3: JSON snapshot save/load
    ├── quantization.py         # Phase 3: SQ8 int8 compression + SQ8Index
    ├── bm25.py                 # Phase 3: BM25 inverted index + keyword scoring
    └── algorithms/
        ├── brute_force.py      # Exact scan  O(N·d)
        ├── kd_tree.py          # KD-Tree     O(log N) — low dimensions
        └── hnsw.py             # HNSW graph  O(log N) — high dimensions ✓
```

### Algorithm Complexity

| Algorithm | Complexity | Best For |
|---|---|---|
| **Brute Force** | O(N·d) | Exact baseline, small datasets |
| **KD-Tree** | O(log N) | Low dimensions (≤ 20D) |
| **HNSW** | O(log N) | High dimensions, production use |
| **BM25** | O(|q|·avg_df) | Keyword / lexical matching |
| **Hybrid (BM25+HNSW+RRF)** | O(log N + |q|·avg_df) | Best of both worlds |

---

## API Reference

### Demo Vector Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/search?v=f1,f2,...&k=5&metric=cosine&algo=hnsw` | K-NN search |
| `GET` | `/search?v=...&category=cs` | **Phase 3**: Filtered K-NN search |
| `POST` | `/insert` | Insert a demo vector |
| `DELETE` | `/delete/{id}` | Delete by ID |
| `GET` | `/items` | List all demo vectors |
| `GET` | `/categories` | **Phase 3**: List unique category labels |
| `GET` | `/benchmark?v=...&k=5&metric=cosine` | Compare all 3 algorithms |
| `GET` | `/hnsw-info` | HNSW graph structure and layer stats |
| `GET` | `/stats` | Database statistics (incl. SQ8 + BM25 stats) |

### Document & RAG Endpoints

| Method | Endpoint | Body | Description |
|---|---|---|---|
| `POST` | `/doc/insert` | `{"title":"...","text":"..."}` | Embed and store document |
| `GET` | `/doc/list` | — | List all stored documents |
| `DELETE` | `/doc/delete/{id}` | — | Delete document chunk |
| `POST` | `/doc/ask` | `{"question":"...","k":3}` | RAG: retrieve + generate |
| `POST` | `/doc/hybrid-search` | `{"question":"...","k":3,"rrf_k":60}` | **Phase 3**: BM25+HNSW+RRF |
| `GET` | `/status` | — | Ollama status and model info |

### Persistence Endpoints (Phase 3)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/persist/save` | Manually save both indexes to disk |
| `GET` | `/persist/status` | View snapshot file paths and sizes |

### Example: Category-Filtered Search
```bash
curl "http://localhost:8080/search?v=0.9,0.85,...&k=5&metric=cosine&algo=hnsw&category=cs"
```

### Example: Hybrid Search
```bash
curl -X POST http://localhost:8080/doc/hybrid-search \
  -H "Content-Type: application/json" \
  -d '{"question":"What is dynamic programming?","k":5}'
```

### Example: Save to Disk
```bash
curl -X POST http://localhost:8080/persist/save
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Ollama: OFFLINE` | Run `ollama serve` in a separate terminal |
| Embedding takes forever | Ollama is downloading the model on first use — wait ~2 min |
| Port 8080 in use | Change port in `app.py` or kill process |
| LLM answer is slow | Normal — llama3.2 takes 10–30s on CPU. Use `llama3.2:1b` for faster responses |
| Hybrid search returns no results | Insert documents first via Tab 2 / `/doc/insert` |

### Use a Smaller/Faster Model
```bash
ollama pull llama3.2:1b
```
Then in `vectordb/ollama_client.py` change `gen_model = "llama3.2:1b"` and restart.

---

## Roadmap

### ✅ Phase 3 — Completed
- [x] Disk persistence (JSON snapshot save/load on startup/shutdown)
- [x] Scalar Quantization (SQ8) for 4× memory compression
- [x] Hybrid Search (BM25 + HNSW with Reciprocal Rank Fusion)
- [x] Metadata filtering (category-filtered HNSW traversal)

### Phase 4 — Advanced RAG (Planned)
- [ ] Cross-Encoder re-ranking for RAG
- [ ] Semantic chunking (embedding-based split points)
- [ ] Citation & sentence grounding
- [ ] HuggingFace `sentence-transformers` support (no Ollama dependency)

### Phase 5 — Visualization (Planned)
- [ ] 3D vector space visualizer (t-SNE / UMAP)
- [ ] Animated HNSW graph inspector

### Phase 6 — Multi-Tenant (Planned)
- [ ] Multi-collection management
- [ ] Python client SDK (`vectordb-client`)
- [ ] Docker & Docker Compose deployment

---

## License

MIT — use this however you want.

---

<div align="center">

Built with ❤️ to learn how vector databases actually work.

</div>
