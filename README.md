# Beyond Search A GraphRAG Pipeline for Literary Knowledge Using Neo4j and LangChain

**Graf Pengetahuan Literatur Tier 4**

**Realasa Femmi Novelika** (5026231113)
**Haliza Putri Amelliani** (5026231213)  
Mata Kuliah: Graf Pengetahuan
Institut Teknologi Sepuluh Nopember (ITS) Surabaya

---

## 📖 Deskripsi Proyek

Proyek ini membangun sistem **Graph-Augmented Retrieval (GraphRAG)** untuk domain literatur menggunakan Neo4j dan LangChain. Sistem menggabungkan dua pendekatan retrieval berbasis Knowledge Graph:

* **Text-to-Cypher**: Pertanyaan *natural language* diterjemahkan ke Cypher query dan dieksekusi langsung ke Neo4j menggunakan `GraphCypherQAChain`.
* **Vector RAG**: Data buku dari graph diekstrak, di-embed menggunakan HuggingFace (`all-MiniLM-L6-v2`), dan disimpan kembali sebagai vector index di Neo4j (`BookChunk`).
* **Hybrid QA**: Kedua hasil retrieval (faktual dari Cypher dan semantik dari Vector) digabungkan oleh LLM menjadi satu jawaban yang akurat, lengkap, dan minim halusinasi.
* **Graph Analytics & ML**: Pemanfaatan Neo4j Graph Data Science (GDS) untuk PageRank Centrality, Louvain Community Detection, dan FastRP Embedding.

---

## Arsitektur Sistem
```
[Wikidata / DBpedia]
│ SPARQL Query
▼
[Neo4j Knowledge Graph]
Book ──WRITTEN_BY──► Author
Book ──BELONGS_TO_GENRE──► Genre
Book ──PUBLISHED_BY──► Publisher
│
├── [graph_analytics.py] GDS Pipeline
│       ├── PageRank Centrality
│       └── Louvain Community Detection
│        
│
└── [NodesProject.py] Hybrid QA Pipeline
├── LLM: OpenRouter (gpt-4o-mini)
├── Text-to-Cypher (GraphCypherQAChain)
├── Vector RAG (HuggingFace + Neo4jVector)
└── Hybrid Answer Combiner (LLM)
```
---

## ⚙️ Spesifikasi Teknis

| Komponen | Spesifikasi |
|---|---|
| **Database** | Neo4j Desktop 2.1.4 (versi 2026.05.0) + GDS Plugin |
| **Bahasa** | Python 3.13 + Cypher |
| **LLM** | OpenRouter `openai/gpt-4o-mini` |
| **Embedding** | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (384 dim) |
| **Dataset** | Wikidata + DBpedia (237+ nodes: 70 Book, 62 Author, 66 Publisher, 39 Genre) |

---

## 📁 Struktur Repository

```
Beyond-Search/
├── .gitignore                   # File untuk mengabaikan file yang tidak diperlukan git
├── NodesProject.py              # Hybrid QA: Text-to-Cypher + Vector RAG + LLM (Interactive)
├── README.md                    # Dokumentasi ini
├── data_literatur_final (1).csv # Dataset mentah domain literatur dari Wikidata/DBpedia
├── graph_analytics.py           # GDS Pipeline: PageRank, Louvain, dan FastRP Embedding
├── setup_db.py                  # Setup awal: koneksi & verifikasi database Neo4j
├── requirements.txt             # Dependency Python berkas proyek
└── .env.example                 # Template environment variables
 
```

---
## 🚀 Instalasi & Konfigurasi

### 1. Prerequisites

- Python 3.10+
- Neo4j Desktop 2.x dengan **GDS Plugin** aktif
- API Key OpenRouter: https://openrouter.ai

### 2. Clone Repository

```bash
git clone https://github.com/realasa23/Beyond-Search
cd graf-pengetahuan
```
### 3. Setup Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```
### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Konfigurasi Environment

Salin file template:

```bash
cp .env.example .env
```

Isi file `.env`:

```env
NEO4J_URI=neo4j://uri
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=pass
OPENAI_API_KEY=sk-or-v1-key
OPENAI_API_BASE=https://openrouter.ai/api/v1
```
### 6. Setup Neo4j

1. **Neo4j Desktop** → Start database
2. Install **GDS Plugin**: klik `...` pada database lalu tab **Plugins** dan install **Graph Data Science**
3. Verifikasi GDS aktif di Neo4j Browser:
   ```cypher
   RETURN gds.version()
   ```

---
## Cara Run

Jalankan secara berurutan:

```bash
# 1. Setup DB 
python setup_db.py

# 2. ML & Graph Analytics
python graph_analytics.py

# 3. Main app — Hybrid QA interaktif
python NodesProject.py
```

