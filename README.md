# GraphRAG Pipeline for Literary Knowledge Using Neo4j and LangChain

**Graf Pengetahuan Literatur Tier 4**

**Realasa Femmi Novelika** (5026231113) & **Haliza Putri Amelliani** (5026231213)

Graf Pengetahuan - Institut Teknologi Sepuluh Nopember (ITS) Surabaya

---

## Project Overview

Proyek ini membangun sistem **Graph-Augmented Retrieval (GraphRAG)** untuk domain literatur menggunakan Neo4j dan LangChain. Sistem menggabungkan dua pendekatan retrieval berbasis Knowledge Graph:

- **Text-to-Cypher**: pertanyaan natural language diterjemahkan ke Cypher query dan dieksekusi langsung ke Neo4j menggunakan `GraphCypherQAChain`
- **Vector RAG**: data buku dari graph di-embed menggunakan HuggingFace (`all-MiniLM-L6-v2`) dan disimpan sebagai vector index di Neo4j
- **Hybrid QA**: kedua hasil digabung oleh LLM menjadi satu jawaban yang akurat dan lengkap
- **Graph Analytics**: PageRank Centrality + Louvain Community Detection via GDS
- **ML on Graph**: FastRP Embedding + K-Means Clustering

---

## Arsitektur

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
        │       ├── Louvain Community Detection
        │       ├── FastRP Embedding (16 dim)
        │       └── K-Means Clustering (k=4)
        │
        └── [NodesProject.py] Hybrid QA Pipeline
                ├── LLM: OpenRouter (gpt-4o-mini)
                ├── Text-to-Cypher (GraphCypherQAChain)
                ├── Vector RAG (HuggingFace + Neo4jVector)
                └── Hybrid Answer Combiner (LLM)
```

---

## Spesifikasi Teknis

| Komponen | Spesifikasi |
|---|---|
| Database | Neo4j Desktop 2.1.4 (versi 2026.05.0) + GDS Plugin |
| Bahasa | Python 3.13 |
| LLM | OpenRouter — `openai/gpt-4o-mini` |
| Embedding | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| Dataset | Wikidata + DBpedia (237+ nodes: 70 Book, 62 Author, 66 Publisher, 39 Genre) |

---

## Struktur Repository

```
graf-pengetahuan/
├── setup_db.py          # Setup awal: koneksi & verifikasi Neo4j
├── graph_analytics.py   # GDS: PageRank, Louvain, FastRP, K-Means
├── NodesProject.py      # Hybrid QA: Text-to-Cypher + Vector RAG + LLM
├── README.md            # Dokumentasi ini
├── requirements.txt     # Dependency Python
├── .env.example         # Template environment variables
├── .gitignore           # File yang diabaikan git
└── screenshots/
    ├── ss1_koneksi_db.png
    ├── ss2_graph_builder.png
    ├── ss3_ml_clustering.png
    └── ss4_llm_rag_cypher.png
```

---

## Instalasi & Konfigurasi

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
NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password_neo4j_kamu
OPENAI_API_KEY=sk-or-v1-xxxxxxxxxxxx
OPENAI_API_BASE=https://openrouter.ai/api/v1
```

### 6. Setup Neo4j

1. Buka **Neo4j Desktop** → Start database
2. Install **GDS Plugin**: klik `...` pada database → tab **Plugins** → install **Graph Data Science**
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

# 3. Main app - Hybrid QA interaktif
python NodesProject.py
```

### Detail setiap file

**`setup_db.py`** koneksi Neo4j, verifikasi GDS, tampilkan statistik node

**`graph_analytics.py`** jalankan GDS pipeline:
- PageRank ke grafik `pagerank.png`
- Louvain Community Detection
- FastRP Embedding + K-Means Clustering → grafik `clustering.png`

**`NodesProject.py`** Hybrid QA interaktif:
1. Koneksi Neo4j + load schema
2. Build vector index dari data buku (proses ~1-2 menit pertama kali)
3. Masuk ke **interactive loop** ketik pertanyaan, tekan Enter
4. Ketik `exit` untuk keluar

**Contoh pertanyaan:**
```
User: Siapa penulis buku dengan genre Mystery?
User: Rekomendasikan buku yang diterbitkan oleh Penguin Books
User: Genre apa yang paling banyak ditulis?
```

---

## Dependencies

```
# requirements.txt
neo4j>=5.0.0
langchain>=0.2.0
langchain-community>=0.2.0
langchain-openai>=0.1.0
langchain-neo4j>=0.1.0
openai>=1.0.0
python-dotenv>=1.0.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
sentence-transformers>=2.2.0
```

Install sekaligus:
```bash
pip install neo4j langchain langchain-community langchain-openai langchain-neo4j openai python-dotenv pandas numpy matplotlib scikit-learn sentence-transformers
```

openrouter.ai/docs
