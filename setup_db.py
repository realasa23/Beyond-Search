"""
LLM Graph Builder (sumber data: GitHub CSV)
============================================
Opsi A: Ekstraksi entitas & relasi dari teks tidak terstruktur menggunakan LLM
(via OpenRouter), lalu hasil ekstraksinya di-MERGE ke Neo4j sebagai graph.

Alur:
1. fetch_csv_from_github()   -> download CSV mentah dari repo GitHub
2. build_raw_documents()     -> gabungkan baris per judul buku jadi satu
                                 blurb teks ("sumber tidak terstruktur")
3. extract_entities_with_llm() -> panggil LLM, minta JSON terstruktur
4. ingest_to_neo4j()          -> MERGE node Book/Author/Genre/Publisher + relasi

Jalankan: python llm_graph_builder.py
"""

import os
import csv
import io
import json
import time
import requests
from collections import defaultdict
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI

load_dotenv()

LOCAL_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
LOCAL_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
LOCAL_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")

CSV_URL = (
    "https://raw.githubusercontent.com/realasa23/Beyond-Search/"
    "a40cbe9b374444a75c415d8cac41b6d6801a5f69/"
    "data_literatur_final%20(1).csv"
)

MAX_BOOKS = int(os.getenv("MAX_BOOKS", "100"))

driver = GraphDatabase.driver(LOCAL_URI, auth=(LOCAL_USERNAME, LOCAL_PASSWORD))
llm_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)



# 1. AMBIL DATASET


def fetch_csv_from_github(url: str) -> list:
    print(f"Mengambil CSV dari GitHub: {url}")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    reader = csv.DictReader(io.StringIO(resp.text))
    rows = list(reader)
    print(f"  -> {len(rows)} baris berhasil diambil.")
    return rows



# 2. BUILD "TEKS TIDAK TERSTRUKTUR" PER BUKU

def build_raw_documents(rows: list) -> list:
    grouped = defaultdict(lambda: {"authors": set(), "genres": set(),
                                    "publishers": set(), "years": set()})

    for row in rows:
        title = (row.get("final_book_title") or "").strip()
        if not title:
            continue
        author = (row.get("authorLabel") or "").strip()
        genre = (row.get("genreLabel") or "").strip()
        publisher = (row.get("final_publisher") or "").strip()
        year = (row.get("release_year") or "").strip()

        bucket = grouped[title]
        if author:
            bucket["authors"].add(author)
        if genre:
            bucket["genres"].add(genre)
        if publisher:
            bucket["publishers"].add(publisher)
        if year:
            bucket["years"].add(year)

    documents = []
    for title, info in grouped.items():
        parts = [f"Title: {title}."]

        if info["authors"]:
            parts.append(f"Written by {', '.join(sorted(info['authors']))}.")
        else:
            parts.append("Author is not specified in the source record.")

        if info["genres"]:
            parts.append(f"Genre(s): {', '.join(sorted(info['genres']))}.")

        if info["publishers"]:
            parts.append(f"Published by {', '.join(sorted(info['publishers']))}.")

        if info["years"]:
            parts.append(f"Release year: {', '.join(sorted(info['years']))}.")

        documents.append({"title": title, "text": " ".join(parts)})

    print(f"  -> {len(documents)} dokumen (blurb) unik dibangun dari CSV.")
    return documents



# 3. EKSTRAKSI ENTITAS MENGGUNAKAN LLM


EXTRACTION_SYSTEM_PROMPT = """You are an information-extraction engine for a
library knowledge graph. Given a short, unstructured text about a book,
extract the following entities and relationships.

Return ONLY valid JSON (no markdown, no commentary) with this exact schema:
{
  "book": "<book title>",
  "authors": ["<author name>", ...],
  "genres": ["<genre name>", ...],
  "publishers": ["<publisher name>", ...],
  "release_year": "<year or null>"
}

Rules:
- "book" must be the exact title as it appears in the text.
- If a field is not mentioned, return an empty list (or null for release_year).
- Do not invent information that is not present in the text.
"""


def extract_entities_with_llm(text: str) -> dict:
    response = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        temperature=0,
    )

    content = response.choices[0].message.content.strip()

    if content.startswith("```"):
        content = content.strip("`")
        content = content.replace("json\n", "", 1).strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        print(f"  [!] Gagal parse JSON dari LLM, dilewati. Raw output:\n{content}")
        return None



# 4. INGESTION KE NEO4J


def clear_database(session):
    print("Menghapus data lama...")
    session.run("MATCH (n) DETACH DELETE n")
    print("Database berhasil dikosongkan.")


def ingest_to_neo4j(session, extracted: dict):
    book = extracted.get("book")
    if not book:
        return

    session.run(
        "MERGE (b:Book {name: $book}) SET b.release_year = $year",
        book=book, year=extracted.get("release_year"),
    )

    for author in extracted.get("authors", []):
        session.run(
            """
            MERGE (a:Author {name: $author})
            WITH a
            MATCH (b:Book {name: $book})
            MERGE (b)-[:WRITTEN_BY]->(a)
            """,
            author=author, book=book,
        )

    for genre in extracted.get("genres", []):
        session.run(
            """
            MERGE (g:Genre {name: $genre})
            WITH g
            MATCH (b:Book {name: $book})
            MERGE (b)-[:BELONGS_TO_GENRE]->(g)
            """,
            genre=genre, book=book,
        )

    for publisher in extracted.get("publishers", []):
        session.run(
            """
            MERGE (p:Publisher {name: $publisher})
            WITH p
            MATCH (b:Book {name: $book})
            MERGE (b)-[:PUBLISHED_BY]->(p)
            """,
            publisher=publisher, book=book,
        )



# 5. MAIN PIPELINE


def run_pipeline(session, documents: list):
    for i, doc in enumerate(documents, start=1):
        print(f"\n[{i}/{len(documents)}] Mengekstrak: {doc['title']}")
        print(f"  Raw text: {doc['text']}")

        extracted = extract_entities_with_llm(doc["text"])
        if extracted is None:
            continue

        print(f"  -> Hasil ekstraksi LLM: {json.dumps(extracted, ensure_ascii=False)}")

        ingest_to_neo4j(session, extracted)
        print("  -> Berhasil di-MERGE ke Neo4j.")

        time.sleep(0.5)  # jaga-jaga rate limit API


if __name__ == "__main__":
    rows = fetch_csv_from_github(CSV_URL)
    documents = build_raw_documents(rows)

    if MAX_BOOKS is not None:
        documents = documents[:MAX_BOOKS]
        print(f"  -> Dibatasi ke {MAX_BOOKS} buku pertama (ubah MAX_BOOKS di .env untuk semua).")

    with driver.session() as session:
        clear_database(session)
        run_pipeline(session, documents)

    driver.close()
    print("\nSetup selesai! Data hasil ekstraksi LLM dari CSV GitHub sudah masuk ke Neo4j.")