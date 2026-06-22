"""
Graph Analytics dengan Neo4j GDS
==================================
Dijalankan SETELAH llm_graph_builder.py selesai membangun graph.

Pipeline:
1. Project graph ke GDS in-memory projection
2. Degree Centrality  -> simpan ke Book.degreeCentrality
3. Louvain Community  -> simpan ke semua node .communityId
4. Jaccard Similarity -> buat relasi (:Book)-[:SIMILAR_TO {score}]->(:Book)
5. Tampilkan ringkasan hasil + top nodes per komunitas

Jalankan: python graph_analytics.py
"""

import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

LOCAL_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
LOCAL_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
LOCAL_PASSWORD = os.getenv("NEO4J_PASSWORD")

GRAPH_NAME = "literatur-graph"

driver = GraphDatabase.driver(LOCAL_URI, auth=(LOCAL_USERNAME, LOCAL_PASSWORD))



# UTILS


def run(session, query: str, params: dict = None):
    result = session.run(query, params or {})
    return result.data()


def drop_projection_if_exists(session):
    exists = run(session, """
        CALL gds.graph.exists($name) YIELD exists
        RETURN exists
    """, {"name": GRAPH_NAME})
    if exists and exists[0]["exists"]:
        session.run("CALL gds.graph.drop($name)", {"name": GRAPH_NAME})
        print(f"  [!] Projection lama '{GRAPH_NAME}' di-drop.")



# 1. BUAT IN-MEMORY PROJECTION


def create_projection(session):
    """
    Project semua node (Book, Author, Genre, Publisher) dan semua relasi
    sebagai UNDIRECTED agar bisa dipakai Louvain & Degree.
    """
    print("\n[1/4] Membuat in-memory graph projection...")

    drop_projection_if_exists(session)

    result = run(session, """
        CALL gds.graph.project(
            $name,
            ['Book', 'Author', 'Genre', 'Publisher'],
            {
                WRITTEN_BY:      { orientation: 'UNDIRECTED' },
                BELONGS_TO_GENRE: { orientation: 'UNDIRECTED' },
                PUBLISHED_BY:    { orientation: 'UNDIRECTED' }
            }
        )
        YIELD graphName, nodeCount, relationshipCount
        RETURN graphName, nodeCount, relationshipCount
    """, {"name": GRAPH_NAME})

    info = result[0]
    print(f"  -> Projection '{info['graphName']}' berhasil.")
    print(f"     Nodes: {info['nodeCount']} | Relationships: {info['relationshipCount']}")



# 2. DEGREE CENTRALITY


def run_degree_centrality(session):
    """
    Hitung Degree Centrality untuk semua node.
    Simpan hasilnya ke properti .degreeCentrality di masing-masing node.
    Node Book dengan degree tinggi = buku yang terhubung ke banyak Author/Genre/Publisher.
    """
    print("\n[2/4] Menjalankan Degree Centrality...")

    run(session, """
        CALL gds.degree.write(
            $name,
            { writeProperty: 'degreeCentrality' }
        )
        YIELD nodePropertiesWritten, centralityDistribution
        RETURN nodePropertiesWritten, centralityDistribution
    """, {"name": GRAPH_NAME})

    top_books = run(session, """
        MATCH (b:Book)
        WHERE b.degreeCentrality IS NOT NULL
        RETURN b.name AS book, b.degreeCentrality AS degree
        ORDER BY degree DESC
        LIMIT 10
    """)

    print("\n  Top 10 Buku (Degree Centrality Tertinggi):")
    print(f"  {'No':<4} {'Buku':<50} {'Degree':>8}")
    print("  " + "-" * 65)
    for i, row in enumerate(top_books, 1):
        print(f"  {i:<4} {str(row['book'])[:48]:<50} {row['degree']:>8.2f}")



# 3. LOUVAIN COMMUNITY DETECTION


def run_louvain(session):
    """
    Deteksi komunitas menggunakan Louvain.
    Simpan .communityId ke semua node.
    Komunitas = cluster buku yang terhubung ke Author/Genre/Publisher yang sama.
    """
    print("\n[3/4] Menjalankan Louvain Community Detection...")

    stats = run(session, """
        CALL gds.louvain.write(
            $name,
            { writeProperty: 'communityId' }
        )
        YIELD communityCount, modularity, modularities
        RETURN communityCount, modularity
    """, {"name": GRAPH_NAME})

    info = stats[0]
    print(f"  -> Total komunitas ditemukan: {info['communityCount']}")
    print(f"  -> Modularity score: {info['modularity']:.4f}")


    communities = run(session, """
        MATCH (b:Book)
        WHERE b.communityId IS NOT NULL
        RETURN b.communityId AS community, count(b) AS bookCount
        ORDER BY bookCount DESC
        LIMIT 10
    """)

    print("\n  Top 10 Komunitas (berdasarkan jumlah buku):")
    print(f"  {'Community ID':<15} {'Jumlah Buku':>12}")
    print("  " + "-" * 30)
    for row in communities:
        print(f"  {row['community']:<15} {row['bookCount']:>12}")


    biggest_community = communities[0]["community"] if communities else None
    if biggest_community:
        sample = run(session, """
            MATCH (b:Book)
            WHERE b.communityId = $cid
            RETURN b.name AS book, b.degreeCentrality AS degree
            ORDER BY degree DESC
            LIMIT 5
        """, {"cid": biggest_community})

        print(f"\n  Contoh buku dari komunitas terbesar (ID: {biggest_community}):")
        for row in sample:
            print(f"    - {row['book']} (degree: {row['degree']})")



# 4. JACCARD SIMILARITY (Book ↔ Book via Genre/Author)


def run_jaccard_similarity(session):
    """
    Hitung Jaccard Similarity antar buku berdasarkan kesamaan Genre dan Author.
    Caranya: bangun node2vec-style neighbor set dari Author + Genre,
    lalu gunakan gds.nodeSimilarity (Jaccard) pada projection khusus Book–Genre dan Book–Author.

    Hasilnya disimpan sebagai relasi (:Book)-[:SIMILAR_TO {score}]->(:Book).
    """
    print("\n[4/4] Menjalankan Jaccard Node Similarity (Book ↔ Book via Genre + Author)...")

    # Drop projection lama kalau ada (perlu projection berbeda: hanya Book–Genre dan Book–Author)
    SIMILARITY_GRAPH = "book-similarity-graph"

    exists = run(session, """
        CALL gds.graph.exists($name) YIELD exists RETURN exists
    """, {"name": SIMILARITY_GRAPH})
    if exists and exists[0]["exists"]:
        session.run("CALL gds.graph.drop($name)", {"name": SIMILARITY_GRAPH})

    # Project hanya Book, Author, Genre dengan relasi DIRECTED (natural)
    run(session, """
        CALL gds.graph.project(
            $name,
            ['Book', 'Author', 'Genre'],
            {
                WRITTEN_BY:       { orientation: 'NATURAL' },
                BELONGS_TO_GENRE: { orientation: 'NATURAL' }
            }
        )
        YIELD graphName, nodeCount, relationshipCount
    """, {"name": SIMILARITY_GRAPH})

    # Hapus relasi SIMILAR_TO lama jika ada
    session.run("MATCH ()-[r:SIMILAR_TO]->() DELETE r")


    stats = run(session, """
        CALL gds.nodeSimilarity.write(
            $name,
            {
                writeRelationshipType: 'SIMILAR_TO',
                writeProperty: 'score',
                topK: 5,
                similarityCutoff: 0.1
            }
        )
        YIELD nodesCompared, relationshipsWritten, similarityDistribution
        RETURN nodesCompared, relationshipsWritten,
               similarityDistribution.mean AS meanScore,
               similarityDistribution.p75  AS p75Score,
               similarityDistribution.max  AS maxScore
    """, {"name": SIMILARITY_GRAPH})

    if stats:
        info = stats[0]
        print(f"  -> Nodes dibandingkan : {info['nodesCompared']}")
        print(f"  -> Relasi SIMILAR_TO  : {info['relationshipsWritten']}")
        print(f"  -> Jaccard mean/p75/max: "
              f"{info['meanScore']:.3f} / {info['p75Score']:.3f} / {info['maxScore']:.3f}")

    # Drop projection similarity (sudah tidak dibutuhkan)
    session.run("CALL gds.graph.drop($name)", {"name": SIMILARITY_GRAPH})


    top_pairs = run(session, """
        MATCH (a:Book)-[r:SIMILAR_TO]->(b:Book)
        RETURN a.name AS book1, b.name AS book2, r.score AS score
        ORDER BY score DESC
        LIMIT 10
    """)

    print("\n  Top 10 Pasangan Buku Paling Mirip (Jaccard):")
    print(f"  {'Buku 1':<35} {'Buku 2':<35} {'Score':>7}")
    print("  " + "-" * 80)
    for row in top_pairs:
        b1 = str(row["book1"])[:33]
        b2 = str(row["book2"])[:33]
        print(f"  {b1:<35} {b2:<35} {row['score']:>7.4f}")



# 5. CLEANUP & SUMMARY


def cleanup(session):
    """Drop main projection setelah semua analitik selesai."""
    exists = run(session, """
        CALL gds.graph.exists($name) YIELD exists RETURN exists
    """, {"name": GRAPH_NAME})
    if exists and exists[0]["exists"]:
        session.run("CALL gds.graph.drop($name)", {"name": GRAPH_NAME})
        print(f"\n  [✓] Projection '{GRAPH_NAME}' di-drop.")


def print_summary(session):
    """Ringkasan properti yang sudah ditulis ke graph."""
    print("\n" + "=" * 60)
    print("RINGKASAN PROPERTI YANG DITAMBAHKAN KE GRAPH")
    print("=" * 60)

    total_books = run(session, "MATCH (b:Book) RETURN count(b) AS n")[0]["n"]
    with_degree = run(session, """
        MATCH (b:Book) WHERE b.degreeCentrality IS NOT NULL RETURN count(b) AS n
    """)[0]["n"]
    with_community = run(session, """
        MATCH (b:Book) WHERE b.communityId IS NOT NULL RETURN count(b) AS n
    """)[0]["n"]
    similar_rels = run(session, """
        MATCH ()-[r:SIMILAR_TO]->() RETURN count(r) AS n
    """)[0]["n"]

    print(f"  Total buku               : {total_books}")
    print(f"  Buku dengan degreeCent.  : {with_degree}")
    print(f"  Buku dengan communityId  : {with_community}")
    print(f"  Relasi SIMILAR_TO        : {similar_rels}")
    print("\n  Node properties baru:")
    print("    Book.degreeCentrality  -> float (konektivitas buku)")
    print("    Book.communityId       -> int   (cluster komunitas)")
    print("    SIMILAR_TO.score       -> float (Jaccard similarity)")
    print("\n  Bisa langsung dipakai di query_engine.py untuk:")
    print("    - Rekomendasi buku serupa")
    print("    - Filter buku paling 'terhubung' dalam suatu genre")
    print("    - Analisis klaster per komunitas")



# MAIN


if __name__ == "__main__":
    print("=" * 60)
    print("GRAPH ANALYTICS — Neo4j GDS")
    print("Pastikan llm_graph_builder.py sudah dijalankan lebih dulu!")
    print("=" * 60)

    with driver.session() as session:
        create_projection(session)
        run_degree_centrality(session)
        run_louvain(session)
        run_jaccard_similarity(session)
        cleanup(session)
        print_summary(session)

    driver.close()
    print("\n Analitik selesai! Properti baru sudah tersimpan di Neo4j.")