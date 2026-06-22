import os
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain, Neo4jVector
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceEmbeddings 
from dotenv import load_dotenv

load_dotenv()

# --- 1. SETUP KREDENSIAL ---
LOCAL_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
LOCAL_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
LOCAL_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")

print("Menghubungkan ke Neo4j Desktop Lokal...")

# --- 2. KONEKSI GRAF ---
graph = Neo4jGraph(
    url=LOCAL_URI,
    username=LOCAL_USERNAME,
    password=LOCAL_PASSWORD
)
graph.refresh_schema()
print(graph.schema)

# --- 3. SETUP LLM ---
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    temperature=0,
    openai_api_key=OPENAI_API_KEY,
    openai_api_base=OPENAI_API_BASE
)

# --- 4. CYPHER PROMPT ---
CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j Cypher query generator.

Schema:
{schema}

Important relationship directions in this graph:
- (Book)-[:WRITTEN_BY]->(Author)
- (Book)-[:BELONGS_TO_GENRE]->(Genre)
- (Book)-[:PUBLISHED_BY]->(Publisher)

Important rules:
- Always use case-insensitive matching using toLower()
- Use CONTAINS instead of = for genre/name matching

Only generate a valid Cypher query. Do not include any explanation or markdown.

Question: {question}
Cypher Query:
"""

cypher_prompt = PromptTemplate(
    input_variables=["schema", "question"],
    template=CYPHER_GENERATION_TEMPLATE
)

# --- 5. SETUP TEXT-TO-CYPHER CHAIN ---
cypher_chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True,
    allow_dangerous_requests=True,
    return_intermediate_steps=True,
    cypher_prompt=cypher_prompt
)

# --- 6. SETUP VECTOR RAG ---
print("\nMembangun vector index dari graph...")

query_result = graph.query("""
    MATCH (b:Book)
    OPTIONAL MATCH (b)-[:WRITTEN_BY]->(a:Author)
    OPTIONAL MATCH (b)-[:BELONGS_TO_GENRE]->(g:Genre)
    OPTIONAL MATCH (b)-[:PUBLISHED_BY]->(p:Publisher)
    RETURN b.name AS book,
           collect(DISTINCT a.name) AS authors,
           collect(DISTINCT g.name) AS genres,
           collect(DISTINCT p.name) AS publishers
""")

documents = []
for row in query_result:
    content = f"""
Book: {row['book']}
Authors: {', '.join(row['authors']) if row['authors'] else 'Unknown'}
Genres: {', '.join(row['genres']) if row['genres'] else 'Unknown'}
Publishers: {', '.join(row['publishers']) if row['publishers'] else 'Unknown'}
    """.strip()
    documents.append(Document(page_content=content, metadata={"book": row["book"]}))

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Neo4jVector.from_documents(
    documents=documents,
    embedding=embeddings,
    url=LOCAL_URI,
    username=LOCAL_USERNAME,
    password=LOCAL_PASSWORD,
    index_name="book_index",
    node_label="BookChunk",
    text_node_property="text",
    embedding_node_property="embedding"
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# RAG Chain pakai LCEL (tanpa RetrievalQA)
rag_prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:
{context}

Question: {question}
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

print("\n=== AI Jaringan Literatur Siap! ===")

# --- 7. HYBRID QA FUNCTION ---
def hybrid_qa(question: str):
    print(f"\n{'='*50}")
    print(f"User: {question}")

    # Step 1: Text-to-Cypher
    print("\n[Text-to-Cypher]")
    try:
        cypher_result = cypher_chain.invoke({"query": question})
        cypher_answer = cypher_result["result"]
        print(f"Cypher Answer: {cypher_answer}")
    except Exception as e:
        cypher_answer = ""
        print(f"Cypher Error: {e}")

    # Step 2: Vector RAG
    print("\n[Vector RAG]")
    try:
        rag_answer = rag_chain.invoke(question)
        print(f"RAG Answer: {rag_answer}")
    except Exception as e:
        rag_answer = ""
        print(f"RAG Error: {e}")

    # Step 3: Combine
    print("\n[Final Combined Answer]")
    combine_prompt = f"""
You are a helpful assistant for a literary knowledge graph system.

Question: {question}

Answer from graph query: {cypher_answer}
Answer from semantic search: {rag_answer}

Combine both answers into one accurate, complete response.
If one answer is empty or says 'I don't know', use the other one.
    """
    final_answer = llm.invoke(combine_prompt)
    print(f"Final: {final_answer.content}")
    return final_answer.content

# --- 8. INTERACTIVE LOOP ---
print("\nKetik pertanyaan kamu (atau 'exit' untuk keluar):")
while True:
    question = input("\nUser: ").strip()
    if question.lower() in ["exit", "quit", "keluar"]:
        print("Sampai jumpa!")
        break
    if not question:
        continue
    hybrid_qa(question)