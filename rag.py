import os
import time
from datetime import datetime
from sentence_transformers import SentenceTransformer
import chromadb
from llama_cpp import Llama

# ── 1. Load embedding model ──
print("Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("Embedding model loaded!")

# ── 2. Connect to ChromaDB ──
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="documents")

# ── 3. Load the LLM ──
print("Loading LLM... (this takes 10-30 seconds)")
llm = Llama(
    model_path="models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    n_ctx=2048,
    verbose=False
)
print("LLM loaded!")


# ── 4. Embed the user query ──
def embed_query(query: str) -> list:
    vector = embedder.encode(query).tolist()
    return vector


# ── 5. Search ChromaDB — returns chunks with page numbers ──
def retrieve_chunks(query: str, top_k: int = 4) -> list:
    query_vector = embed_query(query)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas"]
    )

    chunks = []
    for i, doc in enumerate(results['documents'][0]):
        page_num = results['metadatas'][0][i].get('page', '?')
        chunks.append({
            "text": doc,
            "page": page_num
        })

    print(f"Retrieved {len(chunks)} chunks from ChromaDB")
    return chunks


# ── 6. Format context with page numbers ──
def format_context(chunks: list) -> str:
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"[Source: Page {chunk['page']}]:\n{chunk['text']}\n\n"
    return context.strip()


# ── 7. Build prompt — citations at the end ──
def build_prompt(context: str, query: str) -> str:
    prompt = f"""You are a helpful assistant. Answer the user's question using ONLY the provided context below.
If the answer is not in the context, say "I could not find this information in the provided document."
Write your answer in clear complete sentences first.
At the very end of your answer on a new line add: "Sources: Page X" listing which pages you used.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""
    return prompt


# ── 8. Write to audit log ──
def log_audit(query: str, response: str, latency: float):
    with open("rag_audit.log", "a") as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"Timestamp : {datetime.now()}\n")
        f.write(f"Query     : {query}\n")
        f.write(f"Latency   : {latency:.2f} seconds\n")
        f.write(f"Response  : {response}\n")


# ── 9. Master query function ──
def answer_question(query: str) -> str:
    print(f"\n--- Query: {query} ---")
    start_time = time.time()

    # Step 1: retrieve relevant chunks
    chunks = retrieve_chunks(query)

    # Step 2: format context with page numbers
    context = format_context(chunks)

    # Step 3: build prompt
    prompt = build_prompt(context, query)

    # Step 4: generate answer
    print("Generating answer...")
    output = llm(prompt, max_tokens=512, stop=["QUESTION:", "CONTEXT:"])
    response = output['choices'][0]['text'].strip()

    # Step 5: append page sources at end if model missed them
    pages_used = sorted(set([c['page'] for c in chunks]))
    if "Sources:" not in response:
        pages_str = ", ".join([f"Page {p}" for p in pages_used])
        response += f"\n\n📄 Sources: {pages_str}"

    # Step 6: log to audit
    latency = time.time() - start_time
    log_audit(query, response, latency)

    print(f"Answer generated in {latency:.2f} seconds")
    return response


# ── 10. Test ──
if __name__ == "__main__":
    questions = [
        "What is the penalty for late payment?",
        "What are the confidentiality terms?",
        "How many days notice is needed to terminate the contract?"
    ]

    for question in questions:
        answer = answer_question(question)
        print(f"\nQ: {question}")
        print(f"A: {answer}")
        print("-" * 50)