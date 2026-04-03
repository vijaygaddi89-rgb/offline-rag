import os
import fitz
from cryptography.fernet import Fernet
from sentence_transformers import SentenceTransformer
import chromadb

# ── 1. Load embedding model once ──
print("Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("Embedding model loaded!")

# ── 2. ChromaDB client ──
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="documents")


# ── 3. Extract text page by page ──
def extract_text_by_page(file_path: str) -> list:
    doc = fitz.open(file_path)
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        if text.strip():
            pages.append({
                "text": text,
                "page": page_num
            })
    doc.close()
    print(f"Extracted {len(pages)} pages from {file_path}")
    return pages


# ── 4. Chunk each page separately ──
def chunk_pages(pages: list, chunk_size: int = 512, overlap: int = 50) -> list:
    chunks = []
    for page_data in pages:
        text = page_data["text"]
        page_num = page_data["page"]
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append({
                    "text": chunk,
                    "page": page_num
                })
            start = end - overlap
    print(f"Created {len(chunks)} chunks across {len(pages)} pages")
    return chunks


# ── 5. Encrypt the original file ──
def encrypt_file(file_path: str) -> str:
    key = Fernet.generate_key()
    fernet = Fernet(key)

    with open(file_path, 'rb') as f:
        original_data = f.read()

    encrypted_data = fernet.encrypt(original_data)

    encrypted_path = file_path + ".encrypted"
    with open(encrypted_path, 'wb') as f:
        f.write(encrypted_data)

    key_path = file_path + ".key"
    with open(key_path, 'wb') as f:
        f.write(key)

    print(f"File encrypted → {encrypted_path}")
    print(f"Key saved → {key_path}")
    return encrypted_path


# ── 6. Embed chunks and store with page metadata ──
def embed_and_store(chunks: list, doc_name: str):
    print(f"Embedding {len(chunks)} chunks...")

    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts).tolist()
    ids = [f"{doc_name}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"page": c["page"], "doc": doc_name} for c in chunks]

    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )
    print(f"Stored {len(chunks)} chunks with page numbers in ChromaDB!")


# ── 7. Master function ──
def ingest_document(file_path: str):
    print(f"\n--- Starting ingestion for: {file_path} ---")

    # Step 1: Extract text page by page
    pages = extract_text_by_page(file_path)

    # Step 2: Chunk pages
    chunks = chunk_pages(pages)

    # Step 3: Encrypt original file
    encrypt_file(file_path)

    # Step 4: Embed and store with page metadata
    doc_name = os.path.basename(file_path).replace(" ", "_")
    embed_and_store(chunks, doc_name)

    print(f"\n✓ Ingestion complete for {file_path}")
    print(f"✓ {len(chunks)} chunks stored with page numbers")
    print(f"✓ Original file encrypted on disk")


# ── 8. Test ──
if __name__ == "__main__":
    ingest_document("test.pdf")