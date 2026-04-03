import os
import fitz  # PyMuPDF
from cryptography.fernet import Fernet
from sentence_transformers import SentenceTransformer
import chromadb

# ── 1. Load the embedding model once (not every function call) ──
print("Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("Embedding model loaded!")

# ── 2. ChromaDB client ──
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="documents")


# ── 3. Extract text from PDF ──
def extract_text(file_path: str) -> str:
    doc = fitz.open(file_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    print(f"Extracted {len(full_text)} characters from {file_path}")
    return full_text


# ── 4. Split text into overlapping chunks ──
def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> list:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():  # skip empty chunks
            chunks.append(chunk)
        start = end - overlap  # move back by overlap amount
    print(f"Created {len(chunks)} chunks")
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


# ── 6. Embed chunks and store in ChromaDB ──
def embed_and_store(chunks: list, doc_name: str):
    print(f"Embedding {len(chunks)} chunks...")
    embeddings = embedder.encode(chunks).tolist()

    ids = [f"{doc_name}_chunk_{i}" for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids
    )
    print(f"Stored {len(chunks)} chunks in ChromaDB!")


# ── 7. Master function — runs the full pipeline ──
def ingest_document(file_path: str):
    print(f"\n--- Starting ingestion for: {file_path} ---")

    # Step 1: Extract text
    text = extract_text(file_path)

    # Step 2: Chunk it
    chunks = chunk_text(text)

    # Step 3: Encrypt original file
    encrypt_file(file_path)

    # Step 4: Embed and store chunks
    doc_name = os.path.basename(file_path).replace(" ", "_")
    embed_and_store(chunks, doc_name)

    print(f"\n✓ Ingestion complete for {file_path}")
    print(f"✓ {len(chunks)} chunks stored in ChromaDB")
    print(f"✓ Original file encrypted on disk")


# ── 8. Test it ──
if __name__ == "__main__":
    ingest_document("test.pdf")