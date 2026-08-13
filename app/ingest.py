import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from pinecone import Pinecone, ServerlessSpec
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "legixo-corpus")
CORPUS_DIR = Path("corpus")
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSION = 3072

files = list(CORPUS_DIR.glob("*"))

def chunk_text(text, chunk_size=600, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += (chunk_size - overlap)
    return chunks

def embed_chunks():
    ai_client = genai.Client(api_key=GEMINI_API_KEY)
    pc = Pinecone(api_key= PINECONE_API_KEY)

    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"Creating Pinecone index '{PINECONE_INDEX_NAME}'...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    index = pc.Index(PINECONE_INDEX_NAME)
    print(index)

    files = list(CORPUS_DIR.glob("*"))
    print(f"Processing {len(files)} files from {CORPUS_DIR}...")

    vectors_to_upload = []

    for file_path in files:
        if not file_path.is_file():
            continue

        text = file_path.read_text(encoding="utf-8")
        chunks = chunk_text(text)
        print(f"File: {file_path.name} -> {len(chunks)} chunks")

        for idx, chunk in enumerate(chunks):
            # Generate embedding vector
            res = ai_client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=chunk
            )

            vector = res.embeddings[0].values

            # Package vector with metadata
            chunk_id = f"{file_path.name}_chunk_{idx}"
            vectors_to_upload.append({
                "id": chunk_id,
                "values": vector,
                "metadata": {
                    "text": chunk,
                    "filename": file_path.name
                }
            })
    print(f"\nUpserting {len(vectors_to_upload)} total vectors to Pinecone...")
    index.upsert(vectors=vectors_to_upload)
    
    print("\n✅ Ingestion Complete!")
    stats = index.describe_index_stats()
    print(f"Pinecone Vector Count: {stats.get('total_vector_count')}")

if __name__=="__main__":
    embed_chunks()