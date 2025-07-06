import chromadb
from sentence_transformers import SentenceTransformer
from typing import List
from chunk import chunkDoc

# Initialize ChromaDB persistent client and collection
chroma_client = chromadb.PersistentClient(path="chromadb_data")
collection = chroma_client.get_or_create_collection("law_texts")

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

def embedding(text: str):
    """
    Generate embedding for the input text and save it to ChromaDB.
    Returns the embedding vector.
    """
    vector = model.encode(text)
    # Save to ChromaDB (using text as the document id for simplicity)
    collection.upsert(
        documents=[text],
        embeddings=[vector.tolist()],
        ids=[str(hash(text))]
    )
    return vector

def embed_chunks_from_chunkpy() -> List:
    """
    Reads chunks from chunkDoc in chunk.py and embeds each chunk one by one.
    Returns a list of embedding vectors.
    """
    chunks = chunkDoc()
    vectors = []
    for i, chunk in enumerate(chunks):
        vector = embedding(chunk)
        vectors.append(vector)
        print(f"Embedded chunk {i+1}/{len(chunks)}")
    return vectors

if __name__ == "__main__":
    embed_chunks_from_chunkpy()
    print(chroma_client.list_collections())
