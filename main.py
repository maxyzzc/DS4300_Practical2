import redis
import json
import re
import numpy as np
import ollama
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# CONNECT TO REDIS
redis_client = redis.Redis(host="localhost",
                           port=6379,
                           password=None,
                           decode_responses=True)

# Load Embedding Model (optimized for Llama3)
sentence_transformer = "sentence-transformers/all-MiniLM-L6-v2"
sentence_transformer = "sentence-transformers/all-mpnet-base-v2"
sentence_transformer = "sentence-transformers/all-MiniLM-L6-v2"

embedding_model = SentenceTransformer(sentence_transformer)

def reset_database():
    """Flush entire Redis database, removing all stored keys."""
    redis_client.flushdb()
    print("Redis database has been reset.")


# Read and Preprocess the Text File
def read_and_preprocess(file_path):
    """Reads a text file, converts to lowercase, and removes extra whitespace"""
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    # Convert to lowercase and remove extra spaces/new lines
    text = text.lower()
    text = re.sub(r"\s+", " ", text)

    return text


def chunk_text(text, chunk_size=250, overlap=50):
    """Splits text into overlapping chunks"""
    words = text.split()
    chunks = []

    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

        # Stop when reaching the end of the document
        if i + chunk_size >= len(words):
            break

    return chunks


def generate_embeddings(chunks):
    """Generates vector embeddings for each text chunk using the embedding model."""
    return embedding_model.encode(chunks).tolist()


def store_in_redis(chunks, embeddings):
    """Stores text chunks and corresponding vector embeddings in Redis."""
    pipe = redis_client.pipeline()

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        chunk_id = f"doc_1_chunk_{i}"

        # Store text and vector embedding in Redis Hash
        pipe.hset(chunk_id, mapping={
            "text": chunk,
            "vector": json.dumps(embedding)
        })

    pipe.execute()
    print(f"Stored {len(chunks)} chunks in Redis.")


def retrieve_relevant_chunks(query, top_k=3):
    """Retrieves the top-k most relevant chunks from Redis based on query embedding similarity."""
    # Generate embedding for query
    query_embedding = embedding_model.encode([query])[0]

    # Fetch all stored chunk keys
    chunk_keys = redis_client.keys("doc_1_chunk_*")

    if not chunk_keys:
        print("No data found in Redis.")
        return []

    similarities = []

    # Compute similarity between query embedding and stored embeddings
    for chunk_key in chunk_keys:
        stored_data = redis_client.hgetall(chunk_key)
        stored_vector = json.loads(stored_data["vector"])  # Convert back to list

        # Compute cosine similarity
        similarity = np.dot(query_embedding, stored_vector) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(stored_vector))
        similarities.append((chunk_key, stored_data["text"], similarity))

    # Sort by highest similarity and get top_k chunks
    similarities.sort(key=lambda x: x[2], reverse=True)
    top_chunks = [chunk[1] for chunk in similarities[:top_k]]

    return top_chunks


def generate_llama_response(query, top_k=3):
    """Retrieves relevant chunks from Redis and generates a response using Llama3."""

    # Retrieve relevant text chunks
    relevant_chunks = retrieve_relevant_chunks(query, top_k)

    if not relevant_chunks:
        return "I couldn't find relevant information in the database."

    # Construct prompt with retrieved context
    context = "\n".join(relevant_chunks)
    prompt = f"Use the following lecture notes to answer the question:\n\n{context}\n\nQuestion: {query}\nAnswer:"

    # Query Ollama’s Llama3.2:3B model
    response = ollama.chat(model="llama3.2:3b", messages=[{"role": "user", "content": prompt}])

    return response["message"]["content"]

def run_llm(k=3):
    print(f"\n{75 * '-'}")
    while True:
        query = input("Input query here ('quit' to exit): ")
        print(f"{75*'-'}")

        if query == 'quit':
            break

        response = generate_llama_response(query, top_k=k)
        print("\nLlama3's Response:\n", response)
        print(f"\n{75*'-'}")

def main():

    # Parameters
    k = 5
    chunk_size = 200
    overlap = 50

    # Read and preprocess the text
    text = read_and_preprocess("data/document_1.txt")

    # Split text into chunks and embedd
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    embeddings = generate_embeddings(chunks)

    # Store the chunks and embeddings in Redis
    reset_database()
    store_in_redis(chunks, embeddings)

    print("\nDocument has been processed and stored in Redis.")

    run_llm(k=k)

if __name__ == "__main__":
    main()
