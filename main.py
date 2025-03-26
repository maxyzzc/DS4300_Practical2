import redis
import json
import re
import numpy as np
import ollama
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import chromadb
import faiss
import pandas as pd
import time

# CONNECT TO REDIS
redis_client = redis.Redis(host="localhost",
                           port=6379,
                           password=None,
                           decode_responses=True)

chroma_client = chromadb.PersistentClient(path="chroma_db")

faiss_text = {}
faiss_index = faiss.IndexFlatL2(384)


def reset_redis_database():
    """Flush entire Redis database, removing all stored keys."""
    redis_client.flushdb()
    print("Redis database has been reset.")


def reset_chroma_database():
    """Flush entire Chroma database, removing all stored collections."""
    collections = chroma_client.list_collections()

    for collection in collections:
        chroma_client.delete_collection(collection)

    print("Chroma database has been reset.")


def reset_faiss_database():
    """Resets the FAISS database by creating a new index and clearing the text store."""
    global faiss_index, faiss_text
    faiss_index = faiss.IndexFlatL2(384)
    faiss_text.clear()
    print("FAISS database has been reset.")


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


def generate_embeddings(chunks, embedding_model=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")):
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


def store_in_chroma(chunks, embeddings):
    """Stores text chunks and corresponding vector embeddings in ChromaDB."""
    collection_name = "text_chunks"

    # Create or get existing collection
    collection = chroma_client.get_or_create_collection(name=collection_name)

    # Prepare documents to insert
    documents = [
        {
            "id": f"doc_1_chunk_{i}",
            "metadata": {"text": chunk},  # Store text as metadata
            "embedding": embedding
        }
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]

    # Insert data into ChromaDB
    collection.add(
        ids=[doc["id"] for doc in documents],
        embeddings=[doc["embedding"] for doc in documents],
        metadatas=[doc["metadata"] for doc in documents]
    )

    print(f"Stored {len(chunks)} chunks in ChromaDB.")


def store_in_faiss(chunks, embeddings):
    """Stores text chunks and corresponding vector embeddings in FAISS."""
    global faiss_text

    embeddings_np = np.array(embeddings).astype("float32")
    start_id = faiss_index.ntotal
    end_id = start_id + len(embeddings)

    # Add embeddings to FAISS index
    faiss_index.add(embeddings_np)

    # Store text chunks in faiss_text using FAISS index IDs
    for i, chunk in enumerate(chunks):
        faiss_text[start_id + i] = chunk

    print(f"Stored {len(chunks)} chunks in FAISS.")


def retrieve_relevant_chunks(query, embedding_model, db="redis", top_k=3):
    """Retrieves the top-k most relevant chunks based on query embedding similarity."""
    query_embedding = embedding_model.encode([query])[0]

    if db == "redis":
        chunk_keys = redis_client.keys("doc_1_chunk_*")
        if not chunk_keys:
            print("No data found in Redis.")
            return []
        similarities = []
        for chunk_key in chunk_keys:
            stored_data = redis_client.hgetall(chunk_key)
            stored_vector = json.loads(stored_data["vector"])
            similarity = np.dot(query_embedding, stored_vector) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(stored_vector)
            )
            similarities.append((chunk_key, stored_data["text"], similarity))
        similarities.sort(key=lambda x: x[2], reverse=True)
        return [chunk[1] for chunk in similarities[:top_k]]

    elif db == "chroma":
        collection = chroma_client.get_or_create_collection(name="text_chunks")
        results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
        return [metadata["text"] for metadata in results["metadatas"][0]]

    elif db == "faiss":
        distances, indices = faiss_index.search(np.array([query_embedding]).astype("float32"), top_k)
        return [faiss_text[idx] for idx in indices[0] if idx in faiss_text]

    else:
        raise ValueError("Invalid database option. Choose from 'redis', 'chroma', or 'faiss'.")


def generate_llama_response(query, embedding_model, db, top_k=3, llm='llama3.2:3b'):
    """Retrieves relevant chunks from Redis and generates a response using Llama3."""

    # Retrieve relevant text chunks
    relevant_chunks = retrieve_relevant_chunks(query, embedding_model, db, top_k)

    if not relevant_chunks:
        return "I couldn't find relevant information in the database."

    # Construct prompt with retrieved context
    context = "\n".join(relevant_chunks)
    prompt = f"Use the following lecture notes to answer the question:\n\n{context}\n\nQuestion: Answer shortly. {query}\nAnswer:"
    print(prompt)

    # Query Ollama’s Llama3.2:3B model
    response = ollama.chat(model=llm, messages=[{"role": "user", "content": prompt}])

    return response["message"]["content"]


def run_llm(k=3, db="redis", embedding_model=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"), llm='llama3.2:3b'):
    print(f"\n{75 * '-'}")

    print(f"Ollama Model: {llm}")
    while True:
        query = input("Input query here ('quit' to exit): ")
        print(f"{75 * '-'}")

        if query == 'quit':
            break

        response = generate_llama_response(query, embedding_model, db, top_k=k, llm=llm)
        print("\nLlama3's Response:\n", response)
        print(f"\n{75 * '-'}")


def test_llm(embedding_model, questions, db, k=3):
    responses = []
    for q in questions:
        responses.append(generate_llama_response(q, embedding_model, db, k))
    return responses


def main():

    version = input("Enter 0 to run model. Enter 1 to test different model versions: ")

    if version == "1":

        # Initialize questions to ask each model

        questions = [
            "Describe the CAP theorem and how it applies to MongoDB.",
            "Describe how data is organized and indexed in a B+ tree.",
            "What are the key advantages of using NoSQL databases over traditional relational databases for large-scale information storage?",
            "In what scenario would you prefer using a B+ tree over a hash index?",
            "Write a MongoDB query to retrieve all documents where the nested field user.profile.age is less than 25.",
        ]
        # Parameters
        k = 5
        chunk_sizes = [200, 500, 1000]
        overlaps = [0, 50, 100]
        sentence_transformers = [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2"
            "sentence-transformers/all-MiniLM-L6-v2"
        ]
        databases = ["redis", "chroma", "faiss"]

        # Read and preprocess the text
        text = read_and_preprocess("document_1.txt")

        df = pd.DataFrame(
            columns=['time', 'chunk_size', 'overlaps', 'sentence_transformer', 'db', 'q1', 'q2', 'q3', 'q4', 'q5',
                     'q6', 'q7', 'q8', 'q9', 'q10', 'q11', 'q12', 'q13', 'q14', 'q15', 'q16', 'q17', 'q18', 'q19',
                     'q20'])

        all_responses = []

        # Split text into chunks and embedd
        for c in chunk_sizes:
            for o in overlaps:
                chunks = chunk_text(text, chunk_size=c, overlap=o)
                for s in sentence_transformers:
                    embedding_model = SentenceTransformer(s)
                    embeddings = generate_embeddings(chunks, embedding_model)

                    for db in databases:
                        if db == "redis":
                            # Store the chunks and embeddings in Redis
                            reset_redis_database()
                            store_in_redis(chunks, embeddings)
                        elif db == "chroma":
                            reset_chroma_database()
                            store_in_chroma(chunks, embeddings)
                        elif db == "faiss":
                            reset_faiss_database()
                            store_in_faiss(chunks, embeddings)
                            # Evaluate memory, speed of indexing, speed of querying

                        print("\nDocument has been processed and stored in", db, ".")

                        start_time = time.time()

                        responses = test_llm(embedding_model, questions, db, k)
                        responses.insert(0, time.time() - start_time)
                        responses.insert(1, c)
                        responses.insert(2, o)
                        responses.insert(3, s)
                        responses.insert(4, db)

                        all_responses.append(responses)

        df = pd.DataFrame(all_responses, columns=['time', 'chunk_size', 'overlaps', 'sentence_transformer', 'db', 'q1',
                                                  'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10', 'q11', 'q12',
                                                  'q13', 'q14', 'q15', 'q16', 'q17', 'q18', 'q19', 'q20'])

        df.to_csv("all_responses.csv")

    else:

        # Parameters
        k = 3
        chunk_size = 300
        overlap = 50

        # Read and preprocess the text
        text = read_and_preprocess("document_1.txt")

        # Split text into chunks and embedd
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        embeddings = generate_embeddings(chunks)

        # Store the chunks and embeddings in Redis
        reset_redis_database()
        store_in_redis(chunks, embeddings)

        # Running LLM
        llm = 'gemma3:1b'
        run_llm(k=k, db="redis", llm=llm)


if __name__ == "__main__":
    main()
