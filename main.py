import tracemalloc
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


def reset_faiss_database(embedding_model):
    """Resets the FAISS database by creating a new index and clearing the text store."""
    global faiss_index, faiss_text

    if embedding_model == "sentence-transformers/all-mpnet-base-v2":
        faiss_index = faiss.IndexFlatL2(768)
    else:
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

    tracemalloc.start()
    start_time = time.time()

    embeddings = embedding_model.encode(chunks).tolist()

    embedding_time = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Converts memory to mb
    embedding_memory = peak / 10**6  
    return embeddings, embedding_time, embedding_memory

def store_in_redis(chunks, embeddings):
    """Stores text chunks and corresponding vector embeddings in Redis."""
    tracemalloc.start()

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

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    db_memory = peak / 10**6
    return db_memory


def store_in_chroma(chunks, embeddings):
    """Stores text chunks and corresponding vector embeddings in ChromaDB."""
    tracemalloc.start()

    # Create or get existing collection
    collection_name = "text_chunks"
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

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    db_memory = peak / 10**6
    return db_memory


def store_in_faiss(chunks, embeddings):
    """Stores text chunks and corresponding vector embeddings in FAISS."""
    tracemalloc.start()

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

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    db_memory = peak / 10**6  # Convert to MB
    return db_memory


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

    start_time = time.time()
    # Retrieve relevant text chunks
    relevant_chunks = retrieve_relevant_chunks(query, embedding_model, db, top_k)
    db_indexing_time = time.time() - start_time

    if not relevant_chunks:
        return "I couldn't find relevant information in the database.", db_indexing_time

    # Construct prompt with retrieved context
    context = "\n".join(relevant_chunks)
    prompt = f"Use the following lecture notes to answer the question:\n\n{context}\n\nQuestion: Answer shortly. {query}\nAnswer:"

    # Query Ollama’s Llama3.2:3B model
    response = ollama.chat(model=llm, messages=[{"role": "user", "content": prompt}])

    return response["message"]["content"], db_indexing_time


def run_llm(k=3, db="redis", embedding_model=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"), llm='llama3.2:3b'):
    print(f"\n{75 * '-'}")

    print(f"Ollama Model: {llm}")
    while True:
        query = input("Input query here ('quit' to exit): ")
        print(f"{75 * '-'}")

        if query == 'quit':
            break

        response = generate_llama_response(query, embedding_model, db, top_k=k, llm=llm)[0]
        print("\nLlama3's Response:\n", response)
        print(f"\n{75 * '-'}")


def test_llm(embedding_model, questions, db, k, llm):

    print(f"\n{75 * '-'}")
    print(f"Ollama Model: {llm}")

    responses = []
    retrieval_times = []  

    for q in questions:
        print(f"{75 * '-'}")
        response = generate_llama_response(q, embedding_model, db, top_k=k, llm=llm)
        print(response[0])
        retrieval_times.append(response[1]) 
        responses.append(response[0])

    db_indexing_time = sum(retrieval_times)
    return responses, db_indexing_time


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
        k = 3
        chunk_sizes = [300, 500]
        overlaps = [25, 50]
        sentence_transformers = [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "BAAI/bge-small-en-v1.5"
        ]
        databases = ["redis", "chroma", "faiss"]
        llms = ['llama3.2:3b', 'gemma3:1b']

        # Read and preprocess the text
        text = read_and_preprocess("data/document_1.txt")

        df = pd.DataFrame(columns=['total_time', 'llm', 'db', 'sentence_transformer', 'chunk_size', 'overlaps',
                           'embedding_time', 'embedding_memory', 'db_indexing_time', 'db_memory', 
                           'q1', 'q2', 'q3', 'q4', 'q5'])

        all_responses = []

        # Iterate over different configurations
        for c in chunk_sizes:
            for o in overlaps:
                chunks = chunk_text(text, chunk_size=c, overlap=o)
                for s in sentence_transformers:
                    embedding_model = SentenceTransformer(s)
                    
                    embeddings, embedding_time, embedding_memory = generate_embeddings(chunks, embedding_model)

                    for llm in llms:
                        for db in databases:
                            if db == "redis":
                                reset_redis_database()
                                db_memory = store_in_redis(chunks, embeddings)
                            elif db == "chroma":
                                reset_chroma_database()
                                db_memory = store_in_chroma(chunks, embeddings)
                            elif db == "faiss":
                                reset_faiss_database(s)
                                db_memory = store_in_faiss(chunks, embeddings)

                            print(f"\nDocument has been processed and stored in {db}.")

                            start_time = time.time()
                            responses, db_indexing_time = test_llm(embedding_model, questions, db, k, llm)
                            total_time = time.time() - start_time

                            responses.insert(0, total_time)
                            responses.insert(1, llm)
                            responses.insert(2, db)
                            responses.insert(3, s)
                            responses.insert(4, c)
                            responses.insert(5, o)
                            responses.insert(6, embedding_time)
                            responses.insert(7, embedding_memory)
                            responses.insert(8, db_indexing_time)
                            responses.insert(9, db_memory)

                            all_responses.append(responses)

        df = pd.DataFrame(all_responses, columns=['total_time', 'llm', 'db', 'sentence_transformer', 'chunk_size', 'overlaps',
                           'embedding_time', 'embedding_memory', 'db_indexing_time', 'db_memory', 
                           'q1', 'q2', 'q3', 'q4', 'q5'])

        df.to_csv("all_responses.csv")

    else:

        # Parameters
        k = 3
        chunk_size = 300
        overlap = 50

        # Read and preprocess the text
        text = read_and_preprocess("data/document_1.txt")

        # Split text into chunks and embedd
        transformer = "sentence-transformers/all-MiniLM-L6-v2"
        embedding_model = SentenceTransformer(transformer)
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        embeddings = generate_embeddings(chunks, embedding_model=embedding_model)[0]

        # Store the chunks and embeddings in Chroma
        reset_faiss_database(transformer)
        memory = store_in_faiss(chunks, embeddings)

        # Running LLM
        llm = 'gemma3:1b'
        run_llm(k=k, db="faiss", embedding_model=embedding_model, llm=llm)

if __name__ == "__main__":
    main()