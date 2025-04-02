# RAG LLM Pipeline

This project implements a Retrieval-Augmented Generation (RAG) pipeline using various vector databases and local LLMs (via Ollama). The pipeline allows you to chunk, embed, and store text in a vector database, and then retrieve relevant chunks to answer questions using a local language model.

## Features
- Uses Sentence Transformers to generate embeddings.
- Stores embeddings in Redis, ChromaDB, or FAISS.
- Retrieves top-k relevant chunks to construct prompts.
- Runs LLMs locally via Ollama (`llama3.2:3b`, `gemma3:1b`).
- Supports two modes: interactive question-answering and full benchmarking.

## Getting Started

### 1. Install Dependencies

Ensure you have Python 3.9+ and run:
```bash
pip install -r requirements.txt
```

### 2. Set Up Local Services

- Redis: Make sure Redis is running locally on port `6379`.
- Ollama: Install and run [Ollama](https://ollama.com) and pull the models you intend to use:
  ```bash
  ollama pull llama3.2:3b
  ollama pull gemma3:1b
  ```

## How to Run

### Step 1: Prepare Input Text

Place your input text file at:

```
data/document_1.txt
```

This file will be read, preprocessed, chunked, and embedded.

### Step 2: Run the Script

From the root directory, run:

```bash
python rag_llm.py
```

You’ll be prompted with:

```
Enter 0 to run model. Enter 1 to test different model versions:
```

- Enter `0` to enter interactive QA mode using FAISS by default.
- Enter `1` to run a full benchmark across:
  - Different chunk sizes and overlaps
  - Embedding models
  - Vector databases (`redis`, `chroma`, `faiss`)
  - LLMs via Ollama (`llama3.2:3b`, `gemma3:1b`)

Benchmark results will be saved to:

```
all_responses.csv
```

## Example: Interactive Mode

After entering `0`, you'll see:

```
Input query here ('quit' to exit):
```

You can type any question related to the document content. The script will:
1. Retrieve top-k relevant chunks from FAISS
2. Construct a prompt with that context
3. Generate a response using the specified Ollama model

## Output

- `all_responses.csv`: Stores benchmark results when using test mode
- Console Output: Real-time responses and timing data

## Requirements

See `requirements.txt` for all necessary Python packages.
