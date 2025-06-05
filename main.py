import os
import sys
import json
import faiss

from config.configLoader import load_config
from document_embedding.embeddings import generate_embeddings
from vector_database.faissHandler import (
    create_faiss_index,
    add_embeddings_to_index,
    save_faiss_index,
    load_faiss_index,
    search_faiss_index,
)
from ollama.queryHandling import generate_response
from sentiment.sentiment_helper import get_sentiment

def build_and_save_index(json_dir: str, index_path: str):
    """
    1) Read every .json in `json_dir` (expects each JSON has "articleBody" and "url").
    2) Build embeddings over those articleBody strings.
    3) Create a FAISS index and add all embeddings.
    4) Save the index to disk at `index_path`.
    Returns: (faiss_index, chunks_list)
      - chunks_list = [{"text": articleBody, "filename": filename, "url": url}, ...]
    """
    chunks = []
    for fname in sorted(os.listdir(json_dir)):
        if not fname.lower().endswith(".json"):
            continue

        full_path = os.path.join(json_dir, fname)
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARNING] Could not read '{fname}': {e}", file=sys.stderr)
            continue

        # Extract articleBody and URL from each JSON
        body = data.get("articleBody", "").strip()
        url  = data.get("url", "").strip()
        if not body or not url:
            # Skip if either field is missing or empty
            continue

        chunks.append({
            "text":     body,
            "filename": fname,
            "url":      url
        })

    if not chunks:
        raise RuntimeError(f"No JSON files with 'articleBody' + 'url' found in '{json_dir}'.")

    # Build embeddings for all article bodies in one batch
    texts = [c["text"] for c in chunks]
    print(f"⏳ Generating embeddings for {len(texts)} articles...")
    embeddings = generate_embeddings(texts)

    # Create FAISS index (L2 metric)
    dim = embeddings.shape[1]
    index = create_faiss_index(dim, metric=faiss.METRIC_L2)

    # Add embeddings to index
    index.add(embeddings)
    print(f"✅ Built FAISS index with {len(texts)} vectors (dimension={dim}).")

    # Save to disk
    save_faiss_index(index, index_path)
    print(f"✅ Saved FAISS index to '{index_path}'.")
    return index, chunks


def load_or_build_index(json_dir: str, index_path: str):
    """
    If `index_path` exists, load it. Otherwise, build a new index from JSON files
    in `json_dir` and save it. Always return (index, chunks_list).
    Each chunk now includes "text", "filename", and "url".
    """
    if os.path.exists(index_path):
        # Load the FAISS index from disk
        index = load_faiss_index(index_path)
        print(f"🔍 Loaded existing FAISS index from '{index_path}'.")

        # Rebuild the chunks list (so we know which text+filename+url correspond to each vector)
        chunks = []
        for fname in sorted(os.listdir(json_dir)):
            if not fname.lower().endswith(".json"):
                continue
            full_path = os.path.join(json_dir, fname)
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"[WARNING] Could not read '{fname}': {e}", file=sys.stderr)
                continue

            body = data.get("articleBody", "").strip()
            url  = data.get("url", "").strip()
            if not body or not url:
                continue

            chunks.append({
                "text":     body,
                "filename": fname,
                "url":      url
            })
        return index, chunks

    else:
        # Build a fresh index if none exists
        return build_and_save_index(json_dir, index_path)

# Load configuration
config = load_config()
json_dir   = config["json_output_directory"]      
index_path = config["faiss_index_path"]
top_k      = config["top_k_results"]

# Verify json_dir exists
if not os.path.isdir(json_dir):
    print(f"[ERROR] JSON directory '{json_dir}' does not exist.", file=sys.stderr)
    sys.exit(1)

# Load or build the FAISS index (and retrieve the `chunks` list)
try:
    index, chunks = load_or_build_index(json_dir, index_path)
except Exception as e:
    print(f"[ERROR] Failed to build/load FAISS index: {e}", file=sys.stderr)
    sys.exit(1)

# Enter an interactive console loop
print("\n--- News Fact-Check Console ---")
print("Paste (or type) a piece of text to check if it’s misleading.")
print("Type 'exit' to quit.\n")

while True:
    user_input = input(">> ").strip()
    if user_input.lower() == "exit":
        print("Goodbye.")
        break
    if not user_input:
        continue

    submitted_text = user_input

    # Embed the user’s submitted text
    try:
        query_embedding = generate_embeddings([submitted_text])[0]
    except Exception as e:
        print(f"[ERROR] Could not generate embedding: {e}", file=sys.stderr)
        continue

    # Search the FAISS index for top_k similar articles
    try:
        distances, indices = search_faiss_index(index, query_embedding, top_k=top_k)
    except Exception as e:
        print(f"[ERROR] FAISS search failed: {e}", file=sys.stderr)
        continue

    valid_indices = [i for i in indices if i < len(chunks)]
    if not valid_indices:
        print("❗ No relevant news articles found. Cannot fact-check.")
        continue

    relevant_chunks = [chunks[i] for i in valid_indices]

    # Call your existing Ollama-based generator
    try:
        label, score = get_sentiment(submitted_text)
        print(f"[DistilBERT] Sentiment: {label}  (confidence {score:.2f})\n")

        response_text = generate_response(submitted_text, relevant_chunks, index, True)
    except Exception as e:
        print(f"[ERROR] Ollama query failed: {e}", file=sys.stderr)
        continue

    # Print the fact-check result
    print("\n--- Fact-Check Result ---")
    print(response_text)
    print("-------------------------\n")
