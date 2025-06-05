from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import json
from config.configLoader import load_config
from vector_database.faissHandler import search_faiss_index, create_faiss_index, add_embeddings_to_index
import os

config = load_config()

def generate_embeddings(text_chunks, batch_size=32):
    """
    Generate embeddings for a list of text chunks in batches.
    
    Args:
        text_chunks (list): List of text chunks (strings).
        batch_size (int): Number of text chunks to process in one batch.
    
    Returns:
        np.ndarray: Array of embeddings, shape = (len(text_chunks), embedding_dim).
    """
    model = SentenceTransformer(config["embedding_model"])
    embeddings = []
    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i + batch_size]
        embeddings.extend(model.encode(batch, show_progress_bar=True))
    return np.array(embeddings, dtype="float32")


def find_relevant_chunks_with_faiss(query, json_paths, embedding_dim, top_k=5, metric=faiss.METRIC_L2):
    """
    Find the most relevant articleBody chunks using FAISS for a given query across multiple JSON files.
    Each JSON file must include these keys: "articleBody" (text) and "url" (source URL).

    Args:
        query (str): The search query / user claim.
        json_paths (list[str]): List of file paths to .json files. Each JSON should have:
                               {
                                 "articleBody": "<full article text…>",
                                 "url": "https://…"
                               }
        embedding_dim (int): The dimension of the embeddings (must match SentenceTransformer output size).
        top_k (int): Number of top chunks to return.
        metric (int): Distance metric for FAISS (faiss.METRIC_L2 or faiss.METRIC_IP).

    Returns:
        list[dict]: The top‐k relevant chunks, each dict containing:
                    {
                      "text":     <articleBody string>,
                      "filename": <basename of JSON file>,
                      "url":      <URL string from inside that JSON>
                    }
    """
    # Read each JSON file, collect its "articleBody" and "url"
    chunks = []
    for path in json_paths:
        if not path.lower().endswith(".json"):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARNING] Could not read JSON '{path}': {e}")
            continue

        body = data.get("articleBody", "").strip()
        url  = data.get("url", "").strip()
        if not body or not url:
            # Skip if either field is missing or empty
            continue

        filename = os.path.basename(path)
        chunks.append({
            "text":     body,
            "filename": filename,
            "url":      url
        })

    if not chunks:
        raise RuntimeError("No valid JSON articles with 'articleBody' and 'url' found in the provided paths.")

    # Build embeddings for all article bodies
    texts = [chunk["text"] for chunk in chunks]
    embeddings = generate_embeddings(texts) 

    # Create a FAISS index (L2 or IP) and add embeddings
    index = create_faiss_index(embedding_dim, metric=metric)
    add_embeddings_to_index(index, embeddings)

    # Embed the query string
    query_embedding = generate_embeddings([query])[0]  

    # Search the FAISS index for top_k nearest neighbors
    distances, indices = search_faiss_index(index, query_embedding, top_k=top_k)

    # Map the indices back to the chunk dictionaries
    valid_indices = [i for i in indices if 0 <= i < len(chunks)]
    return [chunks[i] for i in valid_indices]
