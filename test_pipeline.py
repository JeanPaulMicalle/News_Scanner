import os
import sys
import json
import faiss
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from config.configLoader import load_config
from document_embedding.embeddings import generate_embeddings
from vector_database.faissHandler import (
    create_faiss_index,
    add_embeddings_to_index,
    save_faiss_index,
    load_faiss_index,
    search_faiss_index,
)
from ollama.queryHandling import generate_response, query_ollama

# --- Baseline function (LLM-only) ---
def baseline_predict(prompt, model_name, ollama_url):
    """
    Query the LLM with a minimal prompt and extract 'Misleading' or 'Not misleading'.
    """
    minimal_prompt = (
        f"You are a fact-checker. Determine whether the following statement is misleading or not:\n"
        f"\"{prompt}\"\n"
        f"Answer with \"Misleading\" or \"Not misleading,\" and briefly explain."
    )
    try:
        response = query_ollama(minimal_prompt, think_remover=False,
                                model_name=model_name, ollama_url=ollama_url)
    except Exception as e:
        print(f"[Baseline] Ollama query failed: {e}", file=sys.stderr)
        return None

    # Simple extraction: look for 'Misleading' or 'Not misleading' at start
    if response.strip().startswith("Misleading"):
        return "Misleading"
    elif response.strip().startswith("Not misleading"):
        return "Not misleading"
    else:
        # Fallback: search keywords in response
        if "misleading" in response.lower():
            return "Misleading"
        else:
            return "Not misleading"

# --- Pipeline function (RAG + Sentiment) ---
def pipeline_predict(prompt, index, chunks, top_k, model_name, ollama_url):
    """
    Use the RAG + Sentiment pipeline to predict 'Misleading' or 'Not misleading'.
    """
    try:
        response = generate_response(prompt, chunks, index, think_remover=True)
    except Exception as e:
        print(f"[Pipeline] Ollama query failed: {e}", file=sys.stderr)
        return None

    if response.strip().startswith("Misleading"):
        return "Misleading"
    elif response.strip().startswith("Not misleading"):
        return "Not misleading"
    else:
        if "misleading" in response.lower():
            return "Misleading"
        else:
            return "Not misleading"

def load_or_build_index(json_dir, index_path):
    """
    Load FAISS index if it exists; otherwise build a new one from JSON files.
    Returns (index, chunks).
    """
    if os.path.exists(index_path):
        index = load_faiss_index(index_path)
        print(f"Loaded existing FAISS index from '{index_path}'")
        chunks = []
        for fname in sorted(os.listdir(json_dir)):
            if not fname.lower().endswith(".json"):
                continue
            full_path = os.path.join(json_dir, fname)
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except:
                continue
            body = data.get("articleBody", "").strip()
            url = data.get("url", "").strip()
            if not body or not url:
                continue
            chunks.append({"text": body, "filename": fname, "url": url})
        return index, chunks
    else:
        # Build new index
        chunks = []
        for fname in sorted(os.listdir(json_dir)):
            if not fname.lower().endswith(".json"):
                continue
            full_path = os.path.join(json_dir, fname)
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except:
                continue
            body = data.get("articleBody", "").strip()
            url = data.get("url", "").strip()
            if not body or not url:
                continue
            chunks.append({"text": body, "filename": fname, "url": url})

        if not chunks:
            raise RuntimeError(f"No valid JSON articles found in '{json_dir}'")

        texts = [c["text"] for c in chunks]
        embeddings = generate_embeddings(texts)
        dim = embeddings.shape[1]
        index = create_faiss_index(dim, metric=faiss.METRIC_L2)
        index.add(embeddings)
        save_faiss_index(index, index_path)
        print(f"Built and saved FAISS index to '{index_path}'")
        return index, chunks

def main(test_csv_path):
    # Load configuration
    config = load_config()
    json_dir = config["json_output_directory"]
    index_path = config.get("faiss_index_path")
    top_k = config.get("top_k_results", 5)
    ollama_model = config["ollama_model_name"]
    ollama_url = config["ollama_url"]

    # Load or build FAISS index and chunks list
    index, chunks = load_or_build_index(json_dir, index_path)

    # Load test dataset (CSV with columns: text,label)
    df = pd.read_csv(test_csv_path)
    if "text" not in df.columns or "label" not in df.columns:
        print("[ERROR] CSV must contain 'text' and 'label' columns.", file=sys.stderr)
        sys.exit(1)

    true_labels = []
    baseline_preds = []
    pipeline_preds = []

    for _, row in df.iterrows():
        prompt = row["text"]
        true = row["label"]
        true_labels.append(true)

        # Baseline prediction
        b_pred = baseline_predict(prompt, model_name=ollama_model, ollama_url=ollama_url)
        baseline_preds.append(b_pred if b_pred is not None else "Not misleading")

        # Retrieve top-k chunks for RAG context
        query_embedding = generate_embeddings([prompt])[0]
        distances, indices = search_faiss_index(index, query_embedding, top_k=top_k)
        valid_idxs = [i for i in indices if i < len(chunks)]
        relevant_chunks = [chunks[i] for i in valid_idxs]

        # Pipeline prediction
        p_pred = pipeline_predict(prompt, index, relevant_chunks, top_k,
                                  model_name=ollama_model, ollama_url=ollama_url)
        pipeline_preds.append(p_pred if p_pred is not None else "Not misleading")

    # Compute metrics for baseline
    baseline_acc = accuracy_score(true_labels, baseline_preds)
    baseline_prec, baseline_rec, baseline_f1, _ = precision_recall_fscore_support(
        true_labels, baseline_preds, labels=["Misleading", "Not misleading"], zero_division=0
    )

    # Compute metrics for pipeline
    pipe_acc = accuracy_score(true_labels, pipeline_preds)
    pipe_prec, pipe_rec, pipe_f1, _ = precision_recall_fscore_support(
        true_labels, pipeline_preds, labels=["Misleading", "Not misleading"], zero_division=0
    )

    # Print results
    print("\n=== Baseline (LLM-Only) Results ===")
    print(f"Accuracy: {baseline_acc:.2f}")
    print(f"Precision (Misleading): {baseline_prec[0]:.2f}, Recall (Misleading): {baseline_rec[0]:.2f}, F1 (Misleading): {baseline_f1[0]:.2f}")
    print(f"Precision (Not misleading): {baseline_prec[1]:.2f}, Recall (Not misleading): {baseline_rec[1]:.2f}, F1 (Not misleading): {baseline_f1[1]:.2f}")

    print("\n=== RAG+Sentiment Pipeline Results ===")
    print(f"Accuracy: {pipe_acc:.2f}")
    print(f"Precision (Misleading): {pipe_prec[0]:.2f}, Recall (Misleading): {pipe_rec[0]:.2f}, F1 (Misleading): {pipe_f1[0]:.2f}")
    print(f"Precision (Not misleading): {pipe_prec[1]:.2f}, Recall (Not misleading): {pipe_rec[1]:.2f}, F1 (Not misleading): {pipe_f1[1]:.2f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_pipeline.py <test_dataset.csv>", file=sys.stderr)
        sys.exit(1)
    test_csv = sys.argv[1]
    main(test_csv)
