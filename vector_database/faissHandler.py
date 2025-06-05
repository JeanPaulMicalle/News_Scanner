import faiss
import numpy as np

def create_faiss_index(embedding_dim, metric=faiss.METRIC_L2):
    """
    Create a FAISS index.

    Args:
        embedding_dim (int): The dimension of the embeddings.
        metric (int): Distance metric 

    Returns:
        faiss.Index: A FAISS index instance.
    """
    return faiss.IndexFlatL2(embedding_dim) if metric == faiss.METRIC_L2 else faiss.IndexFlatIP(embedding_dim)

def add_embeddings_to_index(index, embeddings):
    """
    Add embeddings to a FAISS index.

    Args:
        index (faiss.Index): The FAISS index instance.
        embeddings (np.ndarray): Array of embeddings.
    """
    index.add(embeddings)

def save_faiss_index(index, save_path):
    """
    Save a FAISS index to a file.

    Args:
        index (faiss.Index): The FAISS index instance.
        save_path (str): Path to save the index.
    """
    faiss.write_index(index, save_path)

def load_faiss_index(load_path):
    """
    Load a FAISS index from a file.

    Args:
        load_path (str): Path to the saved index.

    Returns:
        faiss.Index: The loaded FAISS index instance.
    """
    return faiss.read_index(load_path)

def search_faiss_index(index, query_embedding, top_k=5):
    """
    Search for the most similar embeddings in a FAISS index.

    Args:
        index (faiss.Index): The FAISS index instance.
        query_embedding (np.ndarray): The query embedding.
        top_k (int): Number of top results to return.

    Returns:
        tuple: Distances and indices of the top results.
    """
    query_embedding = np.array([query_embedding], dtype="float32")
    distances, indices = index.search(query_embedding, top_k)
    return distances[0], indices[0]
