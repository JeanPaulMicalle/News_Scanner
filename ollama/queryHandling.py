import requests
import json
from document_embedding.embeddings import generate_embeddings
from vector_database.faissHandler import search_faiss_index
from config.configLoader import load_config
from llm_models_code.deepseek.thinkRemover import extract_outside_think
from sentiment.sentiment_helper import get_sentiment

config = load_config()

def query_ollama(prompt, think_remover, model_name=None, ollama_url=None):
    """
    Query the Ollama API to generate a response based on a given prompt.

    Args:
        prompt (str): The prompt to send to the Ollama model for generating a response.
        think_remover(boolean): is set to true will not display think process of deepseek
        model_name (str, optional): The model name to use for the query. If not provided, the default model name is loaded from the configuration file.
        ollama_url (str, optional): The URL of the locally hosted Ollama server. If not provided, the default URL is loaded from the configuration file.

    Returns:
        str: The generated response from the Ollama model, combining all parsed responses if multiple lines are returned.

    Raises:
        ValueError: If `model_name` is not provided and cannot be loaded from the configuration.
        Exception: If the response from Ollama cannot be parsed or if the request to the Ollama API fails.
    """
    if not model_name:
        model_name = config["ollama_model_name"]  # Default model name from config
    if not ollama_url:
        ollama_url = config["ollama_url"]  # Default Ollama URL from config

    if not model_name:
        raise ValueError("The model_name parameter cannot be empty.")

    response = requests.post(
        f"{ollama_url}/api/generate",
        json={"model": model_name, "prompt": prompt}
    )

    # Check if the response was successful
    if response.status_code == 200:
        try:
            # Handle multiple JSON lines in the response
            lines = response.text.strip().splitlines()
            combined_response = ""
            for line in lines:
                parsed_line = json.loads(line)
                combined_response += parsed_line.get("response", "")
            if (think_remover):
                combined_response = extract_outside_think(combined_response)
            else:
                print("Think not removed: ", think_remover)
            return combined_response.strip()
        except json.JSONDecodeError as e:
            print(f"Raw Response: {response.text}")
            raise Exception("Failed to parse JSON response from Ollama.") from e
    else:
        # Handle non-200 responses
        raise Exception(f"Failed to query Ollama: {response.status_code} - {response.text}")

def generate_response(query, chunks, faiss_index, think_remover):
    """
    Generate a response to `query`, using:
      - DistilBERT sentiment on the user’s prompt itself
      - FAISS to fetch relevant news chunks (each with text + url)
      - Ollama to evaluate whether the user’s prompt is misleading,
        explicitly citing URLs and now “adding extra detail” via the prompt’s sentiment.

    Args:
        query (str): The user’s full text (e.g. an article on fast foods).
        chunks (list): List of dicts, each containing:
                       {
                         "text": "<articleBody>",
                         "filename": "<somefile.json>",
                         "url": "<source URL>"
                       }
        faiss_index: A FAISS index built over embeddings of those chunks.
        think_remover (bool): If True, strip out “thinking” patterns from the Deepseek model.

    Returns:
        str: The LLM’s answer. It will begin by referencing
             the prompt’s DistilBERT sentiment, and then cite URLs for any factual claims.
    """
    model_name = config["ollama_model_name"]
    top_k     = config.get("top_k_results", 5)

    # Run DistilBERT‐SST2 on the user’s prompt to get (label, score)
    prompt_label, prompt_score = get_sentiment(query)

    # Embed the query, find top-k relevant chunks via FAISS
    query_embedding = generate_embeddings([query])[0]
    distances, indices = search_faiss_index(faiss_index, query_embedding, top_k=top_k)
    valid_indices = [i for i in indices if 0 <= i < len(chunks)]
    if not valid_indices:
        return "No relevant news articles were found to fact‐check your prompt."

    # Build context by listing each relevant chunk 
    context_sections = []
    for i in valid_indices:
        url     = chunks[i]["url"]
        excerpt = chunks[i]["text"].strip()
        # Prepend each snippet with its URL header
        section = f"===== Source: {url} =====\n{excerpt}"
        context_sections.append(section)
    context = "\n\n".join(context_sections)

    # Construct the final Ollama prompt, including DistilBERT sentiment
    prompt = (
        "You are a news‐fact‐checker. A user has provided a piece of text below, "
        "and you also have a DistilBERT‐derived sentiment label for that user text. "
        "Use the sentiment to add extra context and meaning.  \n\n"
        f"User‐prompt sentiment: {prompt_label} ({prompt_score:.2f})\n\n"
        f"Here are the top {len(valid_indices)} relevant news excerpts (each preceded by its URL):\n\n"
        f"{context}\n\n"
        "Please evaluate the user’s claim and explicitly answer “Misleading” or “Not misleading” "
        "before giving any further explanation. When you make factual claims in your explanation, "
        "you MUST cite each source by writing:\n"
        "    According to <URL>, <fact>.\n"
        "If multiple sources agree, you can combine them like “According to URL1 and URL2, ….”\n\n"
        "User’s full text / claim:\n"
        f"{query}\n\n"
        "Answer (start with “Misleading” or “Not misleading” and then explain, citing sources):"
    )

    response = query_ollama(prompt, think_remover, model_name=model_name)

    used_urls = {chunks[i]["url"] for i in valid_indices}
    citation_text = f"\n\nSources: {', '.join(used_urls)}"

    return response + citation_text