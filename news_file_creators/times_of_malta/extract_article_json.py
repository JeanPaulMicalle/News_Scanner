import json
from bs4 import BeautifulSoup

def extract_article_json_ld(html_path: str, output_json_path: str):
    """
    Extracts and filters the JSON-LD “NewsArticle” object from a local HTML file.

    Args:
        html_path (str): Path to the local HTML file (e.g., "./data/times_of_malta/article.html") 
                         that contains the `<script type="application/ld+json">` block.
        output_json_path (str): Path where the filtered JSON should be written 
                                (e.g., "./json_output/article.json").

    Returns:
        None. On success, writes a new JSON file at `output_json_path`.

    Raises:
        RuntimeError: If no `<script type="application/ld+json">` block with a NewsArticle object 
                      is found in the HTML file.
        json.JSONDecodeError: If a found `<script>` block contains invalid JSON.
        OSError: If `html_path` cannot be opened for reading, or `output_json_path` cannot be written.
    """
    # Load & parse the HTML
    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    article_node = None

    # Find all <script type="application/ld+json"> tags, attempt to parse their JSON
    for script in soup.find_all("script", {"type": "application/ld+json"}):
        raw = script.string
        if not raw:
            continue

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Skip any script tags whose contents are not valid JSON
            continue

        # Case A: JSON-LD uses a "@graph" list
        if isinstance(data, dict) and "@graph" in data:
            for node in data["@graph"]:
                if node.get("@type") == "NewsArticle":
                    article_node = node
                    break

        # Case B (fallback): JSON-LD is directly a NewsArticle object
        if article_node is None and isinstance(data, dict) and data.get("@type") == "NewsArticle":
            article_node = data

        if article_node:
            break

    if article_node is None:
        raise RuntimeError(f"No NewsArticle JSON-LD found in {html_path!r}")

    # Filter out only the desired keys
    keys_to_keep = [
        "articleBody",
        "dateModified",
        "datePublished",
        "description",
        "headline",
        "keywords",
        "url",
        "author"
    ]
    filtered_article = {k: article_node[k] for k in keys_to_keep if k in article_node}

    # Write the filtered JSON to disk
    with open(output_json_path, "w", encoding="utf-8") as outf:
        json.dump(filtered_article, outf, ensure_ascii=False, indent=2)

