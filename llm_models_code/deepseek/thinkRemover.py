import re

def extract_outside_think(response):
    """
        remove everything inside <think>...</think>

    Args:
        response (str): the raw response sent back from ollama api

    Returns:
        re.sub: response with think extracted
    """

    return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()