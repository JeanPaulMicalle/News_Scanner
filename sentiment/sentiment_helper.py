from transformers import pipeline
from config.configLoader import load_config

config = load_config()
model = config["distilBERT_model"]
_sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model
)

def get_sentiment(text: str):
    """
    Run DistilBERT‐SST2 over the first 512 tokens of `text` and return:
       label (str): "POSITIVE" or "NEGATIVE"
       score (float): confidence between 0.0 and 1.0

    If `text` is longer than 512 tokens, this simply truncates:
       text[:512]
    """
    result = _sentiment_pipeline(text[:512])
    r = result[0]
    return r["label"], r["score"]
