import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import CrossEncoder
from src.utils import RESULTS_DIR

class Reranker:
    def __init__(self, model_name="cross-encoder/msmarco-MiniLM-L-6-v2", device="cuda"):
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, documents: pd.DataFrame, top_k: int = 3):
        pairs = [[query, doc] for doc in documents["text"].tolist()]
        scores = self.model.predict(pairs)
        reranked = sorted(zip(documents["text"], scores), key=lambda x: x[1], reverse=True)
        return reranked[:top_k]
