from __future__ import annotations
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

class Reranker:
    """
    Cross-encoder reranker for (query, doc_text) pairs.
    """
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs)
        for i, s in enumerate(scores):
            candidates[i]["rerank_score"] = float(s)
        return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]

