from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer


try:
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

from sklearn.metrics.pairwise import cosine_similarity


class Retriever:
    """
    Dense bi-encoder retriever with optional FAISS index.
    Saves and loads:
      - embeddings: results/doc_embeddings.npy
      - ids:        results/doc_ids.npy
      - faiss idx:  results/retriever.index (if faiss available)
    """
    def __init__(self,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 embeddings_path: Path | str = "../results/doc_embeddings.npy",
                 ids_path: Path | str = "../results/doc_ids.npy",
                 faiss_index_path: Path | str = "../results/retriever.index"):
        self.model = SentenceTransformer(model_name)
        self.embeddings_path = Path(embeddings_path)
        self.ids_path = Path(ids_path)
        self.faiss_index_path = Path(faiss_index_path)

        self.embeddings: np.ndarray | None = None
        self.doc_ids: np.ndarray | None = None
        self.index = None  # FAISS index if used

    def encode_docs(self, texts: List[str], batch_size: int = 256) -> np.ndarray:
        return self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True)

    def build_and_save(self, doc_texts: List[str], doc_ids: List[str | int]):
        self.embeddings = self.encode_docs(doc_texts)
        self.doc_ids = np.array(doc_ids)

        self.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.embeddings_path, self.embeddings)
        np.save(self.ids_path, self.doc_ids)

        if FAISS_AVAILABLE:
            dim = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)  # use inner-product with normalized embeddings
            # normalize to unit length for IP==cosine
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-12
            faiss.normalize_L2(self.embeddings)
            self.index.add(self.embeddings)
            faiss.write_index(self.index, str(self.faiss_index_path))

    def load(self):
        self.embeddings = np.load(self.embeddings_path)
        self.doc_ids = np.load(self.ids_path, allow_pickle=True)
        if FAISS_AVAILABLE and Path(self.faiss_index_path).exists():
            self.index = faiss.read_index(str(self.faiss_index_path))
        else:
            self.index = None

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        q = self.model.encode([query], convert_to_numpy=True)

        if FAISS_AVAILABLE and self.index is not None:
            # Normalize for cosine/IP search
            faiss.normalize_L2(q)
            scores, idxs = self.index.search(q, top_k)
            idxs, scores = idxs[0], scores[0]
        else:
            # Cosine similarity fallback
            sims = cosine_similarity(q, self.embeddings)[0]
            idxs = sims.argsort()[-top_k:][::-1]
            scores = sims[idxs]

        results = []
        for i, s in zip(idxs, scores):
            results.append({"doc_row_index": int(i),
                            "doc_id": str(self.doc_ids[i]),
                            "score": float(s)})
        return results
