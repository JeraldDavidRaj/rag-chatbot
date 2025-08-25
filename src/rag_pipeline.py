from __future__ import annotations
from typing import Dict, Any, List
from pathlib import Path

from .data import load_documents
from .retriever import Retriever
from .reranker import Reranker
from .generator import Generator
from .utils import DATA_DIR, RESULTS_DIR

class RAGPipeline:
    """
    Orchestrates retrieve → rerank → generate with citations.
    """
    def __init__(self,
                 documents_csv: str | Path = DATA_DIR / "documents.csv",
                 retriever_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 generator_model: str = "google/flan-t5-base"):

        self.docs = load_documents(documents_csv)
        self.retriever = Retriever(
            model_name=retriever_model,
            embeddings_path=RESULTS_DIR / "doc_embeddings.npy",
            ids_path=RESULTS_DIR / "doc_ids.npy",
            faiss_index_path=RESULTS_DIR / "retriever.index",
        )
        # Load prebuilt embeddings/index (build them in Notebook 02)
        self.retriever.load()

        self.reranker = Reranker(model_name=reranker_model)
        self.generator = Generator(model_name=generator_model)

    def _make_prompt(self, query: str, contexts: List[str]) -> str:
        context = "\n\n".join([f"- {c}" for c in contexts])
        return (
            "You are a helpful assistant. Answer strictly using the context.\n\n"
            f"Question:\n{query}\n\n"
            f"Context:\n{context}\n\n"
            "Answer in 3–6 sentences, cite nothing beyond the provided context."
        )

    def answer(self, query: str, top_k_retrieve: int = 12, top_k_rerank: int = 5, stream: bool = False) -> Dict[str, Any]:
        # 1) Retrieve (ids + row indices + scores)
        retrieved = self.retriever.search(query, top_k=top_k_retrieve)

        # 2) Build candidate texts
        candidates = []
        for r in retrieved:
            row = self.docs.iloc[r["doc_row_index"]]
            candidates.append({
                "doc_id": str(r["doc_id"]),
                "text": str(row["text"]),
                "ret_score": float(r["score"]),
            })

        # 3) Rerank
        reranked = self.reranker.rerank(query, candidates, top_k=top_k_rerank)

        # 4) Generate
        contexts = [c["text"] for c in reranked]
        prompt = self._make_prompt(query, contexts)

        if stream:
            return {
                "stream": self.generator.stream(prompt, max_new_tokens=256),
                "contexts": reranked
            }
        else:
            out = self.generator.generate(prompt, max_new_tokens=256)
            return {"answer": out, "contexts": reranked}


