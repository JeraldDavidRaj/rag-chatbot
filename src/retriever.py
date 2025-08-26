import faiss
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

RESULTS_DIR = Path("results")

class Retriever:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device="cuda"):
        self.model = SentenceTransformer(model_name, device=device)
        self.index = None
        self.df = None

    def build_index(self, documents_path: str, embeddings_path: str, nlist: int = 100):
        # Load documents
        self.df = pd.read_csv(documents_path)
        
        # Load embeddings
        embeddings = np.load(embeddings_path, mmap_mode="r")
        dimension = embeddings.shape[1]

        # Create IVF index
        quantizer = faiss.IndexFlatL2(dimension)
        self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
        
        # Train index
        train_size = min(10000, len(embeddings))
        self.index.train(embeddings[:train_size])

        # Add embeddings in chunks
        chunk_size = 10000
        for i in range(0, len(embeddings), chunk_size):
            self.index.add(embeddings[i:i+chunk_size])

        print("FAISS IVF index built with", len(self.df), "documents.")

    def save_index(self, index_path="results/faiss_index_ivf.bin"):
        faiss.write_index(self.index, str(index_path))
        print(f"Index saved to {index_path}")

    def load_index(self, index_path="results/faiss_index_ivf.bin", documents_path="results/retriever_documents.csv"):
        self.index = faiss.read_index(str(index_path))
        self.df = pd.read_csv(documents_path)
        self.index.nprobe = 10  # default
        print("Index and documents loaded.")

    def search(self, query: str, top_k: int = 3):
        query_emb = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_emb, top_k)
        return self.df.iloc[indices[0]], distances[0]

