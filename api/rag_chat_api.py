# api/rag_chat_api.py

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline

# ---------------------------
# Project Paths & App Setup
# ---------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

app = FastAPI(title="RAG Chatbot API")

# Mount static folder for CSS
app.mount("/static", StaticFiles(directory=PROJECT_ROOT / "static"), name="static")

# ---------------------------
# Load Retriever Data
# ---------------------------
# Load FAISS IVF index
index_path = RESULTS_DIR / "faiss_index_ivf.bin"
index = faiss.read_index(str(index_path))
index.nprobe = 10  # tune for speed vs accuracy

# Load document metadata
df = pd.read_csv(RESULTS_DIR / "retriever_documents.csv")

# Embedding model on GPU if available
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")

# Optional reranker
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Generator model (stronger generation)
generator = pipeline("text-generation", model="gpt2-medium", device=0)  # device=0 for GPU

# ---------------------------
# RAG Chat Function
# ---------------------------
def rag_chat(query, top_k=3):
    # Encode query
    query_emb = embedder.encode([query], convert_to_numpy=True)

    # Retrieve top_k documents from FAISS
    distances, indices = index.search(query_emb, top_k)
    retrieved_docs = df.iloc[indices[0]]

    # Optional reranking
    pairs = [[query, doc] for doc in retrieved_docs["text"].tolist()]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(retrieved_docs["text"], scores), key=lambda x: x[1], reverse=True)

    # Prepare context
    context = "\n".join([doc for doc, _ in reranked[:top_k]])

    # Generate answer
    input_text = f"Answer the question using the following context:\n{context}\n\nQuestion: {query}\nAnswer:"
    response = generator(input_text, max_length=300, num_return_sequences=1)[0]["generated_text"]

    return response

# ---------------------------
# HTML Chat Interface
# ---------------------------
CHAT_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>RAG Chatbot</title>
<link rel="stylesheet" href="/static/style.css">
</head>
<body>
<div class="chat-container">
    <h2>RAG Chatbot</h2>
    <form method="post">
        <input type="text" name="query" placeholder="Type your question..." required>
        <button type="submit">Send</button>
    </form>
    <div id="response">
        {% if response %}
        <p><strong>Bot:</strong> {{ response }}</p>
        {% endif %}
    </div>
</div>
</body>
</html>
"""

# ---------------------------
# Routes
# ---------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return CHAT_HTML.replace("{% if response %}", "").replace("{% endif %}", "")

@app.post("/", response_class=HTMLResponse)
def chat(query: str = Form(...)):
    response_text = rag_chat(query)
    html_response = CHAT_HTML.replace("{{ response }}", response_text).replace("{% if response %}", "").replace("{% endif %}", "")
    return html_response




