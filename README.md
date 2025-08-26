# rag-chatbot
💬 RAG Chatbot – Retrieval-Augmented Generation Chatbot

🧠 Project Overview

This project implements a Retrieval-Augmented Generation (RAG) Chatbot using Python, FastAPI, and Transformer-based models. It enables interactive question-answering over custom datasets by retrieving relevant context and generating natural language responses. The system can be extended to integrate with PDFs, documents, or external APIs like Bing Search, making it versatile for knowledge-based applications.

The chatbot uses sentence embeddings, retriever, reranker, and generator modules for accurate and context-aware answers. It supports GPU acceleration for faster inference and can be used as a research prototype, educational tool, or enterprise knowledge assistant.

📁 Project Structure

rag-chatbot/
├── data/
│   ├── embeddings/                  # Precomputed embeddings (not included in repo)
│   └── msmarco/                     # Datasets (not included in repo)
├── notebooks/
│   ├── 01_data_preprocessing.ipynb # Data cleaning and preprocessing
│   ├── 02_model_training.ipynb      # Retriever model training and embedding generation
│   ├── 03_reranker_training.ipynb   # Reranker model training
│   └── 04_demo_chatbot.ipynb        # Demo and testing of the chatbot
├── results/                         # Generated embeddings, indices, and outputs (not included)
├── src/
│   ├── __init__.py
│   ├── app.py                       # Main FastAPI app
│   ├── data.py                      # Data loading and preprocessing utilities
│   ├── generator.py                 # Text generation module
│   ├── gradio_app.py                # Gradio interface for quick demo
│   ├── rag_pipeline.py              # RAG pipeline implementation
│   ├── reranker.py                  # Context reranking module
│   ├── retriever.py                 # Document retriever module
│   └── utils.py                     # Helper functions
├── api/
│   └── rag_chat_api.py              # FastAPI endpoints
├── static/
│   └── style.css                    # Frontend CSS styling
├── .gitignore                        # Git ignore rules for data/results/cache
├── README.md                         # Project documentation
├── LICENSE                           # License file
└── requirements.txt                  # Python dependencies


📦 Requirements

Install dependencies using:

pip install -r requirements.txt


Key Libraries:

fastapi, uvicorn – API server

torch, tensorflow – Model training and inference

transformers, sentence-transformers – NLP models

faiss-cpu – Efficient similarity search

gradio – Demo interface

pandas, numpy – Data manipulation

🚀 How to Run

Clone the repository

git clone https://github.com/JeraldDavidRaj/rag-chatbot.git
cd rag-chatbot


Prepare data

Place your dataset CSV files in data/msmarco/ or data/embeddings/ for precomputed embeddings.

Run notebooks

01_data_preprocessing.ipynb – Preprocess data and generate embeddings.

02_model_training.ipynb – Train the retriever model.

03_reranker_training.ipynb – Train the reranker for improved context selection.

04_demo_chatbot.ipynb – Test the chatbot and interact with it.

Start the FastAPI server

uvicorn api.rag_chat_api:app --reload


Open the chat interface

http://127.0.0.1:8000


📊 Features & Highlights

Contextual question answering using RAG pipelines

Supports local embeddings for fast retrieval

Interactive chatbot interface via FastAPI & Gradio

GPU acceleration for high-performance inference

Easily extensible to PDFs, documents, or external APIs

Modular architecture: retriever, reranker, and generator

❗ Notes

data/ and results/ are not included due to size constraints. Add manually if needed.

For optimal performance, GPU is recommended.

For Bing API integration, add environment variables:

BING_API_KEY=YOUR_BING_API_KEY
BING_ENDPOINT=https://api.bing.microsoft.com/v7.0/search


📌 Future Improvements

Add support for multilingual datasets

Integrate document/PDF parsing for richer knowledge bases

Deploy as a cloud API or chatbot web app

Explore advanced NLP models like GPT-style generative models

🤝 Contribution

Pull requests, issues, and suggestions are welcome!

📜 License

This project is open-source under the MIT License.

📬 Contact

Jerald David Raj

📧 jerald7318@gmail.com

🌐 GitHub: @JeraldDavidRaj