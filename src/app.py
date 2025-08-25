from .rag_pipeline import RAGPipeline

def main():
    pipe = RAGPipeline()
    print("💬 RAG Chat (type 'exit' to quit)")
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        out = pipe.answer(q, top_k_retrieve=10, top_k_rerank=5)
        print("\nAssistant:", out["answer"])
        print("\nSources:")
        for c in out["contexts"]:
            print(f"• {c['doc_id']}: {c['text'][:120]}...")

if __name__ == "__main__":
    main()
