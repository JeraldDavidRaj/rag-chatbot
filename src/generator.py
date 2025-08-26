import torch
from transformers import pipeline

class Generator:
    def __init__(self, model_name="gpt2"):
        device_id = 0 if torch.cuda.is_available() else -1
        self.generator = pipeline("text-generation", model=model_name, device=device_id)

    def generate(self, query: str, context: str, max_length: int = 200):
        input_text = (
            f"Answer the question using the following context:\n{context}\n\n"
            f"Question: {query}\nAnswer:"
        )
        response = self.generator(
            input_text, max_length=max_length, num_return_sequences=1
        )[0]["generated_text"]
        return response

