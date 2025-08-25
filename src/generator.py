from __future__ import annotations
from typing import Iterable
import os

# Default: local HF model (no API key required)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Optional: OpenAI if you want (set GENERATOR_BACKEND=openai and OPENAI_API_KEY)
USE_OPENAI = os.environ.get("GENERATOR_BACKEND", "").lower() == "openai"


class Generator:
    """
    Simple text generator with an optional 'stream' method.
    - Local default: google/flan-t5-base
    - Optional OpenAI if env set
    """
    def __init__(self, model_name: str = "google/flan-t5-base"):
        self.model_name = model_name
        if USE_OPENAI:
            from openai import OpenAI
            self.client = OpenAI()
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        if USE_OPENAI:
            # Non-stream, simple call
            resp = self.client.chat.completions.create(
                model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            return resp.choices[0].message.content.strip()
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    def stream(self, prompt: str, max_new_tokens: int = 256) -> Iterable[str]:
        """
        Yields chunks progressively. If using OpenAI, you could switch to streamed API.
        For local HF, we 'fake-stream' by chunking the final text.
        """
        text = self.generate(prompt, max_new_tokens=max_new_tokens)
        chunk_size = 40
        for i in range(0, len(text), chunk_size):
            yield text[: i + chunk_size]
