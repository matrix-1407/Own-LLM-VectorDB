from typing import List

import httpx


class OllamaClient:
    """
    Thin HTTP client wrapping local Ollama REST API.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 11434):
        self.base_url = f"http://{host}:{port}"
        self.embed_model = "nomic-embed-text"
        self.gen_model = "llama3.2"

    def is_available(self) -> bool:
        try:
            with httpx.Client(timeout=2.0) as client:
                r = client.get(f"{self.base_url}/api/tags")
                return r.status_code == 200
        except Exception:
            return False

    def embed(self, text: str) -> List[float]:
        """Returns 768-dimensional embedding vector, or [] on failure."""
        try:
            with httpx.Client(timeout=30.0) as client:
                r = client.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.embed_model, "prompt": text},
                )
                if r.status_code == 200:
                    return r.json().get("embedding", [])
        except Exception as e:
            print(f"[Ollama] embed error: {e}")
        return []

    def generate(self, prompt: str) -> str:
        """Returns generated text from llama3.2, or an error string."""
        try:
            with httpx.Client(timeout=180.0) as client:
                r = client.post(
                    f"{self.base_url}/api/generate",
                    json={"model": self.gen_model, "prompt": prompt, "stream": False},
                )
                if r.status_code == 200:
                    return r.json().get("response", "")
                return f"ERROR: Ollama returned {r.status_code}"
        except Exception as e:
            print(f"[Ollama] generate error: {e}")
            return "ERROR: Ollama unavailable. Run: ollama serve"
