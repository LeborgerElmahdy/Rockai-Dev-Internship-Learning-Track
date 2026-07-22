from RAG.Pipeline.Error_Handler import call_safe_function
from google import genai
from dotenv import load_dotenv
import numpy as np

load_dotenv()
client = genai.Client()

# Normalized vector helps with cosine similarity.
def normalize_embedding(embedding):
    print("Normalizing Semantic Embeddings...")
    vectors = np.array([e.values for e in embedding.embeddings])
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

def embed(texts: list[str], model = "gemini-embedding-001", config=None):
    print(f"[embed] {len(texts)} texts, 1 request")
    return call_safe_function(
        client.models.embed_content,
        model = model,
        contents = texts,
        config = config or {"output_dimensionality": 768},
    )

def generate_response(prompt: str, model="gemini-3-flash", config=None):
    """
    config example: {
        "system_instruction": "You are a helpful assistant.",
        "temperature": 0.7,
        "max_output_tokens": 1024,
    }
    """
    return call_safe_function(
        client.models.generate_content,
        model = model,
        contents = prompt,
        config = config,
    )