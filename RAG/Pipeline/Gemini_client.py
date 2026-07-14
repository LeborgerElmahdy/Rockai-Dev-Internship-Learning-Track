from RAG.Pipeline.Retry_Handler import call_function_with_handling
from google import genai
from dotenv import load_dotenv
import numpy as np

load_dotenv()
client = genai.Client()

# For semantic chunking
def embed_normalized(texts: list[str], model = "gemini-embedding-001", config=None):
    result = call_function_with_handling(
        client.models.embed_content,
        model=model,
        contents=texts,
        config=config or {"output_dimensionality": 768},
    )
    vecs = np.array([e.values for e in result.embeddings])
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True) # Normalized vector helps with cosine similarity.

# Final embedding step before VectorDB
def embed(texts: list[str], model = "gemini-embedding-2", config=None):
    return call_function_with_handling(
        client.models.embed_content,
        model=model,
        contents=texts,
        config=config or {"output_dimensionality": 768},
    )

# General api call for the LLM models
def generate(prompt: str, model="gemini-2.5-flash", config=None):
    """
    config example: {
        "system_instruction": "You are a helpful assistant.",
        "temperature": 0.7,
        "max_output_tokens": 1024,
    }
    """
    return call_function_with_handling(
        client.models.generate_content,
        model=model,
        contents=prompt,
        config=config,
    )



