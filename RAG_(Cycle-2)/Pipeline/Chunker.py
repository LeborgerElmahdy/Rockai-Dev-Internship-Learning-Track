import re
import numpy as np
import tiktoken
from google import genai
from sklearn.feature_extraction.text import TfidfVectorizer

encoder = tiktoken.get_encoding("cl100k_base")

MAX_TOKENS = 500
SIM_THRESHOLD = 0.15

#Statistic chunking via TF-IDF, cheaper on api calls but less accurate since no semantic meaning is being extracted.
def statistic_chunk(text: str) -> list[str]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if len(sentences) <= 1:
        return [text.strip()] if text.strip() else []

    vectors = TfidfVectorizer().fit_transform(sentences).toarray()

    chunks, current, current_tokens = [], [sentences[0]], len(encoder.encode(sentences[0]))

    for i in range(1, len(sentences)):
        a, b = vectors[i - 1], vectors[i]
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
        sim = np.dot(a, b) / denom
        tokens = len(encoder.encode(sentences[i]))

        if sim < SIM_THRESHOLD or current_tokens + tokens > MAX_TOKENS:
            chunks.append(" ".join(current))
            current, current_tokens = [sentences[i]], tokens
        else:
            current.append(sentences[i])
            current_tokens += tokens

    chunks.append(" ".join(current))
    return chunks

if __name__ == "__main__":
    sample_text = (
        "Python is a popular programming language. It is used heavily in data science and AI. "
        "On a completely unrelated note, making lasagna requires pasta sheets and a great sauce. "
        "Bake the lasagna at 180°C for roughly forty-five minutes. "
        "Switching topics again, quantum computing uses qubits instead of standard bits."
    )

    print("--- Running Semantic Chunking ---")
    resulting_chunks = statistic_chunk(sample_text)

    for index, chunk in enumerate(resulting_chunks):
        print(f"\n[Chunk {index + 1}]:")
        print(chunk)