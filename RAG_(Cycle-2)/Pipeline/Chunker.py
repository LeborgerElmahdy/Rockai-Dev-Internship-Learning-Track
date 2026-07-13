import re
import numpy as np
import tiktoken
from sklearn.feature_extraction.text import TfidfVectorizer

encoder = tiktoken.get_encoding("cl100k_base")

MAX_TOKENS = 500
SIM_THRESHOLD = 0.15


def semantic_chunk(text: str) -> list[str]:
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