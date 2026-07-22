import re
import tiktoken
import numpy as np
from enum import Enum
from sklearn.feature_extraction.text import TfidfVectorizer
from RAG.Pipeline.Gemini_client import embed, normalize_embedding

encoder = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(encoder.encode(text))

def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

class ChunkMethod(Enum):
    NAIVE = "naive"
    STATISTICAL = "statistical"
    SEMANTIC = "semantic"

# Takes input text and vectors (whatever they may be) and performs cosine similary on them and then chunks them.
def _chunkify(sentences: list[str], vectors: np.ndarray, threshold: float,
              Max_Tokens: int = 400, Min_Tokens: int = 100) -> list[str]:
    chunks = []
    current_chunk = [sentences[0]]
    cumulative_token_count = count_tokens(sentences[0])

    for i in range(1, len(sentences)):
        a, b = vectors[i - 1], vectors[i]
        sim = np.dot(a, b)
        sentence_token_count = count_tokens(sentences[i])

        should_break = sim < threshold or cumulative_token_count + sentence_token_count > Max_Tokens
        below_min = cumulative_token_count < Min_Tokens

        if should_break and not below_min:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
            cumulative_token_count = sentence_token_count
        else:
            current_chunk.append(sentences[i])
            cumulative_token_count += sentence_token_count

    chunks.append(" ".join(current_chunk))
    return chunks

def chunk_naive(text: str, chunk_size: int, overlap: int) -> list[str]:
    words  = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start: end]))
        start += chunk_size - overlap
    return chunks

def chunk_statistic(text: str, threshold: float = 0.6, min_tokens: int = 100, max_tokens: int = 400) -> list[str]:
    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return [text.strip()] if text.strip() else []
    
    vectors = TfidfVectorizer().fit_transform(sentences).toarray()
    return _chunkify(sentences, vectors, threshold, Max_Tokens = max_tokens, Min_Tokens = min_tokens)

def chunk_semantic(text: str, threshold: float = 0.6, min_tokens: int = 100, max_tokens: int = 400) -> list[str]:
    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return [text.strip()] if text.strip() else []

    response = embed(sentences)
    vectors = normalize_embedding(response)
    return _chunkify(sentences, vectors, threshold, Max_Tokens=max_tokens, Min_Tokens=min_tokens)

def chunk_block(block: dict, strategy: ChunkMethod) -> list[str]:
    if block["source_type"] == ".csv":
        return [block["text"]]

    match strategy:
        case ChunkMethod.NAIVE:
            return chunk_naive(block["text"], chunk_size=50, overlap=5)
        case ChunkMethod.STATISTICAL:
            return chunk_statistic(block["text"])
        case ChunkMethod.SEMANTIC:
            return chunk_semantic(block["text"])
        case _:
            raise ValueError(f"Unknown strategy: {strategy}")

