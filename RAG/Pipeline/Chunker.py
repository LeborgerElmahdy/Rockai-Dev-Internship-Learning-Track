import re
import numpy as np
import tiktoken
from dotenv import load_dotenv
from google import genai
from sklearn.feature_extraction.text import TfidfVectorizer
from RAG.Pipeline import Gemini_client as gc, Parser as p
from dataclasses import dataclass, field

load_dotenv()
client = genai.Client()
encoder = tiktoken.get_encoding("cl100k_base")

MAX_TOKENS = 350
SEMANTIC_SIM_THRESHOLD = 0.6
STATISTIC_SIM_THRESHOLD = 0.15

@dataclass
class Chunk:
    text: str
    source_file: str
    block_type: str
    metadata: dict = field(default_factory=dict)
    vector: list = field(default_factory=list)

def count_tokens(text: str) -> int:
    return len(encoder.encode(text))

def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

# Takes its inputs from either chunking method, and splits based on cosine similarity.
def _chunkify(sentences: list[str], vectors: np.ndarray, threshold: float) -> list[str]:
    chunks, current_chunk, current_tokens = [], [sentences[0]], count_tokens(sentences[0])

    for i in range(1, len(sentences)):
        a, b = vectors[i -1], vectors[i]

        # Cosine Similarity is usually Dot(A,B) / ||A|| * ||B||
        # But since in both chunkers, the output is L2 normalized vectors, the denominator will always be 1 so i omitted it's calculation to save performance
        # Ill keep the original calculation commented just in case.

        # calculates the ||A|| * ||B|| (denominator)
        #denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
        #similarity = np.dot(a,b) / denom

        similarity = np.dot(a,b)
        token_count = count_tokens(sentences[i])

        # Splits if either similarity is too low, or max token count per chunk is reached.
        if similarity < threshold or token_count + current_tokens > MAX_TOKENS:
            chunks.append("".join(current_chunk))
            current_chunk, current_tokens = [sentences[i]], token_count
        else:
            current_chunk.append(sentences[i])
            current_tokens += token_count

    chunks.append(" ".join(current_chunk))
    return chunks

#Statistic chunking via TF-IDF, cheaper since no api calls but less accurate since no semantic meaning is being extracted.
def statistic_chunk(text: str) -> list[str]:
    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return [text.strip()] if text.strip() else []

    #Apparently outputs normalized vectors by default? quite nice
    vectors = TfidfVectorizer().fit_transform(sentences).toarray()
    return _chunkify(sentences, vectors, STATISTIC_SIM_THRESHOLD)

# Gemini Embedding 1 based, costs api calls
def semantic_chunk(sentences: list[str], vectors: np.ndarray) -> list[str]:
    if len(sentences) <= 1:
        return [sentences[0].strip()] if sentences and sentences[0].strip() else []
    return _chunkify(sentences, vectors, SEMANTIC_SIM_THRESHOLD)

def chunk_blocks(blocks: list[dict], method="statistic") -> list[Chunk]:
    result = []

    if method == "semantic":
        # Split every block into sentences up front
        block_sentences = [_split_sentences(b["text"]) for b in blocks]
        flat_sentences = [s for sents in block_sentences for s in sents]

        if not flat_sentences:
            return result

        # One embedding call for the entire file — still one vector per sentence
        all_vectors = gc.embed_normalized(flat_sentences, config={"output_dimensionality": 768})

        # Slice the flat results back out per block, in order
        idx = 0
        for block, sentences in zip(blocks, block_sentences):
            if not sentences:
                continue
            vecs = all_vectors[idx: idx + len(sentences)]
            idx += len(sentences)

            for chunk_text in semantic_chunk(sentences, vecs):
                result.append(Chunk(
                    text=chunk_text,
                    source_file=block["source_file"],
                    block_type=block["block_type"],
                    metadata=block["metadata"],
                ))
    else:
        for block in blocks:
            for chunk_text in statistic_chunk(block["text"]):
                result.append(Chunk(
                    text=chunk_text,
                    source_file=block["source_file"],
                    block_type=block["block_type"],
                    metadata=block["metadata"],
                ))

    return result
# For manual testing.
# if __name__ == "__main__":
    # sample_text = (
    #     "Python is a popular programming language. It is used heavily in data science and AI. "
    #     "On a completely unrelated note, making lasagna requires pasta sheets and a great sauce. "
    #     "Bake the lasagna at 180°C for roughly forty-five minutes. "
    #     "Switching topics again, quantum computing uses qubits instead of standard bits."
    # )
    # ADD , rows_per_block = 1 FOR CSV AND XLSX FILES ONLY
    # blocks = p.parse_file("RAG/Sample Files/sample.csv")

    # print("=====================Statistic=====================")
    # for block in blocks:
    #     for chunk in statistic_chunk(block["text"]):
    #         print(f"[SEPARATOR]{chunk}\n")

    # print("=====================Semantic=====================")
    # for block in blocks:
    #     for chunk in semantic_chunk(block["text"]):
    #         print(f"[SEPARATOR]{chunk}\n ")