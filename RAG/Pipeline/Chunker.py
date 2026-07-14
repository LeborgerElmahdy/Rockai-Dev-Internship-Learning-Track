import re
import numpy as np
import tiktoken
from dotenv import load_dotenv
from google import genai
from sklearn.feature_extraction.text import TfidfVectorizer
from RAG.Pipeline import Gemini_client as gc, Parser as p
load_dotenv()
client = genai.Client()
encoder = tiktoken.get_encoding("cl100k_base")

MAX_TOKENS = 500
SEMANTIC_SIM_THRESHOLD = 0.6
STATISTIC_SIM_THRESHOLD = 0.15

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
def semantic_chunk(text: str) -> list[str]:
    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return [text.strip()] if text.strip() else []
    
    #Gemini call happens here, returns embeddings with 768 dimensions
    vector_embeddings = gc.embed_normalized(sentences, config = {"output_dimensionality": 768})
    return _chunkify(sentences, vector_embeddings, SEMANTIC_SIM_THRESHOLD)


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