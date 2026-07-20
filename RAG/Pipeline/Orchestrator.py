from RAG.Pipeline.Parser import parse_file
from RAG.Pipeline.Chunker import chunk_blocks
from RAG.Pipeline.Gemini_client import embed
from RAG.Pipeline.VectorDB import store_chunks, query
from fastapi import FastAPI

#uvicorn RAG.Pipeline.Orchestrator:api --reload
api = FastAPI()

@api.post('/ingest')
def ingest_file(path: str, method: str = "semantic", rows_per_block: int = 1):
    parsed = parse_file(path, rows_per_block)
    chunks = chunk_blocks(parsed, method)

    texts = [c.text for c in chunks]
    embedding_result = embed(texts)

    for chunk, embedding in zip(chunks, embedding_result.embeddings):
        chunk.vector = embedding.values
        print(f"ZINGY{chunk} \n")

    store_chunks(chunks)
    return {"chunks ingested": len(chunks)}

@api.get('/query')
def ask(question: str, top_k: int = 3):
    query_vector = embed([question]).embeddings[0].values
    results = query(query_vector, top_k)
    return results