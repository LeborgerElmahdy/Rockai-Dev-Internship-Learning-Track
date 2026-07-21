from RAG.Pipeline.Parser import parse_document
from RAG.Pipeline.Chunker import chunk_blocks
from RAG.Pipeline.Gemini_client import embed, generate
from RAG.Pipeline.VectorDB import store_chunks, query, drop
from fastapi import FastAPI

#uvicorn RAG.Pipeline.Orchestrator:api --reload
api = FastAPI()

@api.post('/ingest')
def ingest_file(path: str, method: str = "semantic", rows_per_block: int = 1):
    parsed = parse_document(path)
    chunks = chunk_blocks(parsed, method)

    texts = [c.text for c in chunks]
    embedding_result = embed(texts)

    for chunk, embedding in zip(chunks, embedding_result.embeddings):
        chunk.vector = embedding.values

    store_chunks(chunks)
    return {"chunks ingested": len(chunks)}

@api.get('/query')
def ask(question: str, top_k: int = 3):
    query_vector = embed([question]).embeddings[0].values
    results = query(query_vector, top_k)
    return results

@api.post('/drop')
def drop_table(table_name: str = "rag_chunks", persist_path: str = "./RAG/lancedb_store"):
    drop(table_name, persist_path)

#api.get('/generate)
def generate_reply(question: str, top_k: int = 3):
    query_vector = embed([question]).embeddings[0].values
    results = query(query_vector, top_k)
    context = build_prompt_context(results)

    #gemini-3-flash
    #gemini-3.1-flash-lite

    response = generate(
        prompt=f"Context:\n{context}\nQuestion: {question}",
        model = "gemini-3.1-flash-lite",
        config={
            "system_instruction": (
                "You are a helpful assistant. Answer using only the provided context. "
                "If the answer isn't in the context, say you don't know."
            ),
            "temperature": 0.3,
        },
    )
    return response.text

def build_prompt_context(results: list[dict]) -> str:
    return "\n".join(
    f"Source:{r["source_file"]}\n{r["text"]}\n{r["metadata"]}"
    for r in results
    )