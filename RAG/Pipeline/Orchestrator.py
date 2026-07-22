from RAG.Pipeline.Parser import parse_document, string_to_hash_id
from RAG.Pipeline.Chunker import chunk_block, ChunkMethod
from RAG.Pipeline.Gemini_client import embed, generate_response
from RAG.Pipeline.VectorDB import store_chunks, query, drop
from fastapi import FastAPI

#uvicorn RAG.Pipeline.Orchestrator:api --reload
api = FastAPI()

def build_chunks(block: dict, method: ChunkMethod) -> list[dict]:
    "Chunk a parsed block's text and return chunk dicts matching parse_document's structure."
    text_chunks = chunk_block(block, method)

    output_chunks = []
    for i, chunk_text in enumerate(text_chunks):
        output_chunks.append(
            {
                "id": string_to_hash_id(block["source_file"], block["source_type"], i, chunk_text),
                "source_file": block["source_file"],
                "source_type": block["source_type"],
                "text": chunk_text,
                "vector": None,
            }
        )
    return output_chunks

@api.post('/ingest')
def ingest_file(path: str, method: str = "semantic"):
    parsed_blocks = parse_document(path)
    method = ChunkMethod(method)

    data_chunks = []
    for block in parsed_blocks:
        data_chunks += build_chunks(block, method)

    texts = [c["text"] for c in data_chunks]
    embedding_result = embed(texts)

    for chunk, embedding in zip(data_chunks, embedding_result.embeddings):
        chunk["vector"] = embedding.values

    store_chunks(data_chunks)
    return {"chunks ingested": len(data_chunks)}

@api.get('/query')
def ask(question: str, top_k: int = 3):
    query_vector = embed([question]).embeddings[0].values
    results = query(query_vector, top_k)
    return results

@api.post('/drop')
def drop_table(table_name: str = "rag_chunks", persist_path: str = "./RAG/lancedb_store"):
    drop(table_name, persist_path)

#api.get('/generate)
def generate_query_reply(question: str, top_k: int = 3):
    query_vector = embed([question]).embeddings[0].values
    results = query(query_vector, top_k)
    context = build_prompt_context(results)

    #gemini-3-flash
    #gemini-3.1-flash-lite

    response = generate_response(
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
    f"Source:{r["source_file"]}\n{r["text"]}"
    for r in results
    )