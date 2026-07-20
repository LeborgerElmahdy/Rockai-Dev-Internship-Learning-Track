import lancedb
import hashlib
import pyarrow as pa

# Row ID will be the hash of the text stored in the chunk.
def string_to_hash_id(source_file: str, chunk_index: int, text: str) -> str:
    key = f"{source_file}::{chunk_index}::{text}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()

def get_table(persist_path: str = "./RAG/lancedb_store", table_name: str = "rag_chunks", dim: int = 768):
    db = lancedb.connect(persist_path)

    schema = pa.schema([
        pa.field("id", pa.string()),
        pa.field("text", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), dim)),
        pa.field("source_file", pa.string()),
        pa.field("block_type", pa.string()),
        pa.field("metadata", pa.string()),
    ])

    table = db.create_table(table_name, exist_ok = True, schema = schema)

    return table

def store_chunks(chunks: list, persist_path: str = "./RAG/lancedb_store", dim: int = 768):
    import json

    table = get_table(persist_path, dim=dim)

    rows = [{
        "id": string_to_hash_id(c.source_file, i, c.text),
        "text": c.text,
        "vector": list(c.vector),
        "source_file": c.source_file,
        "block_type": c.block_type,
        "metadata": json.dumps(c.metadata),
    } for i, c in enumerate(chunks)]

    (
    table.merge_insert("id")
    .when_not_matched_insert_all()   # only inserts new ids, skips existing ones entirely
    .execute(rows)
    )

    return table

def query(query_vector: list, top_k: int = 5, persist_path: str = "./RAG/lancedb_store"):
    table = get_table(persist_path)
    return table.search(query_vector).limit(top_k).to_list()

