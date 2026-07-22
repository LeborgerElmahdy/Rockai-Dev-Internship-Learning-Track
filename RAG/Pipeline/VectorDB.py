import lancedb
import pyarrow as pa

def get_table(persist_path: str = "./RAG/lancedb_store", table_name: str = "rag_chunks", dim: int = 768):
    db = lancedb.connect(persist_path)

    schema = pa.schema([
        pa.field("id", pa.string()),
        pa.field("text", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), dim)),
        pa.field("source_file", pa.string()),
        pa.field("source_type", pa.string()),
    ])

    table = db.create_table(table_name, exist_ok=True, schema=schema)
    return table


def store_chunks(chunks: list[dict], persist_path: str = "./RAG/lancedb_store", dim: int = 768):
    table = get_table(persist_path, dim=dim)

    rows = [{
        "id": c["id"],
        "text": c["text"],
        "vector": list(c["vector"]),
        "source_file": c["source_file"],
        "source_type": c["source_type"],
    } for c in chunks]

    (
        table.merge_insert("id")
        .when_not_matched_insert_all()   # only inserts new ids, skips existing ones entirely
        .execute(rows)
    )

    return table


def query(query_vector: list, top_k: int = 3, persist_path: str = "./RAG/lancedb_store"):
    table = get_table(persist_path)
    return table.search(query_vector).limit(top_k).to_list()


def drop(table_name: str = "rag_chunks", persist_path: str = "./lancedb_store"):
    db = lancedb.connect(persist_path)
    db.drop_table(table_name, ignore_missing=True)