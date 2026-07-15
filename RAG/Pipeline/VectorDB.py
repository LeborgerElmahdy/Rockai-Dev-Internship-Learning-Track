import lancedb


def get_table(persist_path: str = "./lancedb_store", table_name: str = "rag_chunks", dim: int = 768):
    db = lancedb.connect(persist_path)

    if table_name in db.table_names():
        return db.open_table(table_name)

    # create with a dummy row just to establish schema, then delete it
    schema_row = [{
        "id": "__init__",
        "text": "",
        "vector": [0.0] * dim,
        "source_file": "",
        "block_type": "",
        "metadata": "{}",
    }]
    table = db.create_table(table_name, data=schema_row)
    table.delete("id = '__init__'")
    return table


def store_chunks(chunks: list, persist_path: str = "./lancedb_store", dim: int = 768):
    import json

    table = get_table(persist_path, dim=dim)

    rows = [{
        "id": f"{c.source_file}-{i}",
        "text": c.text,
        "vector": list(c.vector),
        "source_file": c.source_file,
        "block_type": c.block_type,
        "metadata": json.dumps(c.metadata),
    } for i, c in enumerate(chunks)]

    table.add(rows)
    return table


def query(query_vector: list, top_k: int = 5, persist_path: str = "./lancedb_store"):
    table = get_table(persist_path)
    return table.search(query_vector).limit(top_k).to_list()

