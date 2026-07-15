from RAG.Pipeline.Parser import parse_file
from RAG.Pipeline.Chunker import chunk_blocks
from RAG.Pipeline.Gemini_client import embed, embed_normalized, generate
from RAG.Pipeline.VectorDB import store_chunks, query

parsed = parse_file("RAG/Sample Files/sample.csv", rows_per_block = 1)

# statistic / semantic
chunks = chunk_blocks(parsed, method = "semantic")


texts = [c.text for c in chunks]
embedding_result = embed(texts)

for chunk, embedding in zip(chunks, embedding_result.embeddings):
    chunk.vector = embedding.values

store_chunks(chunks)

query_vector = embed(["What is Fatima's department?"]).embeddings[0].values
results = query(query_vector, top_k=3)

for r in results:
    print(r["text"], r["metadata"], r["_distance"])


# for i in range (0, len(parsed)):
#      print(f"-{parsed[i]["text"]}\n")

# for i in range (0, len(chunks)):
#      print(f"-{chunks[i].vector}\n")

# print(parsed[0]["text"], "\n")
# print(chunks[0].text,    "\n")
# print(chunks[0].vector,  "\n")