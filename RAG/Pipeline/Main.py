from RAG.Pipeline.Parser import parse_file
from RAG.Pipeline.Chunker import chunk_blocks
from RAG.Pipeline.Gemini_client import embed, embed_normalized, generate

parsed = parse_file("RAG/Sample Files/sample.txt", rows_per_block = 1)

# statistic / semantic
chunks = chunk_blocks(parsed, method = "semantic")


texts = [c.text for c in chunks]
embedding_result = embed(texts)

for chunk, embedding in zip(chunks, embedding_result.embeddings):
    chunk.vector = embedding.values

# for i in range (0, len(parsed)):
#      print(f"-{parsed[i]["text"]}\n")

# for i in range (0, len(chunks)):
#      print(f"-{chunks[i].vector}\n")

print(parsed[0]["text"], "\n")

print(chunks[0].text, "\n")
print(chunks[0].vector, "\n")