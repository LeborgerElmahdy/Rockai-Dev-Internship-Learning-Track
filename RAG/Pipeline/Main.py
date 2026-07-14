from RAG.Pipeline.Parser import parse_file
from RAG.Pipeline.Chunker import semantic_chunk, statistic_chunk
from RAG.Pipeline.Gemini_client import call_function_with_handling

parsed = parse_file("RAG/Sample Files/sample.pdf")

chunked_stat = []
chunked_sem = []

for block in parsed:
    for chunk in statistic_chunk(block["text"]):
        chunked_stat.append({
            "text": chunk,
            "source_file": block["source_file"],
            "block_type": block["block_type"],
            "metadata": block["metadata"],
        })

for block in parsed:
    for chunk in semantic_chunk(block["text"]):
        chunked_sem.append({
            "text": chunk,
            "source_file": block["source_file"],
            "block_type": block["block_type"],
            "metadata": block["metadata"],
        })

print(f"[SEPARATOR_STATSTIC]{chunked_stat[2]["text"]}")
print(f"[SEPARATOR_SEMANTIC]{chunked_sem[2]["text"]}")
