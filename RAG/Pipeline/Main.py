from RAG.Pipeline.Orchestrator import ingest_file, ask

#statistic / semantic
ingest_file("RAG/Sample Files/sample2.docx", method = "semantic")
print("Ingestion Complete!")

question = "What software does sohayl use?"
results = ask(question, top_k = 1)

print(question)

for r in results:
    print(r["id"], r["text"], r["metadata"], r["_distance"])
