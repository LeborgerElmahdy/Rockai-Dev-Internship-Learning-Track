from RAG.Pipeline.Orchestrator import ingest_file, ask

#statistic / semantic
ingest_file("RAG/Sample Files/sample2.docx", method = "semantic")
print("Ingestion Complete!")
results = ask("What software does sohayl use?", top_k = 1)
print(len(results))
for r in results:
    print(r["text"], r["metadata"], r["_distance"])
