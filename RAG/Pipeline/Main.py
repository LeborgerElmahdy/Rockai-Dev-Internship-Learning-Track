from RAG.Pipeline.Orchestrator import ingest_file, ask

#statistic / semantic
# print("Starting Ingestion Process...")
# ingest_file("RAG/Sample Files/sample2.docx", method = "semantic")
# print("Ingestion Complete!")

#question = "What software does sohayl use?"
#question = "What university did sohayl attend?"
question = "what are sohayl's projects?"

print("Query Request pending...")
results = ask(question, top_k = 3)

print(f"Query:{question} \nResults:")

for r in results:
    #print(r["id"], r["text"], r["metadata"], r["_distance"])
    print("-",r["text"], r["_distance"])
