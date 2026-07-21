import RAG.Pipeline.Orchestrator as orc

print("Dropping old Data..")
orc.drop_table()

#statistic / semantic
print("Starting Ingestion Process...")
orc.ingest_file("RAG/Sample Files/sample2.docx", method = "semantic")
print("Ingestion Complete!")

#question = "What software does sohayl use?"
#question = "Where is sohayls profile?"
question = "what are sohayl's projects?"

print(f"QueryDB results:\n")
qr = orc.ask(question, top_k = 5)
for r in qr:
#     #print(r["id"], r["text"], r["metadata"], r["_distance"])
    print("-",r["text"], r["_distance"])

print("QueryAPI Request pending...")
result = orc.generate_reply(question, top_k = 5)

print(f"Query:{question} \nResults:{result}")

