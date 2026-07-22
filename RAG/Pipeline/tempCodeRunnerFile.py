print(f"QueryDB results:\n")
qr = orc.ask(question, top_k = 5)
for r in qr:
    print("-", r["id"], r["text"], r["_distance"])