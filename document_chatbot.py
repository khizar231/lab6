"""Interactive source-traceable evidence assistant; requires embeddings by default."""
import json
from pathlib import Path
from task2_chunks import make_chunks
from lab6_search import Retriever

def main():
    docs = json.loads(Path("corpus.json").read_text(encoding="utf-8"))
    strategy = input("Chunk strategy [sentence/0/10] (default sentence): ").strip() or "sentence"
    backend = input("Backend [embedding/tfidf] (default embedding): ").strip() or "embedding"
    try:
        engine = Retriever(make_chunks(docs, strategy), backend)
    except Exception as exc:
        print("Could not initialize selected retrieval backend:", repr(exc))
        print("For a quick local fallback, restart and choose tfidf.")
        return
    print("Evidence assistant. Type exit to stop.")
    while True:
        try:
            query = input("\\nQuestion: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if query.strip().casefold() == "exit":
            break
        hits = engine.search(query, k=3)
        if not hits:
            print("No candidate evidence; rephrase or contact a human.")
            continue
        print("Candidate evidence only — inspect before relying on it:")
        for n, (record, score) in enumerate(hits, 1):
            print(f'{n}. {record["id"]} score={score:.3f} source={record["doc"]} '
                  f'characters={record["start"]}:{record["end"]}')
        chosen = input("Choose 1-3 to inspect; press Enter if none apply: ").strip()
        if not chosen.isdigit() or not 1 <= int(chosen) <= len(hits):
            print("No answer selected. Rephrase or contact a human.")
            continue
        record, score = hits[int(chosen)-1]
        print("Retrieved passage:", record["text"])
        print("Source:", record["doc"], "characters", record["start"], record["end"])
        print("This is retrieved evidence, not an automatically verified answer.")

if __name__ == "__main__":
    main()
