"""Task 2 wiring check; embedding backend requires model availability."""
from task2_chunks import make_chunks
from lab6_search import Retriever, show_hits

TEXTS = {
    "it": "A forgotten password can be reset using the recovery link. Never share recovery codes.",
    "assessment": "Assignments are uploaded through the learning portal. Submit the final file as a PDF.",
}
if __name__ == "__main__":
    records = make_chunks(TEXTS, "sentence")
    for backend in ("tfidf", "embedding"):
        print("Backend:", backend)
        engine = Retriever(records, backend)
        print("Model load seconds:", engine.load_seconds)
        show_hits(engine.search("How can I regain access to my account?", k=2))
