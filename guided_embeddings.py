"""Run a minimal real-embedding retrieval demonstration."""
from task2_chunks import make_chunks
from lab6_search import Retriever, show_hits

DOCS = {
    "it": "A forgotten password can be reset using the recovery link. Never share recovery codes.",
    "assessment": "Assignments are uploaded through the learning portal. Submit the final file as a PDF.",
    "printer": "For blank printer pages, check ink levels and run the cleaning utility.",
}

if __name__ == "__main__":
    records = make_chunks(DOCS, "sentence")
    engine = Retriever(records, "embedding")
    print("Model load seconds:", engine.load_seconds)
    print("Corpus encoding seconds:", engine.encode_seconds)
    show_hits(engine.search("How can I regain access to my account?", k=3))
