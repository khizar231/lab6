"""Compare paraphrased queries with TF-IDF and sentence embeddings."""
from task2_chunks import make_chunks
from lab6_search import Retriever, show_hits

DOCS = {
    "password": "A forgotten password can be reset using the recovery link. Never share recovery codes.",
    "assignment": "Assignments are uploaded through the learning portal. Submit the final file as a PDF.",
    "contractor": "Contractors can work remotely only if the signed contract permits it.",
}
QUERIES = [
    "How can I regain access to my account?",
    "Where should I hand in my coursework?",
    "Is working from home allowed for contractors?",
]

if __name__ == "__main__":
    records = make_chunks(DOCS, "sentence")
    for backend in ("tfidf", "embedding"):
        print("\\nBACKEND:", backend)
        engine = Retriever(records, backend)
        print("Model load seconds:", engine.load_seconds)
        for q in QUERIES:
            print("\\nQUERY:", q)
            show_hits(engine.search(q, k=3))
