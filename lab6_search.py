"""Shared retrieval backends for TF-IDF and real SentenceTransformer embeddings."""
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Retriever:
    def __init__(self, records, backend="tfidf", model_name="sentence-transformers/all-MiniLM-L6-v2"):
        if backend not in {"tfidf", "embedding"}:
            raise ValueError("backend must be 'tfidf' or 'embedding'")
        if not records:
            raise ValueError("records must not be empty")
        self.records = records
        self.backend = backend
        self.model = None
        self.vectorizer = None
        self.load_seconds = 0.0
        self.encode_seconds = 0.0
        texts = [r["text"] for r in records]
        if backend == "tfidf":
            self.vectorizer = TfidfVectorizer()
            self.matrix = self.vectorizer.fit_transform(texts)
            self.load_seconds = 0.0
        else:
            from sentence_transformers import SentenceTransformer
            t = time.perf_counter()
            self.model = SentenceTransformer(model_name, device="cpu")
            self.load_seconds = time.perf_counter() - t
            t = time.perf_counter()
            self.matrix = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            self.encode_seconds = time.perf_counter() - t

    def search(self, query, k=3):
        if not query.strip():
            return []
        if self.backend == "tfidf":
            qv = self.vectorizer.transform([query])
            scores = cosine_similarity(qv, self.matrix).ravel()
        else:
            qv = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)
            scores = (qv @ self.matrix.T).ravel()
        order = np.argsort(-scores, kind="stable")[:k]
        return [(self.records[i], float(scores[i])) for i in order]

def show_hits(hits):
    for rank, (record, score) in enumerate(hits, 1):
        print(rank, record["id"], f"{score:.4f}",
              f'{record["doc"]}:{record["start"]}-{record["end"]}', record["text"])
