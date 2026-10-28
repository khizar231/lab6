"""Evaluate 3 chunking strategies x 2 retrieval backends; writes retrieval_log.csv."""
import csv, json, time
from pathlib import Path
import numpy as np
from task2_chunks import make_chunks
from lab6_search import Retriever

def evaluate(hits_by_query, items):
    answerable = [(h, q) for h, q in zip(hits_by_query, items) if q["doc"] is not None]
    if not answerable:
        return None, None
    hit_values, reciprocal = [], []
    for hits, q in answerable:
        good_ranks = [rank for rank, (c, _) in enumerate(hits, 1)
                      if c["doc"] == q["doc"] and q["evidence"] in c["text"]]
        hit_values.append(bool(good_ranks))
        reciprocal.append(1 / good_ranks[0] if good_ranks else 0.0)
    return float(np.mean(hit_values)), float(np.mean(reciprocal))

def validate(docs, items):
    assert len(docs) >= 5, "Need at least five documents"
    assert sum(q["split"] == "dev" and q["doc"] is not None for q in items) >= 4
    assert sum(q["split"] == "test" and q["doc"] is not None for q in items) >= 4
    assert sum(q["doc"] is None for q in items) >= 2
    for q in items:
        assert q["split"] in {"dev", "test"}
        if q["doc"] is not None:
            assert q["doc"] in docs and q["evidence"] in docs[q["doc"]]

def main():
    docs = json.loads(Path("corpus.json").read_text(encoding="utf-8"))
    items = json.loads(Path("queries.json").read_text(encoding="utf-8"))
    validate(docs, items)
    query_texts = [q["query"] for q in items]
    all_logs, summaries = [], []
    for strategy in ("0", "10", "sentence"):
        chunks = make_chunks(docs, strategy)
        texts = [c["text"] for c in chunks]
        for backend in ("tfidf", "embedding"):
            # Model initialization and corpus encoding are timed separately inside Retriever.
            t = time.perf_counter()
            try:
                engine = Retriever(chunks, backend)
            except Exception as exc:
                if backend == "embedding":
                    print("Embedding configuration unavailable:", repr(exc))
                    print("Install requirements and ensure model download/cache is available; TF-IDF runs continue.")
                    continue
                raise
            setup_seconds = time.perf_counter() - t
            model = engine.model
            if model is not None:
                lengths = [len(model.tokenizer(x, truncation=False)["input_ids"]) for x in texts + query_texts]
                max_len = max(lengths) if lengths else 0
                if max_len > model.max_seq_length:
                    raise ValueError(f"Model token limit exceeded: {max_len} > {model.max_seq_length}")
            else:
                max_len = None
            # Search each query and log top three text, scores, IDs, offsets, evidence labels.
            all_hits = []
            search_start = time.perf_counter()
            for q in items:
                hits = engine.search(q["query"], k=3)
                all_hits.append(hits)
            search_seconds = time.perf_counter() - search_start
            for split in ("dev", "test"):
                indices = [i for i, q in enumerate(items) if q["split"] == split]
                split_hits = [all_hits[i] for i in indices]
                split_items = [items[i] for i in indices]
                hit, mrr = evaluate(split_hits, split_items)
                summaries.append({
                    "strategy": strategy, "backend": backend, "split": split,
                    "chunk_count": len(chunks), "hit_at_3": hit, "mrr_at_3": mrr,
                    "setup_seconds": setup_seconds, "model_load_seconds": engine.load_seconds,
                    "corpus_encode_seconds": engine.encode_seconds,
                    "search_all_queries_seconds": search_seconds, "max_model_tokens": max_len
                })
                print("\\nSUMMARY", summaries[-1])
            for q, hits in zip(items, all_hits):
                for rank, (c, score) in enumerate(hits, 1):
                    all_logs.append({
                        "strategy": strategy, "backend": backend, "query": q["query"],
                        "split": q["split"], "answerable": q["doc"] is not None,
                        "rank": rank, "score": score, "chunk_id": c["id"], "doc": c["doc"],
                        "start": c["start"], "end": c["end"], "text": c["text"],
                        "gold_doc": q["doc"], "gold_evidence": q["evidence"],
                        "evidence_in_chunk": bool(q["evidence"] and q["evidence"] in c["text"])
                    })
    if not all_logs:
        raise RuntimeError("No retrieval results were produced.")
    with open("retrieval_log.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_logs[0]))
        writer.writeheader()
        writer.writerows(all_logs)
    with open("evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print("\\nWrote retrieval_log.csv and evaluation_summary.json")
    print("Unanswerable queries are logged but excluded from Hit@3/MRR@3 relevance metrics.")
    print("Inspect unanswerable-query scores separately; a top-ranked candidate is not proof of an answer.")

if __name__ == "__main__":
    main()
