"""Task 2: source-traceable word windows and sentence-aware chunking."""
import re

def word_spans(text):
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]

def windows(text, size=40, overlap=10, offset=0):
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Require size > 0 and 0 <= overlap < size")
    spans = word_spans(text)
    result = []
    step = size - overlap
    for i in range(0, len(spans), step):
        j = min(i + size, len(spans))
        result.append((offset + spans[i][0], offset + spans[j - 1][1]))
        if j == len(spans):
            break
    return result

def sentence_chunks(text, size=40, offset=0):
    """Teaching splitter: punctuation followed by whitespace; long sentences fall back to word windows."""
    if size <= 0:
        raise ValueError("size must be positive")
    cuts = [0] + [m.end() for m in re.finditer(r"[.!?]\s+", text)] + [len(text)]
    out, pending, count = [], None, 0
    for a, b in zip(cuts, cuts[1:]):
        segment = text[a:b]
        n = len(segment.split())
        if not n:
            continue
        if n > size:
            if pending is not None:
                out.append((offset + pending[0], offset + pending[1]))
            out.extend(windows(segment, size=size, overlap=0, offset=offset + a))
            pending, count = None, 0
        elif pending is not None and count + n <= size:
            pending, count = (pending[0], b), count + n
        else:
            if pending is not None:
                out.append((offset + pending[0], offset + pending[1]))
            pending, count = (a, b), n
    if pending is not None:
        out.append((offset + pending[0], offset + pending[1]))
    return out

def make_chunks(docs, strategy, size=40):
    result = []
    for doc_id, text in docs.items():
        if strategy == "sentence":
            spans = sentence_chunks(text, size=size)
        elif strategy in ("0", "10"):
            spans = windows(text, size=size, overlap=int(strategy))
        else:
            raise ValueError("strategy must be '0', '10', or 'sentence'")
        for i, (a, b) in enumerate(spans):
            result.append({"id": f"{doc_id}:{i}", "doc": doc_id,
                           "start": a, "end": b, "text": text[a:b]})
    return result

def word_count(text):
    return len(text.split())

def self_test():
    assert windows("") == []
    assert windows("short text") == [(0, 10)]
    try:
        windows("a b", 2, 2)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid overlap was accepted")
    long_sentence = " ".join(["word"] * 95) + "."
    assert all(word_count(long_sentence[a:b]) <= 40
               for a, b in sentence_chunks(long_sentence))
    docs = {"example": "Reset your password. Keep recovery codes private."}
    for strategy in ("0", "10", "sentence"):
        for c in make_chunks(docs, strategy):
            assert c["text"] == docs[c["doc"]][c["start"]:c["end"]]
    return "Chunking boundary checks passed."

if __name__ == "__main__":
    print(self_test())
    sample = {"policy": " ".join(["A short synthetic policy sentence."] * 12)}
    for strategy in ("0", "10", "sentence"):
        chunks = make_chunks(sample, strategy)
        print(strategy, "chunks:", len(chunks))
        for c in chunks:
            print(c["id"], c["start"], c["end"], word_count(c["text"]))
