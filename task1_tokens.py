"""Task 1: compare whitespace splitting, Unicode word-like regex, and model subwords."""
import re

SAMPLES = [
    "AI, search!",
    "I can't log in.",
    "help@example.com",
    "CS4944",
    "",
    "یہ ایک مثال ہے",
]

def compare(text):
    whitespace = text.split()
    word_like = re.findall(r"\b\w+\b", text, flags=re.UNICODE)
    print(f"Original: {text!r}")
    print(f"Whitespace tokens ({len(whitespace)}): {whitespace}")
    print(f"Unicode word-like tokens ({len(word_like)}): {word_like}")
    print("-" * 60)

def model_tokenizer_examples():
    """Optional comparison; requires sentence-transformers and model download/cache."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Model tokenizer skipped: install requirements.txt to enable sentence-transformers.")
        return
    try:
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        for text in SAMPLES[:2]:
            print(f"Model tokenizer for {text!r}: {model.tokenizer.tokenize(text)}")
    except Exception as exc:
        print("Model tokenizer unavailable (model may need downloading):", repr(exc))

if __name__ == "__main__":
    for sample in SAMPLES:
        compare(sample)
    assert re.findall(r"\b\w+\b", "", flags=re.UNICODE) == []
    print("Two differences: whitespace splitting keeps punctuation attached and treats each space-delimited")
    print("piece as one token; the regex extracts word-like runs, splitting punctuation and email addresses.")
    print("Normalization risk: lowercasing or stripping punctuation may erase distinctions (for example,")
    print("case-sensitive identifiers such as CS4944). This regex is not a complete multilingual tokenizer.")
    model_tokenizer_examples()
