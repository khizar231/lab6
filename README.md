# Artificial Intelligence Lab Assignment — Tasks 1–3

## Scope
This package follows the uploaded lab manual: tokenization comparison, a TF-IDF FAQ chatbot, source-traceable chunking, TF-IDF versus real sentence embeddings, retrieval evaluation, and a confirmation-based evidence chatbot.

## Files
- `task1_tokens.py` — whitespace, Unicode word-like regex, and optional model tokenizer comparison.
- `guided_chatbot.py`, `task1_chatbot_check.py` — FAQ chatbot and repeatable checks.
- `task2_chunks.py` — 40-word windows with overlap 0/10 and sentence-aware packing, source offsets and IDs.
- `lab6_search.py` — TF-IDF and real SentenceTransformer retrieval backends.
- `guided_embeddings.py`, `guided_suggestions.py`, `task2_semantic_check.py` — guided semantic retrieval examples.
- `task3_retrieval.py` — six configuration evaluation, timing, token-length audit, top-three CSV log and JSON summary.
- `document_chatbot.py` — interactive source-evidence display with user confirmation.
- `corpus.json`, `queries.json` — five expanded synthetic documents and ten labeled queries.
- `Lab6_Report.docx` — report template with methodology and honest results-recording tables.

## Setup
Python 3.10+ is recommended. From this directory:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The first SentenceTransformer run may download `sentence-transformers/all-MiniLM-L6-v2`. Internet access or a pre-cached model is required for first-time download. If unavailable, Task 1 and TF-IDF modes can still run; the semantic-embedding requirement must be completed when the model is available.

## Run tasks
```bash
python task1_tokens.py
python task1_chatbot_check.py
python guided_chatbot.py
python -c "from task2_chunks import self_test; print(self_test())"
python guided_embeddings.py
python guided_suggestions.py
python task2_semantic_check.py
python task3_retrieval.py
python document_chatbot.py
```

`task3_retrieval.py` writes `retrieval_log.csv` and `evaluation_summary.json` in the current directory. Preserve these actual outputs with your submission. The script prints measured timings and metrics from the machine on which it is run.

## Evaluation protocol
- The corpus has five synthetic documents, each written to be roughly 150–250 words; verify lengths with your own script before submission.
- Four answerable queries are development examples and four are held-out test examples. Two queries are labeled unanswerable.
- `Hit@3` is 1 when at least one top-three chunk contains the gold document and exact evidence substring. `MRR@3` is the reciprocal rank of the first such chunk, or 0 if absent.
- Unanswerable questions are excluded from answerable Hit@3/MRR@3; their top candidates are still logged for false-positive analysis.
- Compare `0`, `10`, and `sentence` chunk strategies for both `tfidf` and `embedding`. Choose a strategy based on development results only, then report its held-out test result without retuning.
- A high score for an unanswerable query does not establish answerability: retrieval ranks the best available candidates even when none is relevant. Calibrate any rejection threshold on separate development examples; do not assume a universal cosine threshold.

## Notes and limitations
- Sentence splitting is intentionally a teaching-level punctuation/whitespace heuristic. It can split incorrectly around abbreviations (e.g., `Dr. Ali`), decimal numbers, initials, or punctuation inside quotations. It is not a complete multilingual sentence segmenter.
- `\b\w+\b` is a word-like regex, not a complete multilingual tokenizer. It may split contractions and email addresses and treats many Unicode letters as word characters, but token boundaries are language-dependent.
- The synthetic corpus is for reproducible coursework practice, not an authoritative policy source.
- Do not fabricate execution output, performance results, or a winning strategy. Run the commands, inspect failures, and fill the report with the output from your own machine.
- Keep virtual environments, model caches, credentials, and private data out of the repository.
