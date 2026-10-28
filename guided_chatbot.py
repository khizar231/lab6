"""Small TF-IDF FAQ chatbot with exact-match and suggestion behavior."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

FAQS = [
    {"question": "How do I reset my password?",
     "answer": "Use the password recovery link on the sign-in page. Never share recovery codes.",
     "source": "it_help.txt"},
    {"question": "Where do I upload assignments?",
     "answer": "Upload assignments through the learning portal and submit the final file as a PDF.",
     "source": "course_guide.txt"},
    {"question": "What is required for an illness extension?",
     "answer": "Illness extensions require medical evidence before the deadline.",
     "source": "assessment_policy.txt"},
    {"question": "Can contractors work remotely?",
     "answer": "Contractors can work remotely only if the signed contract permits it.",
     "source": "contract_policy.txt"},
]

def build_bot():
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform([f["question"] for f in FAQS])
    return {"vectorizer": vectorizer, "matrix": matrix, "faqs": FAQS}

def reply(engine, question, threshold=0.20):
    if not question.strip():
        return "Please enter a question."
    normalized = question.strip().casefold()
    for faq in engine["faqs"]:
        if normalized == faq["question"].casefold():
            return f'{faq["answer"]} (Source: {faq["source"]})'
    qv = engine["vectorizer"].transform([question])
    scores = cosine_similarity(qv, engine["matrix"]).ravel()
    best = scores.argsort()[::-1][:2]
    if len(best) and scores[best[0]] >= threshold:
        suggestions = "\n".join(
            f'- {engine["faqs"][i]["question"]} (score={scores[i]:.3f})'
            for i in best if scores[i] >= threshold
        )
        return "Possible matches — please confirm which question you mean:\n" + suggestions
    return "No matching FAQ. Please rephrase or contact the course team."

if __name__ == "__main__":
    bot = build_bot()
    print("FAQ assistant. Type exit to stop.")
    while True:
        try:
            q = input("Question: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q.strip().casefold() == "exit":
            break
        print(reply(bot, q))
