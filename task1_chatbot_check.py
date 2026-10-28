"""Task 1 FAQ behavior checks and transcript examples."""
from guided_chatbot import build_bot, reply

def main():
    engine = build_bot()
    examples = [
        "How do I reset my password?",
        "password recovery",
        "",
        "volcano geology",
    ]
    for q in examples:
        print(f"INPUT: {q!r}")
        print("OUTPUT:", reply(engine, q))
        print()
    assert "it_help.txt" in reply(engine, "How do I reset my password?")
    assert reply(engine, "") == "Please enter a question."
    assert "No matching FAQ" in reply(engine, "volcano geology")
    assert "Possible matches" in reply(engine, "password recovery")
    print("FAQ behavior checks passed.")

if __name__ == "__main__":
    main()
