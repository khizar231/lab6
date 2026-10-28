"""Validate that corpus.json and queries.json exist; does not overwrite files."""
import json
from pathlib import Path

def main():
    for name in ("corpus.json", "queries.json"):
        p = Path(name)
        if not p.exists():
            raise FileNotFoundError(f"{name} is missing. Run from the project folder.")
        data = json.loads(p.read_text(encoding="utf-8"))
        print(name, "loaded:", len(data), "top-level records")
    print("Dataset files are present and valid JSON.")
if __name__ == "__main__":
    main()
