import os

PROJECT_STRUCTURE = [
    "src/",
    "data/raw",
    "data/processed",
    "tests",
]

FILES = {
    "README.md": "# Transformers Learning Project\n",
    "requirements.txt": "",
    "src/day01/tokenizer_exploration.py": "# Day 1 tokenizer exploration\n",
    "tests/test_tokenizer.py": "# placeholder for tokenizer tests\n",
}

def create_structure():
    for path in PROJECT_STRUCTURE:
        os.makedirs(path, exist_ok=True)
        print(f"Created directory: {path}")

    for file_path, content in FILES.items():
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Created file: {file_path}")

if __name__ == "__main__":
    create_structure()