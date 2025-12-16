import nltk
from pathlib import Path

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

INPUT_DIR = Path("data/extracted_text")
OUTPUT_DIR = Path("data/chunks")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 400
OVERLAP = 50

def chunk_text(text):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current = []
    word_count = 0

    for sentence in sentences:
        words = sentence.split()
        current.append(sentence)
        word_count += len(words)

        if word_count >= CHUNK_SIZE:
            chunks.append(" ".join(current))
            current = current[-OVERLAP:]
            word_count = sum(len(s.split()) for s in current)

    if current:
        chunks.append(" ".join(current))

    return chunks

for file in INPUT_DIR.glob("*.txt"):
    text = file.read_text(encoding="utf-8")
    chunks = chunk_text(text)

    out_file = OUTPUT_DIR / f"{file.stem}_chunks.txt"
    out_file.write_text("\n\n---\n\n".join(chunks), encoding="utf-8")

    print(f"{file.name}: {len(chunks)} chunks")
