import nltk
import yaml
from pathlib import Path

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Download NLTK data
nltk.download("punkt", quiet=True)

# Configuration
INPUT_DIR = Path(config["data"]["extracted_text_dir"])
OUTPUT_DIR = Path(config["data"]["chunk_dir"])
CHUNK_SIZE = config["chunking"]["chunk_size"]
OVERLAP = config["chunking"]["overlap"]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def chunk_text(text, chunk_size, overlap):
    """Chunks text into smaller pieces of a target size."""
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
