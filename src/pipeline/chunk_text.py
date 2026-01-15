
import yaml
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- CONFIGURATION ---
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

INPUT_DIR = Path(config["data"]["extracted_text_dir"])
OUTPUT_DIR = Path(config["data"]["chunk_dir"])
CHUNK_SIZE = config["chunking"]["chunk_size"]
OVERLAP = config["chunking"]["overlap"]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- INITIALIZE TEXT SPLITTER ---
# This splitter is semantically aware. It tries to split on paragraphs ("\n\n"),
# then sentences ("."), then newlines ("\n"), and finally words, to keep
# related text together in the same chunk.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=OVERLAP,
    length_function=len,
)

# --- MAIN EXECUTION ---
print("Starting text chunking with RecursiveCharacterTextSplitter...")
# Ensure there are files to process
if not any(INPUT_DIR.glob("*.txt")):
    print(f"No text files found in {INPUT_DIR}. Skipping chunking.")
else:
    for file in INPUT_DIR.glob("*.txt"):
        print(f"  - Chunking: {file.name}")
        text = file.read_text(encoding="utf-8")

        # The splitter returns a list of strings.
        chunks = text_splitter.split_text(text)

        # Define output file path, e.g., "doc1.txt" -> "doc1_chunks.txt"
        out_file = OUTPUT_DIR / f"{file.stem}_chunks.txt"
        out_file.write_text("\n\n---\n\n".join(chunks), encoding="utf-8")

        print(f"    -> Created {len(chunks)} chunks.")

print("\nText chunking completed.")
