import fitz
import yaml
from pathlib import Path

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

INPUT_DIR = Path(config["data"]["raw_pdf_dir"])
OUTPUT_DIR = Path(config["data"]["extracted_text_dir"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_text(pdf_path):
    """Extracts text from a single PDF file."""
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        pages.append(page.get_text())
    return "\n".join(pages)

for pdf in INPUT_DIR.glob("*.pdf"):
    text = extract_text(pdf)
    output_file = OUTPUT_DIR / f"{pdf.stem}.txt"
    output_file.write_text(text, encoding="utf-8")
    print(f"Extracted: {pdf.name}")
